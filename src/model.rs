use crate::config::Config;
use anyhow::Result;
use ndarray::{Array, Axis, Ix2, s};
use opencv::{
    core::{Mat, MatTraitConst, Rect, Size, VecN},
    imgproc::{InterpolationFlags, resize},
};
use ort::{
    execution_providers::{CPUExecutionProvider, TensorRTExecutionProvider},
    session::{Session, builder::GraphOptimizationLevel},
};
use std::time::Instant;

const CXYWH_OFFSET: usize = 4;

pub struct Model {
    session: Session,
    input_name: String,
    output_name: String,
    input_size: usize,
    conf: f32,
    iou: f32,
    roi: Rect,
}

impl Model {
    pub fn new(config: Config) -> Result<Self> {
        let providers = match config.model_provider.as_str() {
            "TensorRT" | "tensorrt" | "trt" => vec![
                TensorRTExecutionProvider::default()
                    .with_device_id(config.gpu_id.unwrap_or(0))
                    .with_engine_cache(true)
                    .with_engine_cache_path(config.trt_cache_dir)
                    .with_profile_min_shapes(config.trt_min_shapes)
                    .with_profile_opt_shapes(config.trt_opt_shapes)
                    .with_profile_max_shapes(config.trt_max_shapes)
                    .with_max_partition_iterations(
                        config.trt_max_partition_iterations.unwrap_or(10),
                    )
                    .with_max_workspace_size(config.gpu_mem_limit.unwrap_or(1024 * 1024 * 1024))
                    .with_fp16(config.trt_fp16.unwrap_or(false))
                    // allow value from [0, 5]
                    // levels below 3 do not guarantee good engine performance, but greatly improve build time
                    .with_builder_optimization_level(
                        config.trt_builder_optimization_level.unwrap_or(3),
                    )
                    .with_dla(config.trt_dla_enable.unwrap_or(false))
                    .with_dla_core(config.trt_dla_core.unwrap_or(0))
                    .with_auxiliary_streams(config.trt_auxiliary_streams.unwrap_or(-1))
                    .build(),
            ],
            _ => vec![
                CPUExecutionProvider::default()
                    .with_arena_allocator()
                    .build(),
            ],
        };
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(providers)?
            .with_intra_threads(1)?
            .with_independent_thread_pool()?
            .commit_from_file(config.model_path)?;
        let input_name = session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect::<Vec<_>>()
            .pop()
            .unwrap();
        let output_name = session
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect::<Vec<_>>()
            .pop()
            .unwrap();
        Ok(Self {
            session,
            input_name,
            output_name,
            input_size: config.model_input_size,
            conf: config.model_conf,
            iou: config.model_iou,
            roi: Rect::new(
                config.region_left as i32,
                config.region_top as i32,
                config.region_width as i32,
                config.region_height as i32,
            ),
        })
    }
}

impl Model {
    #[inline]
    pub fn infer(&self, mat: &Mat) -> Result<Vec<Bbox>> {
        // preprocess
        let pre_time = Instant::now();
        let mut inputs =
            Array::<f32, _>::from_elem((1, 3, self.input_size, self.input_size), 114. / 255.)
                .into_dyn();
        let input = Mat::roi(mat, self.roi)?.clone_pointee();
        let (w0, h0) = (input.cols() as f32, input.rows() as f32);
        let (ratio, w_new, h_new) = self.scale_wh(w0, h0);
        let (w_new, h_new) = (w_new as i32, h_new as i32);
        let mut img = Mat::default();
        let _ = resize(
            &input,
            &mut img,
            Size::new(w_new, h_new),
            0f64,
            0f64,
            InterpolationFlags::INTER_LINEAR as i32,
        )?;
        let dh = (self.input_size - h_new as usize) / 2;
        let dw = (self.input_size - w_new as usize) / 2;

        for row in 0..img.rows() as usize {
            for col in 0..img.cols() as usize {
                let v = img.at_2d::<VecN<u8, 3>>(row as i32, col as i32)?;
                inputs[[0, 0, row + dh, col + dw]] = (v.0[2] as f32) / 255.0;
                inputs[[0, 1, row + dh, col + dw]] = (v.0[1] as f32) / 255.0;
                inputs[[0, 2, row + dh, col + dw]] = (v.0[0] as f32) / 255.0;
            }
        }
        let pre_time = pre_time.elapsed();

        // inference
        let infer_time = Instant::now();
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => inputs]?)?;
        let infer_time = infer_time.elapsed();

        // postprocess
        let post_time = Instant::now();
        let outputs = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()?
            .remove_axis(Axis(0))
            .into_dimensionality::<Ix2>()?;
        let mut bboxes: Vec<Bbox> = Vec::new();
        for pred in outputs.axis_iter(Axis(1)) {
            // confidence filter
            let score = pred.slice(s![CXYWH_OFFSET..CXYWH_OFFSET + 1])[0];
            if score < self.conf {
                continue;
            }
            let bbox = pred.slice(s![0..CXYWH_OFFSET]);
            let class = pred.slice(s![CXYWH_OFFSET + 1..CXYWH_OFFSET + 2])[0] as u8;

            // bbox re-scale
            let cx = bbox[0] / ratio;
            let cy = bbox[1] / ratio;
            let w = bbox[2] / ratio;
            let h = bbox[3] / ratio;
            let x = cx - w / 2. - dw as f32 / ratio + self.roi.x as f32;
            let y = cy - h / 2. - dh as f32 / ratio + self.roi.y as f32;
            let bbox =
                Bbox::new(x, y, w, h, score, class).bound(mat.cols() as f32, mat.rows() as f32);
            bboxes.push(bbox);
        }
        self.non_max_suppression(&mut bboxes);
        let post_time = post_time.elapsed();
        tracing::debug!(
            "[Model] preprocess took: {:?}, infer took: {:?}, postprocess took: {:?}, total took: {:?}",
            pre_time,
            infer_time,
            post_time,
            pre_time + infer_time + post_time,
        );

        Ok(bboxes)
    }

    #[inline]
    fn scale_wh(&self, w0: f32, h0: f32) -> (f32, f32, f32) {
        let r = (self.input_size as f32 / w0).min(self.input_size as f32 / h0);
        (r, (w0 * r).round(), (h0 * r).round())
    }

    #[inline]
    fn non_max_suppression(&self, xs: &mut Vec<Bbox>) {
        xs.sort_by(|b1, b2| b2.confidence().partial_cmp(&b1.confidence()).unwrap());

        let mut current_index = 0;
        for index in 0..xs.len() {
            let mut drop = false;
            for prev_index in 0..current_index {
                let iou = xs[prev_index].iou(&xs[index]);
                if iou > self.iou {
                    drop = true;
                    break;
                }
            }
            if !drop {
                xs.swap(current_index, index);
                current_index += 1;
            }
        }
        xs.truncate(current_index);
    }
}

#[derive(Debug, Clone, Default, Copy)]
pub struct Point2f {
    x: f32,
    y: f32,
}

impl Point2f {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn x(&self) -> f32 {
        self.x
    }

    #[inline]
    pub fn y(&self) -> f32 {
        self.y
    }

    #[inline]
    pub fn to_vec_u32(&self) -> Vec<u32> {
        vec![self.x as u32, self.y as u32]
    }

    #[inline]
    pub fn l2_distance(&self, point: &Self) -> f32 {
        (self.x - point.x) * (self.x - point.x) + (self.y - point.y) * (self.y - point.y)
    }
}
#[derive(Debug, Clone, Copy, Default)]
pub struct Bbox {
    xmin: f32,
    ymin: f32,
    width: f32,
    height: f32,
    confidence: f32,
    class: u8,
}

impl Bbox {
    #[inline]
    pub fn new_from_xywh(xmin: f32, ymin: f32, width: f32, height: f32) -> Self {
        Self {
            xmin,
            ymin,
            width,
            height,
            ..Default::default()
        }
    }

    #[inline]
    pub fn new(xmin: f32, ymin: f32, width: f32, height: f32, confidence: f32, class: u8) -> Self {
        Self {
            xmin,
            ymin,
            width,
            height,
            confidence,
            class,
        }
    }

    #[inline]
    pub fn to_vec_i32(&self) -> Vec<i32> {
        vec![
            self.xmin as i32,
            self.ymin as i32,
            self.width as i32,
            self.height as i32,
        ]
    }

    pub fn width(&self) -> f32 {
        self.width
    }

    pub fn height(&self) -> f32 {
        self.height
    }

    pub fn xmin(&self) -> f32 {
        self.xmin
    }

    pub fn ymin(&self) -> f32 {
        self.ymin
    }

    pub fn xmax(&self) -> f32 {
        self.xmin + self.width
    }

    pub fn ymax(&self) -> f32 {
        self.ymin + self.height
    }

    pub fn tl(&self) -> Point2f {
        Point2f::new(self.xmin, self.ymin)
    }

    pub fn br(&self) -> Point2f {
        Point2f::new(self.xmax(), self.ymax())
    }

    pub fn cxcy_scale(&self, scale_x: Option<f32>, scale_y: Option<f32>) -> Point2f {
        Point2f::new(
            self.xmin + self.width / 2. * scale_x.unwrap_or(1.),
            self.ymin + self.height / 2. * scale_y.unwrap_or(1.),
        )
    }

    pub fn cxcy(&self) -> Point2f {
        Point2f::new(self.xmin + self.width / 2., self.ymin + self.height / 2.)
    }

    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    pub fn class(&self) -> u8 {
        self.class
    }

    #[inline]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    #[inline]
    pub fn intersection_area(&self, another: &Bbox) -> f32 {
        let l = self.xmin.max(another.xmin);
        let r = (self.xmin + self.width).min(another.xmin + another.width);
        let t = self.ymin.max(another.ymin);
        let b = (self.ymin + self.height).min(another.ymin + another.height);
        (r - l + 1.).max(0.) * (b - t + 1.).max(0.)
    }

    #[inline]
    pub fn union(&self, another: &Bbox) -> f32 {
        self.area() + another.area() - self.intersection_area(another)
    }

    #[inline]
    pub fn iou(&self, another: &Bbox) -> f32 {
        self.intersection_area(another) / self.union(another)
    }

    #[inline]
    pub fn bound(&self, bound_width: f32, bound_height: f32) -> Self {
        let xmin = self.xmin.max(0.0f32).min(bound_width);
        let ymin = self.ymin.max(0.0f32).min(bound_height);
        let width = (self.width + xmin).min(bound_width) - xmin;
        let height = (self.height + ymin).min(bound_height) - ymin;
        Self {
            xmin,
            ymin,
            width,
            height,
            confidence: self.confidence,
            class: self.class,
        }
    }
}
