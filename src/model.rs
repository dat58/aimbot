use crate::config::{Config, SCALE_HEAD_X, SCALE_HEAD_Y};
use anyhow::{Result, bail};
use ndarray::{Array, Axis, Ix2, s};
use opencv::{
    core::{Mat, MatTraitConst, MatTraitConstManual, Rect, Size, VecN},
    imgproc::{InterpolationFlags, resize},
};
use ort::{
    execution_providers::{
        CPUExecutionProvider, MIGraphXExecutionProvider, OpenVINOExecutionProvider,
        ROCmExecutionProvider, TensorRTExecutionProvider,
    },
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::{Session, SessionInputValue, builder::GraphOptimizationLevel},
    value::TensorRefMut,
};
use std::borrow::Cow;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::time::Instant;

const CXYWH_OFFSET: usize = 4;

pub struct Model {
    session: Session,
    input_name: String,
    output_name: String,
    input_size: usize,
    conf: [f32; 2],
    iou: f32,
    roi: Rect,
    crop: bool,
    /// `v_light_*` model: nearest-neighbour stretch preprocess and a raw
    /// multi-class output instead of the letterboxed two-class path.
    light: bool,
    /// Persistent NCHW input buffer for the light path. ONNX Runtime borrows it
    /// in place, so steady-state inference neither allocates nor copies it.
    light_input: RefCell<Vec<f32>>,
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
            "Migraphx" | "migraphx" | "mrx" => vec![
                MIGraphXExecutionProvider::default()
                    .with_exhaustive_tune(true)
                    .with_device_id(config.gpu_id.unwrap_or(0))
                    .build(),
            ],
            "Rocm" | "rocm" => vec![
                ROCmExecutionProvider::default()
                    .with_device_id(config.gpu_id.unwrap_or(0))
                    .with_tuning(true)
                    .with_mem_limit(config.gpu_mem_limit.unwrap_or(1024 * 1024 * 1024))
                    .with_hip_graph(true)
                    .build(),
            ],
            "OpenVino" | "openvino" => vec![
                OpenVINOExecutionProvider::default()
                    .with_device_id(config.gpu_id.unwrap_or(0))
                    .with_cache_dir(&config.openvino_cache_dir)
                    // allowed [CPU, GPU, NPU, GPU.0, GPU.1, ...]
                    .with_device_type(&config.openvino_device_type)
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
            .with_intra_threads(config.intra_threads)?
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
        let crop = config.screen_width != config.region_width
            || config.screen_height != config.region_height;
        let light = config.light_model;
        tracing::info!(
            "[Model] {} pipeline: {}x{} input",
            if light { "v_light" } else { "letterbox" },
            config.model_input_size,
            config.model_input_size
        );
        let light_input = RefCell::new(if light {
            vec![0.0f32; 3 * config.model_input_size * config.model_input_size]
        } else {
            Vec::new()
        });
        Ok(Self {
            session,
            input_name,
            output_name,
            input_size: config.model_input_size,
            conf: [config.model_conf_body, config.model_conf_head],
            iou: config.model_iou,
            roi: Rect::new(
                config.region_left as i32,
                config.region_top as i32,
                config.region_width as i32,
                config.region_height as i32,
            ),
            crop,
            light,
            light_input,
        })
    }
}

impl Model {
    #[inline]
    pub fn infer(&self, mat: &Mat) -> Result<Bboxes> {
        if self.light {
            self.infer_light(mat)
        } else {
            self.infer_letterbox(mat)
        }
    }

    /// Region of the frame the model looks at, clamped to the frame.
    fn source_roi(&self, mat: &Mat) -> Result<Rect> {
        let (w, h) = (mat.cols(), mat.rows());
        if w <= 0 || h <= 0 {
            bail!("[Model] empty frame");
        }
        if !self.crop {
            return Ok(Rect::new(0, 0, w, h));
        }
        let roi = self.roi;
        if roi.x < 0
            || roi.y < 0
            || roi.width <= 0
            || roi.height <= 0
            || roi.x + roi.width > w
            || roi.y + roi.height > h
        {
            bail!(
                "[Model] region {}x{}+{}+{} does not fit inside the {}x{} frame",
                roi.width,
                roi.height,
                roi.x,
                roi.y,
                w,
                h
            );
        }
        Ok(roi)
    }

    /// Nearest-neighbour stretch of the region into an RGB CHW f32 buffer,
    /// normalized by 1/255. No letterbox and no padding, matching capfkaplus:
    /// each axis gets its own scale and the decoder undoes it with the inverse
    /// per-axis factor. The whole model input carries image, none of it
    /// padding, at the cost of distorting a non-square region.
    fn preprocess_light(&self, mat: &Mat, roi: Rect, out: &mut [f32]) -> Result<()> {
        const INV_255: f64 = 1.0 / 255.0;
        let size = self.input_size;

        // A continuous frame is read straight through, region included; a
        // non-continuous one needs one crop copy.
        let cropped;
        let (data, stride, base) = if mat.is_continuous() {
            let stride = mat.cols() as usize * 3;
            let base = roi.y as usize * stride + roi.x as usize * 3;
            (mat.data_bytes()?, stride, base)
        } else {
            cropped = Mat::roi(mat, roi)?.clone_pointee();
            (cropped.data_bytes()?, roi.width as usize * 3, 0usize)
        };

        let (rw, rh) = (roi.width as usize, roi.height as usize);
        let sx = rw as f64 / size as f64;
        let sy = rh as f64 / size as f64;
        let plane = size * size;

        for y in 0..size {
            let row = base + ((y as f64 * sy) as usize).min(rh - 1) * stride;
            for x in 0..size {
                let p = row + ((x as f64 * sx) as usize).min(rw - 1) * 3;
                let idx = y * size + x;
                // BGR source, RGB output.
                out[idx] = (data[p + 2] as f64 * INV_255) as f32;
                out[plane + idx] = (data[p + 1] as f64 * INV_255) as f32;
                out[2 * plane + idx] = (data[p] as f64 * INV_255) as f32;
            }
        }
        Ok(())
    }

    /// Decode the raw `[1, 4 + classes, N]` light output. Channels are
    /// `cx, cy, w, h` in model-input pixels followed by one score per class —
    /// no objectness channel and no NMS inside the graph.
    ///
    /// The `v_light_*` models emit five classes (body, head, tm, ability,
    /// flash). Only body and head are targets, and they keep the same 0/1
    /// meaning the two-class models use, so everything downstream is unchanged.
    fn decode_light(&self, shape: &[i64], values: &[f32], roi: Rect, mat: &Mat) -> Result<Bboxes> {
        if shape.len() != 3 || shape[0] != 1 || shape[1] < 5 {
            bail!("[Model] expected a [1,C,N] light output, got {:?}", shape);
        }
        let channels = shape[1] as usize;
        let candidates = shape[2] as usize;
        if values.len() < channels * candidates {
            bail!(
                "[Model] light output truncated: {} values for {}x{}",
                values.len(),
                channels,
                candidates
            );
        }

        let sx = roi.width as f32 / self.input_size as f32;
        let sy = roi.height as f32 / self.input_size as f32;
        let (bound_w, bound_h) = (mat.cols() as f32, mat.rows() as f32);
        let mut bboxes = Bboxes::default();

        for i in 0..candidates {
            // Argmax over every class, not just the first two: that way a
            // teammate or ability box wins its own candidate and is skipped,
            // instead of being read as a body at whatever score it happens to
            // have on channel 4.
            let mut best = f32::NEG_INFINITY;
            let mut class = 0usize;
            for c in CXYWH_OFFSET..channels {
                let score = values[c * candidates + i];
                if score > best {
                    best = score;
                    class = c - CXYWH_OFFSET;
                }
            }
            if class > 1 || best < self.conf[class] {
                continue;
            }

            // bbox re-scale: inverse of the per-axis stretch
            let cx = values[i] * sx;
            let cy = values[candidates + i] * sy;
            let w = values[2 * candidates + i] * sx;
            let h = values[3 * candidates + i] * sy;
            let x = cx - w / 2. + roi.x as f32;
            let y = cy - h / 2. + roi.y as f32;
            let bbox = Bbox::new(x, y, w, h, best, class as u8).bound(bound_w, bound_h);
            bboxes.push(bbox, class);
        }

        self.non_max_suppression(&mut bboxes.class_0);
        self.non_max_suppression(&mut bboxes.class_1);
        Ok(bboxes)
    }

    #[inline]
    fn infer_light(&self, mat: &Mat) -> Result<Bboxes> {
        if mat.channels() != 3 {
            bail!(
                "[Model] expected a 3-channel BGR frame, got {} channels",
                mat.channels()
            );
        }
        let roi = self.source_roi(mat)?;

        // preprocess
        let pre_time = Instant::now();
        let mut input = self.light_input.borrow_mut();
        self.preprocess_light(mat, roi, &mut input)?;
        let pre_time = pre_time.elapsed();

        // inference
        let infer_time = Instant::now();
        let mem = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Arena,
            MemoryType::CPUInput,
        )?;
        // Borrows the persistent buffer rather than handing ORT a fresh copy.
        let tensor = unsafe {
            TensorRefMut::<f32>::from_raw(
                mem,
                input.as_mut_ptr().cast(),
                vec![1, 3, self.input_size as i64, self.input_size as i64],
            )
        }?;
        let inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
            vec![(Cow::Borrowed(self.input_name.as_str()), tensor.into())];
        let outputs = self.session.run(inputs)?;
        let infer_time = infer_time.elapsed();

        // postprocess
        let post_time = Instant::now();
        let (shape, values) = outputs[self.output_name.as_str()].try_extract_raw_tensor::<f32>()?;
        let bboxes = self.decode_light(shape, values, roi, mat)?;
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
    fn infer_letterbox(&self, mat: &Mat) -> Result<Bboxes> {
        // preprocess
        let pre_time = Instant::now();
        let mut inputs =
            Array::<f32, _>::from_elem((1, 3, self.input_size, self.input_size), 114. / 255.)
                .into_dyn();
        let input = if self.crop {
            std::borrow::Cow::Owned(Mat::roi(mat, self.roi)?.clone_pointee())
        } else {
            std::borrow::Cow::Borrowed(mat)
        };
        let (w0, h0) = (input.cols() as f32, input.rows() as f32);
        let (ratio, w_new, h_new) = self.scale_wh(w0, h0);
        let (w_new, h_new) = (w_new as i32, h_new as i32);
        let mut img = Mat::default();
        let _ = resize(
            input.as_ref(),
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
        let mut bboxes = Bboxes::default();
        for pred in outputs.axis_iter(Axis(1)) {
            // confidence filter
            let scores = pred.slice(s![CXYWH_OFFSET..CXYWH_OFFSET + 2]);
            let class = if scores[0] > scores[1] { 0 } else { 1 };
            if scores[class] < self.conf[class] {
                continue;
            }
            let bbox = pred.slice(s![0..CXYWH_OFFSET]);

            // bbox re-scale
            let cx = bbox[0] / ratio;
            let cy = bbox[1] / ratio;
            let w = bbox[2] / ratio;
            let h = bbox[3] / ratio;
            let x = cx - w / 2. - dw as f32 / ratio + self.roi.x as f32;
            let y = cy - h / 2. - dh as f32 / ratio + self.roi.y as f32;
            let bbox = Bbox::new(x, y, w, h, scores[class], class as u8)
                .bound(mat.cols() as f32, mat.rows() as f32);
            bboxes.push(bbox, class);
        }
        self.non_max_suppression(&mut bboxes.class_0);
        self.non_max_suppression(&mut bboxes.class_1);
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

#[derive(Debug, Clone, Default)]
pub struct Bboxes {
    pub class_0: Vec<Bbox>,
    pub class_1: Vec<Bbox>,
}

impl Bboxes {
    pub fn push(&mut self, bbox: Bbox, class: usize) {
        if class == 0 {
            self.class_0.push(bbox);
        } else {
            self.class_1.push(bbox);
        }
    }

    pub fn build(&mut self, iou: f32) {
        let mut class_1 = Vec::with_capacity(self.class_0.len());
        let mut removed = Vec::with_capacity(self.class_1.len());
        for i in 0..self.class_0.len() {
            let mut max_iou = 0.0;
            let mut max_index = -1;
            for j in 0..self.class_1.len() {
                if !removed.contains(&j) {
                    let current_iou = self.class_0[i].iou(&self.class_1[j]);
                    if current_iou >= iou && current_iou > max_iou {
                        max_index = j as i32;
                        max_iou = current_iou;
                    }
                }
            }
            if max_index != -1 {
                class_1.push(self.class_1[max_index as usize]);
                removed.push(max_index as usize);
            } else {
                let xmin =
                    (1. - SCALE_HEAD_X) / 2. * self.class_0[i].width() + self.class_0[i].xmin();
                let ymin = self.class_0[i].ymin();
                let width = self.class_0[i].width() * SCALE_HEAD_X;
                let height = self.class_0[i].height() * SCALE_HEAD_Y;
                let bbox = Bbox::new(xmin, ymin, width, height, 1.0, 1);
                class_1.push(bbox);
            }
        }
        self.class_1 = class_1;
    }

    pub fn len(&self) -> usize {
        self.class_0.len() + self.class_1.len()
    }

    pub fn sort_by<F>(&mut self, compare: F)
    where
        F: Fn(&Bbox, &Bbox) -> Ordering + Clone,
    {
        self.class_0.sort_by(compare.clone());
        self.class_1.sort_by(compare);
    }
}
