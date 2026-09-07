use crate::config::{Config, SCALE_HEAD_X, SCALE_HEAD_Y};
use anyhow::{Result, bail};
use ndarray::{Array, Axis, Ix2, s};
use opencv::{
    core::{Mat, MatTraitConst, MatTraitConstManual, Point, Rect, Size, VecN},
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

/// How the incoming frame maps onto the screen.
///
/// The three fields come apart when the capture layer already cropped the frame
/// down to `REGION_*` (`CAPTURE_ROI`): `read` then covers the whole frame,
/// while `origin` and `bound` still speak screen coordinates so detections land
/// where the crosshair does.
#[derive(Clone, Copy, Debug)]
struct Source {
    /// Rect to sample inside the frame.
    read: Rect,
    /// Screen-space position of `read`, added back onto every detection.
    origin: Point,
    /// Screen-space extent detections are clamped to.
    bound: Size,
    /// Screen-space position of the *frame*, which is not the same thing as
    /// `origin`: reading a region out of a whole frame leaves the frame itself
    /// at `(0, 0)`. Only a frame the capture layer already cropped down to the
    /// region sits at the region's origin. Anything drawing a screen-space
    /// detection onto the frame subtracts this.
    frame_origin: Point,
}

/// Where to read from a `frame_w` x `frame_h` frame, and how to put detections
/// back into screen coordinates.
///
/// Pulled out of [`Model`] as a pure function because this is the arithmetic
/// that decides whether a detection lands on the crosshair or hundreds of
/// pixels away, and a `Model` needs an ONNX session to exist.
fn source_for(crop: bool, roi: Rect, screen: Size, frame_w: i32, frame_h: i32) -> Result<Source> {
    let (w, h) = (frame_w, frame_h);
    if w <= 0 || h <= 0 {
        bail!("[Model] empty frame");
    }
    if !crop {
        return Ok(Source {
            read: Rect::new(0, 0, w, h),
            origin: Point::new(0, 0),
            bound: Size::new(w, h),
            frame_origin: Point::new(0, 0),
        });
    }
    if roi.width <= 0 || roi.height <= 0 {
        bail!(
            "[Model] region must be non-empty, got {}x{}",
            roi.width,
            roi.height
        );
    }
    // A frame that is exactly the configured region arrived pre-cropped from
    // the capture layer, so cropping again would cut a region out of a region.
    // The read rect covers the frame while origin/bound stay screen-space.
    if w == roi.width && h == roi.height {
        return Ok(Source {
            read: Rect::new(0, 0, w, h),
            origin: Point::new(roi.x, roi.y),
            bound: screen,
            // The frame *is* the region, so it starts where the region does.
            frame_origin: Point::new(roi.x, roi.y),
        });
    }
    if roi.x < 0 || roi.y < 0 || roi.x + roi.width > w || roi.y + roi.height > h {
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
    Ok(Source {
        read: roi,
        origin: Point::new(roi.x, roi.y),
        bound: Size::new(w, h),
        // A whole frame spans screen space from its own top-left corner, even
        // though only `roi` is read out of it.
        frame_origin: Point::new(0, 0),
    })
}

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
    /// Screen geometry detections are expressed in. Only equal to the frame
    /// size when the capture layer hands over whole frames.
    screen: Size,
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
            screen: Size::new(config.screen_width as i32, config.screen_height as i32),
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

    /// Region of the frame the model looks at, and where it sits on screen.
    fn source(&self, mat: &Mat) -> Result<Source> {
        source_for(self.crop, self.roi, self.screen, mat.cols(), mat.rows())
    }

    /// Screen-space position of `frame`'s top-left corner.
    ///
    /// Detections come out in screen coordinates, so anything that wants to
    /// draw them onto the frame — or express them relative to it — has to
    /// subtract this. `(0, 0)` for a whole frame; the region origin only when
    /// the capture layer pre-cropped the frame down to the region
    /// (`CAPTURE_ROI`). Shares `source()` with inference so the two cannot
    /// disagree about where the frame sits.
    pub fn frame_origin(&self, frame: &Mat) -> Result<Point> {
        Ok(self.source(frame)?.frame_origin)
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
    fn decode_light(&self, shape: &[i64], values: &[f32], src: &Source) -> Result<Bboxes> {
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

        let sx = src.read.width as f32 / self.input_size as f32;
        let sy = src.read.height as f32 / self.input_size as f32;
        let (bound_w, bound_h) = (src.bound.width as f32, src.bound.height as f32);
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
            let x = cx - w / 2. + src.origin.x as f32;
            let y = cy - h / 2. + src.origin.y as f32;
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
        let src = self.source(mat)?;

        // preprocess
        let pre_time = Instant::now();
        let mut input = self.light_input.borrow_mut();
        self.preprocess_light(mat, src.read, &mut input)?;
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
        let bboxes = self.decode_light(shape, values, &src)?;
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
        let src = self.source(mat)?;
        let mut inputs =
            Array::<f32, _>::from_elem((1, 3, self.input_size, self.input_size), 114. / 255.)
                .into_dyn();
        // A pre-cropped frame is already the region, so `read` covers it all.
        let input = if src.read.width == mat.cols() && src.read.height == mat.rows() {
            std::borrow::Cow::Borrowed(mat)
        } else {
            std::borrow::Cow::Owned(Mat::roi(mat, src.read)?.clone_pointee())
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
            let x = cx - w / 2. - dw as f32 / ratio + src.origin.x as f32;
            let y = cy - h / 2. - dh as f32 / ratio + src.origin.y as f32;
            let bbox = Bbox::new(x, y, w, h, scores[class], class as u8)
                .bound(src.bound.width as f32, src.bound.height as f32);
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

#[cfg(test)]
mod tests {
    use super::*;

    const SCREEN: Size = Size {
        width: 1920,
        height: 1080,
    };
    /// The region from the shipped `.env`.
    const REGION: Rect = Rect {
        x: 864,
        y: 444,
        width: 192,
        height: 192,
    };

    /// Screen coordinate a detection at `(mx, my)` in model space lands on.
    /// Mirrors the arithmetic in `decode_light` so the two stay in step.
    fn to_screen(src: &Source, input_size: i32, mx: f32, my: f32) -> (f32, f32) {
        let sx = src.read.width as f32 / input_size as f32;
        let sy = src.read.height as f32 / input_size as f32;
        (mx * sx + src.origin.x as f32, my * sy + src.origin.y as f32)
    }

    #[test]
    fn a_whole_frame_is_read_through_the_region() {
        let src = source_for(true, REGION, SCREEN, 1920, 1080).unwrap();
        assert_eq!(src.read, REGION);
        assert_eq!(src.origin, Point::new(864, 444));
        assert_eq!(src.bound, Size::new(1920, 1080));
    }

    #[test]
    fn a_pre_cropped_frame_is_read_whole_but_stays_screen_space() {
        let src = source_for(true, REGION, SCREEN, 192, 192).unwrap();
        assert_eq!(src.read, Rect::new(0, 0, 192, 192));
        // The offset has to survive, or every detection lands at the top-left
        // corner of the screen instead of on the crosshair.
        assert_eq!(src.origin, Point::new(864, 444));
        // Bounding to the 192x192 frame would clamp every box to the edge.
        assert_eq!(src.bound, SCREEN);
    }

    /// The property the whole `CAPTURE_ROI` change rests on: a detection maps
    /// to the same screen pixel whether the capture layer cropped or not.
    #[test]
    fn both_paths_map_a_detection_to_the_same_screen_pixel() {
        let whole = source_for(true, REGION, SCREEN, 1920, 1080).unwrap();
        let cropped = source_for(true, REGION, SCREEN, 192, 192).unwrap();
        for &(mx, my) in &[(0., 0.), (96., 96.), (191., 191.), (48.5, 137.25)] {
            assert_eq!(
                to_screen(&whole, 192, mx, my),
                to_screen(&cropped, 192, mx, my),
                "model-space ({mx}, {my}) diverges"
            );
        }
        // Centre of the region is the centre of the screen for this config.
        assert_eq!(to_screen(&cropped, 192, 96., 96.), (960., 540.));
    }

    #[test]
    fn a_region_of_a_different_size_still_scales_per_axis() {
        // A 256-wide region into a 192 input stretches by 4/3 on that axis.
        let roi = Rect::new(800, 400, 256, 192);
        let src = source_for(true, roi, SCREEN, 256, 192).unwrap();
        assert_eq!(to_screen(&src, 192, 0., 0.), (800., 400.));
        assert_eq!(to_screen(&src, 192, 192., 192.), (800. + 256., 400. + 192.));
    }

    /// `frame_origin` is what the debug overlay subtracts, and it is NOT
    /// `origin`: `origin` locates the region, `frame_origin` locates the frame.
    /// They only coincide when the capture layer pre-cropped. Getting this
    /// wrong draws every box shifted by the region offset.
    #[test]
    fn frame_origin_locates_the_frame_not_the_region() {
        // Whole frame: the frame starts at the screen's corner even though only
        // the region is read out of it.
        let whole = source_for(true, REGION, SCREEN, 1920, 1080).unwrap();
        assert_eq!(whole.origin, Point::new(864, 444));
        assert_eq!(whole.frame_origin, Point::new(0, 0));

        // Pre-cropped: the frame *is* the region.
        let cropped = source_for(true, REGION, SCREEN, 192, 192).unwrap();
        assert_eq!(cropped.origin, Point::new(864, 444));
        assert_eq!(cropped.frame_origin, Point::new(864, 444));

        // No crop at all.
        let plain = source_for(false, REGION, SCREEN, 1920, 1080).unwrap();
        assert_eq!(plain.frame_origin, Point::new(0, 0));
    }

    /// A detection at the centre of the region must land inside whichever frame
    /// it is drawn onto, in both capture modes.
    #[test]
    fn a_screen_detection_draws_inside_the_frame_in_both_modes() {
        for (fw, fh) in [(1920, 1080), (192, 192)] {
            let src = source_for(true, REGION, SCREEN, fw, fh).unwrap();
            let (sx, sy) = to_screen(&src, 192, 96., 96.);
            // Same screen pixel regardless of mode.
            assert_eq!((sx, sy), (960., 540.), "frame {fw}x{fh}");

            // Exactly what main.rs computes for the overlay.
            let dx = sx - src.frame_origin.x as f32;
            let dy = sy - src.frame_origin.y as f32;
            assert!(
                (0. ..fw as f32).contains(&dx) && (0. ..fh as f32).contains(&dy),
                "frame {fw}x{fh}: draw point ({dx}, {dy}) falls outside the frame"
            );
            let expected = if fw == 192 { (96., 96.) } else { (960., 540.) };
            assert_eq!((dx, dy), expected, "frame {fw}x{fh}");
        }
    }

    /// The region need not be centred; the overlay shift must still be exact.
    #[test]
    fn an_off_centre_region_shifts_exactly() {
        let roi = Rect::new(1184, 624, 192, 192);
        let screen = Size::new(2560, 1440);
        // Whole 2560x1440 frame: no shift, box drawn at its screen position.
        let whole = source_for(true, roi, screen, 2560, 1440).unwrap();
        assert_eq!(whole.frame_origin, Point::new(0, 0));
        assert_eq!(to_screen(&whole, 192, 96., 96.), (1280., 720.));

        // Pre-cropped: shift by the region origin puts it at the frame centre.
        let cropped = source_for(true, roi, screen, 192, 192).unwrap();
        assert_eq!(cropped.frame_origin, Point::new(1184, 624));
        let (sx, sy) = to_screen(&cropped, 192, 96., 96.);
        assert_eq!(
            (sx - 1184., sy - 624.),
            (96., 96.),
            "off-centre region does not shift onto the frame centre"
        );
    }

    #[test]
    fn no_crop_reads_and_bounds_the_frame_itself() {
        let src = source_for(false, REGION, SCREEN, 1280, 720).unwrap();
        assert_eq!(src.read, Rect::new(0, 0, 1280, 720));
        assert_eq!(src.origin, Point::new(0, 0));
        assert_eq!(src.bound, Size::new(1280, 720));
    }

    #[test]
    fn a_region_outside_a_whole_frame_is_rejected() {
        // A region placed for 1080p, against a 720p frame: it runs off both
        // edges, which is the failure the whole-frame path has to catch.
        // 1600+192 = 1792 and 800+192 = 992: inside 1080p, off the edge at 720p.
        let low = Rect::new(1600, 800, 192, 192);
        assert!(source_for(true, low, SCREEN, 1280, 720).is_err());
        // ...and the same region is fine once the frame is big enough.
        assert!(source_for(true, low, SCREEN, 1920, 1080).is_ok());
        assert!(source_for(true, Rect::new(-1, 0, 192, 192), SCREEN, 1920, 1080).is_err());
        assert!(source_for(true, Rect::new(0, 0, 0, 192), SCREEN, 1920, 1080).is_err());
        assert!(source_for(true, REGION, SCREEN, 0, 0).is_err());
    }
}
