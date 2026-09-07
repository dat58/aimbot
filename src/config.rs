use crate::stream::elgato::OutputFormat;
use crate::stream::v4l2::parse_fourcc;
use std::{env::var, path::PathBuf, time::Duration};

pub const SCALE_HEAD_Y: f32 = 1. / 6.;
pub const SCALE_HEAD_X: f32 = 0.6;
pub const SCALE_NECK_Y: f32 = 2.5 / 6.;
pub const SCALE_CHEST_Y: f32 = 3.0 / 6.;
pub const SCALE_ABDOMEN_Y: f32 = 5.0 / 6.;
pub const WIN_DPI_SCALE_FACTOR: f64 = 96.;

/// Geometry of the frames the pipeline actually receives, plus the per-axis
/// factor that turns a frame pixel into a screen pixel.
///
/// `REGION_*` and every detection are expressed in *frame* pixels — `REGION_*`
/// indexes the incoming frame directly. The game, however, renders at
/// `SCREEN_*`, so a mouse delta has to be converted before it means anything.
/// The two spaces coincide only when the capture matches the screen.
///
/// `CAPTURE_*` describes the frame only for a V4L2 source; NDI and UDP ignore
/// it entirely, so those are pinned at 1:1 with the screen rather than scaled
/// by a number that means nothing for them.
pub fn frame_geometry(
    source_stream: &str,
    screen: (u32, u32),
    capture: (u32, u32),
) -> ((u32, u32), (f64, f64)) {
    let source = source_stream.trim();
    let v4l2 = source.starts_with("elgato://") || source.starts_with("v4l2://");
    let frame = if v4l2 && capture.0 > 0 && capture.1 > 0 {
        capture
    } else {
        screen
    };
    let scale = (
        screen.0 as f64 / frame.0 as f64,
        screen.1 as f64 / frame.1 as f64,
    );
    (frame, scale)
}

/// Mouse counts for a target `frame_delta` pixels from the crosshair.
///
/// The delta arrives in frame pixels and is scaled to screen pixels first,
/// because `MOUSE_DPI`/`GAME_SENS` are calibrated against what the game draws,
/// not against whatever resolution the capture card happens to deliver.
pub fn mouse_counts(
    frame_delta: (f32, f32),
    frame_to_screen: (f64, f64),
    game_sens: f64,
    mouse_dpi: f64,
) -> (f64, f64) {
    let counts = |d: f32, scale: f64| -> f64 {
        d as f64 * scale * WIN_DPI_SCALE_FACTOR / game_sens / mouse_dpi
    };
    (
        counts(frame_delta.0, frame_to_screen.0),
        counts(frame_delta.1, frame_to_screen.1),
    )
}

#[derive(Debug, Clone)]
pub struct Config {
    pub event_listener_port: u16,

    pub source_stream: String,
    pub ndi_source_name: Option<String>,
    pub ndi_timeout: std::time::Duration,
    pub screen_width: u32,
    pub screen_height: u32,
    /// Geometry of the frames that actually arrive, which is what `REGION_*`
    /// and every detection are measured in. See [`frame_geometry`].
    pub frame_width: u32,
    pub frame_height: u32,
    /// Frame pixel -> screen pixel, per axis. `(1.0, 1.0)` when the capture
    /// already matches the screen.
    pub frame_to_screen: (f64, f64),
    pub region_top: u32,
    pub region_left: u32,
    pub region_width: u32,
    pub region_height: u32,
    pub scale_min_zone1: f32,
    pub scale_min_zone2: f32,
    /// condition to trigger, if fov <= L2 distance -> allow trigger
    pub fov: f32,

    pub model_provider: String,
    pub model_path: PathBuf,
    pub model_input_size: usize,
    pub model_conf_body: f32,
    pub model_conf_head: f32,
    pub model_iou: f32,
    pub build_head_iou: Option<f32>,

    /// True for the `v_light_*` models, which need the nearest-neighbour
    /// stretch preprocess and the raw multi-class output decoder.
    pub light_model: bool,

    pub capture_device: String,
    pub capture_width: u32,
    pub capture_height: u32,
    pub capture_fps: u32,
    pub capture_fourcc: Option<u32>,
    pub capture_buffers: u32,
    pub capture_drop_stale: bool,
    pub capture_timeout: Duration,
    pub capture_output: OutputFormat,
    /// Hand only `REGION_*` downstream instead of the whole frame. The
    /// detection pipeline never looks outside that region, so converting or
    /// JPEG-decoding the rest is pure latency. Turn it off for the `debug`
    /// feature's whole-frame bbox overlay.
    pub capture_roi: bool,

    pub gpu_id: Option<i32>,
    pub gpu_mem_limit: Option<usize>,
    pub trt_min_shapes: String,
    pub trt_opt_shapes: String,
    pub trt_max_shapes: String,
    pub trt_fp16: Option<bool>,
    pub trt_max_partition_iterations: Option<u32>,
    pub trt_builder_optimization_level: Option<u8>,
    pub trt_dla_enable: Option<bool>,
    pub trt_dla_core: Option<u32>,
    pub trt_auxiliary_streams: Option<i8>,
    pub trt_cache_dir: String,

    pub openvino_cache_dir: String,
    pub openvino_device_type: String,
    pub intra_threads: usize,

    pub makcu_port: String,
    pub makcu_baud: u32,
    pub makcu_listen: bool,
    pub mouse_dpi: f64,
    pub game_sens: f64,

    pub esp_port: Option<String>,
    pub default_aim_mode: Option<u8>,
}

impl Config {
    pub fn new() -> Self {
        let event_listener_port = var("EVENT_LISTENER_PORT")
            .unwrap_or(String::from("10000"))
            .parse::<u16>()
            .expect("EVENT_LISTENER_PORT is not a valid port");
        let source_stream = var("SOURCE_STREAM").expect("No SOURCE_STREAM specified");
        let ndi_source_name = var("NDI_SOURCE_NAME").ok();
        let ndi_timeout = std::time::Duration::from_millis(
            var("NDI_TIMEOUT")
                .unwrap_or(String::from("1000"))
                .parse::<u64>()
                .expect("NDI_TIMEOUT is not a valid integer"),
        );
        let screen_width = var("SCREEN_WIDTH")
            .unwrap_or("1920".to_string())
            .parse::<u32>()
            .expect("SCREEN_WIDTH is not a number");
        let screen_height = var("SCREEN_HEIGHT")
            .unwrap_or("1080".to_string())
            .parse::<u32>()
            .expect("SCREEN_HEIGHT is not a number");
        let region_top = var("REGION_TOP")
            .unwrap_or("0".to_string())
            .parse::<u32>()
            .expect("REGION_TOP is not a number");
        let region_left = var("REGION_LEFT")
            .unwrap_or("0".to_string())
            .parse::<u32>()
            .expect("REGION_LEFT is not a number");
        let region_width = var("REGION_WIDTH")
            .unwrap_or("0".to_string())
            .parse::<u32>()
            .expect("REGION_WIDTH is not a number");
        let region_height = var("REGION_HEIGHT")
            .unwrap_or("0".to_string())
            .parse::<u32>()
            .expect("REGION_HEIGHT is not a number");
        let scale_min_zone1 = var("SCALE_MIN_ZONE1")
            .unwrap_or("0.5".to_string())
            .parse::<f32>()
            .expect("SCALE_MIN_ZONE1 is not a number");
        let scale_min_zone2 = var("SCALE_MIN_ZONE2")
            .unwrap_or("0.8".to_string())
            .parse::<f32>()
            .expect("SCALE_MIN_ZONE2 is not a number");
        let model_provider = var("MODEL_PROVIDER").unwrap_or("cpu".to_string());
        let model_path = PathBuf::from(var("MODEL_PATH").expect("No MODEL_PATH specified"));
        if !model_path.is_file() {
            panic!("Model path is not a file");
        }
        let model_input_size = var("MODEL_INPUT_SIZE")
            .expect("No MODEL_INPUT_SIZE specified")
            .parse::<usize>()
            .expect("MODEL_INPUT_SIZE is not a number");
        let model_conf_body = var("MODEL_CONF_BODY")
            .expect("No MODEL_CONF_BODY specified")
            .parse::<f32>()
            .expect("MODEL_CONF_BODY is not a number");
        let model_conf_head = var("MODEL_CONF_HEAD")
            .expect("No MODEL_CONF_HEAD specified")
            .parse::<f32>()
            .expect("MODEL_CONF_HEAD is not a number");
        let model_iou = var("MODEL_IOU")
            .expect("No MODEL_IOU specified")
            .parse::<f32>()
            .expect("MODEL_IOU is not a number");
        let build_head_iou = var("BUILD_HEAD_IOU")
            .ok()
            .map(|o| o.parse::<f32>().expect("BUILD_HEAD_IOU is not a number"));

        // The `v_light_*` models need a different preprocess and output decoder;
        // the file name is what selects it.
        let light_model = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.to_ascii_lowercase().starts_with("v_light"));

        let capture_device = var("CAPTURE_DEVICE").unwrap_or("/dev/video0".to_string());
        let capture_width = var("CAPTURE_WIDTH")
            .unwrap_or("1920".to_string())
            .parse::<u32>()
            .expect("CAPTURE_WIDTH is not a number");
        let capture_height = var("CAPTURE_HEIGHT")
            .unwrap_or("1080".to_string())
            .parse::<u32>()
            .expect("CAPTURE_HEIGHT is not a number");
        let capture_fps = var("CAPTURE_FPS")
            .unwrap_or("60".to_string())
            .parse::<u32>()
            .expect("CAPTURE_FPS is not a number");
        let capture_fourcc = var("CAPTURE_FOURCC")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(|v| parse_fourcc(&v).expect("CAPTURE_FOURCC is not a valid fourcc"));
        let capture_buffers = var("CAPTURE_BUFFERS")
            .unwrap_or("3".to_string())
            .parse::<u32>()
            .expect("CAPTURE_BUFFERS is not a number");
        let capture_drop_stale = var("CAPTURE_DROP_STALE")
            .unwrap_or("true".to_string())
            .parse::<bool>()
            .expect("CAPTURE_DROP_STALE is not a bool");
        let capture_timeout = Duration::from_millis(
            var("CAPTURE_TIMEOUT_MS")
                .unwrap_or("1000".to_string())
                .parse::<u64>()
                .expect("CAPTURE_TIMEOUT_MS is not a valid integer"),
        );
        let capture_output = OutputFormat::parse(&var("CAPTURE_OUTPUT").unwrap_or_default())
            .expect("CAPTURE_OUTPUT is not valid");
        let ((frame_width, frame_height), frame_to_screen) = frame_geometry(
            &source_stream,
            (screen_width, screen_height),
            (capture_width, capture_height),
        );
        if (frame_width, frame_height) != (screen_width, screen_height) {
            tracing::info!(
                "[Config] frames arrive at {}x{} while the game renders at {}x{}; \
                 REGION_* and detections are in frame pixels and mouse deltas are \
                 scaled by {:.4}x/{:.4}y",
                frame_width,
                frame_height,
                screen_width,
                screen_height,
                frame_to_screen.0,
                frame_to_screen.1,
            );
        }
        let capture_roi = var("CAPTURE_ROI")
            .unwrap_or("true".to_string())
            .parse::<bool>()
            .expect("CAPTURE_ROI is not a bool");
        let gpu_id = var("GPU_ID").ok().and_then(|s| s.parse::<i32>().ok());
        let gpu_mem_limit = var("GPU_MEM_LIMIT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        let trt_min_shapes = var("TRT_MIN_SHAPES").expect("TRT_MIN_SHAPES missing");
        let trt_opt_shapes = var("TRT_OPT_SHAPES").expect("TRT_OPT_SHAPES missing");
        let trt_max_shapes = var("TRT_MAX_SHAPES").expect("TRT_MAX_SHAPES missing");
        let trt_fp16 = var("TRT_FP16").ok().and_then(|s| s.parse::<bool>().ok());
        let trt_max_partition_iterations = var("TRT_MAX_PARTITION_ITERATIONS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok());
        let trt_builder_optimization_level = var("TRT_BUILDER_OPTIMIZATION_LEVEL")
            .ok()
            .and_then(|s| s.parse::<u8>().ok());
        let trt_dla_enable = var("TRT_DLA_ENABLE")
            .ok()
            .and_then(|s| s.parse::<bool>().ok());
        let trt_dla_core = var("TRT_DLA_CORE").ok().and_then(|s| s.parse::<u32>().ok());
        let trt_auxiliary_streams = var("TRT_AUXILIARY_STREAMS")
            .ok()
            .and_then(|s| s.parse::<i8>().ok());
        let trt_cache_dir = var("TRT_CACHE_DIR").expect("No TRT_CACHE_DIR specified");
        let openvino_cache_dir =
            var("OPENVINO_CACHE_DIR").expect("No OPENVINO_CACHE_DIR specified");
        let openvino_device_type = var("OPENVINO_DEVICE_TYPE").unwrap_or("CPU".to_string());
        let intra_threads = var("INTRA_THREADS")
            .unwrap_or("1".to_string())
            .parse::<usize>()
            .expect("INTRA_THREADS must be a number");
        let makcu_port = var("MAKCU_PORT").expect("No MAKCU_PORT specified");
        let makcu_baud = var("MAKCU_BAUD")
            .unwrap_or("115200".to_string())
            .parse::<u32>()
            .expect("MAKCU_BAUD is not an integer");
        let makcu_listen = var("MAKCU_LISTEN")
            .unwrap_or("false".to_string())
            .parse::<bool>()
            .expect("MAKCU_LISTEN is not a bool");
        let mouse_dpi = var("MOUSE_DPI")
            .unwrap_or("1000.".to_string())
            .parse::<f64>()
            .expect("MOUSE_DPI is not a number");
        let game_sens = var("GAME_SENS")
            .unwrap_or("1.".to_string())
            .parse::<f64>()
            .expect("GAME_SENS is not a number");
        let esp_port = var("ESP_PORT").ok();
        // Compared against `dist`, which is a frame-pixel distance because
        // `min_zone` comes straight off bbox dimensions.
        let fov = var("FOV")
            .unwrap_or((frame_width as f32).to_string())
            .parse()
            .unwrap();
        let default_aim_mode = var("DEFAULT_AIM_MODE")
            .ok()
            .and_then(|v| Some(v.parse::<u8>().expect("DEFAULT_AIM_MODE is not a u8")));
        Self {
            event_listener_port,
            source_stream,
            ndi_source_name,
            ndi_timeout,
            screen_width,
            screen_height,
            frame_width,
            frame_height,
            frame_to_screen,
            region_top,
            region_left,
            region_width,
            region_height,
            scale_min_zone1,
            scale_min_zone2,
            model_provider,
            model_path,
            model_input_size,
            model_conf_body,
            model_conf_head,
            model_iou,
            build_head_iou,
            light_model,
            capture_device,
            capture_width,
            capture_height,
            capture_fps,
            capture_fourcc,
            capture_buffers,
            capture_drop_stale,
            capture_timeout,
            capture_output,
            capture_roi,
            gpu_id,
            gpu_mem_limit,
            trt_min_shapes,
            trt_opt_shapes,
            trt_max_shapes,
            trt_fp16,
            trt_max_partition_iterations,
            trt_builder_optimization_level,
            trt_dla_enable,
            trt_dla_core,
            trt_auxiliary_streams,
            trt_cache_dir,
            openvino_cache_dir,
            openvino_device_type,
            intra_threads,
            makcu_port,
            makcu_baud,
            makcu_listen,
            mouse_dpi,
            game_sens,
            esp_port,
            fov,
            default_aim_mode,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_v4l2_source_measures_frames_in_capture_pixels() {
        let (frame, scale) = frame_geometry("elgato://", (2560, 1440), (1920, 1080));
        assert_eq!(frame, (1920, 1080));
        assert!((scale.0 - 1440. / 1080.).abs() < 1e-9);
        assert!((scale.1 - 1440. / 1080.).abs() < 1e-9);
        // `v4l2://` is the same source behind a different scheme.
        assert_eq!(
            frame_geometry("v4l2:///dev/video1", (2560, 1440), (1920, 1080)),
            frame_geometry("elgato://", (2560, 1440), (1920, 1080))
        );
    }

    #[test]
    fn a_matching_capture_needs_no_scaling() {
        let (frame, scale) = frame_geometry("elgato://", (2560, 1440), (2560, 1440));
        assert_eq!(frame, (2560, 1440));
        assert_eq!(scale, (1.0, 1.0));
    }

    /// NDI and UDP never look at `CAPTURE_*`, so scaling by it would apply a
    /// factor derived from a number that describes nothing.
    #[test]
    fn non_v4l2_sources_are_pinned_to_the_screen() {
        for source in ["ndi://192.168.2.3", "udp://127.0.0.1:4200", "assets/clip.mp4"] {
            let (frame, scale) = frame_geometry(source, (2560, 1440), (1920, 1080));
            assert_eq!(frame, (2560, 1440), "{source}");
            assert_eq!(scale, (1.0, 1.0), "{source}");
        }
    }

    #[test]
    fn a_zero_capture_size_falls_back_to_the_screen() {
        let (frame, scale) = frame_geometry("elgato://", (1920, 1080), (0, 0));
        assert_eq!(frame, (1920, 1080));
        assert_eq!(scale, (1.0, 1.0));
    }

    #[test]
    fn mouse_counts_scale_a_frame_delta_into_screen_pixels() {
        // 1:1 capture: the delta passes through the old formula unchanged.
        let (dx, dy) = mouse_counts((100., -50.), (1.0, 1.0), 2.0, 800.);
        assert!((dx - 100. * WIN_DPI_SCALE_FACTOR / 2.0 / 800.).abs() < 1e-9);
        assert!((dy + 50. * WIN_DPI_SCALE_FACTOR / 2.0 / 800.).abs() < 1e-9);

        // Capturing 1920 of a 2560-wide screen: a 100 px frame delta is really
        // a 133.33 px move in the game, so it must produce more counts.
        let scale = 2560. / 1920.;
        let (wide, _) = mouse_counts((100., 0.), (scale, scale), 2.0, 800.);
        assert!((wide / dx - scale).abs() < 1e-9, "{wide} vs {dx}");
    }

    #[test]
    fn mouse_counts_are_symmetric_and_zero_at_the_crosshair() {
        assert_eq!(mouse_counts((0., 0.), (1.3, 1.3), 1.5, 1000.), (0., 0.));
        let (px, py) = mouse_counts((30., 40.), (1.3, 1.3), 1.5, 1000.);
        let (nx, ny) = mouse_counts((-30., -40.), (1.3, 1.3), 1.5, 1000.);
        assert!((px + nx).abs() < 1e-12 && (py + ny).abs() < 1e-12);
    }

    /// End to end: a target at the centre of the frame is the crosshair, so it
    /// must produce no mouse movement no matter how the capture is scaled.
    #[test]
    fn a_target_on_the_crosshair_never_moves_the_mouse() {
        for (screen, capture) in [
            ((2560u32, 1440u32), (2560u32, 1440u32)),
            ((2560, 1440), (1920, 1080)),
            ((1920, 1080), (1280, 720)),
        ] {
            let ((frame_w, frame_h), scale) = frame_geometry("elgato://", screen, capture);
            let crosshair = (frame_w as f32 / 2., frame_h as f32 / 2.);
            let delta = (crosshair.0 - frame_w as f32 / 2., crosshair.1 - frame_h as f32 / 2.);
            assert_eq!(mouse_counts(delta, scale, 1.0, 1000.), (0., 0.));
        }
    }
}
