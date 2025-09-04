use std::{env::var, path::PathBuf};

pub const SCALE_HEAD_Y: f32 = 2. / 6.;
pub const SCALE_NECK_Y: f32 = 2.5 / 6.;
pub const SCALE_CHEST_Y: f32 = 3.5 / 6.;
pub const SCALE_ABDOMEN_Y: f32 = 5.1 / 6.;
pub const WIN_DPI_SCALE_FACTOR: f64 = 96.;

#[derive(Debug, Clone)]
pub struct Config {
    pub event_listener_port: u16,

    pub source_stream: String,
    pub ndi_source_name: Option<String>,
    pub ndi_timeout: std::time::Duration,
    pub screen_width: u32,
    pub screen_height: u32,
    pub region_top: u32,
    pub region_left: u32,
    pub region_width: u32,
    pub region_height: u32,
    pub scale_min_zone: f32,
    pub sleep_interval_capture: Option<std::time::Duration>,

    pub model_provider: String,
    pub model_path: PathBuf,
    pub model_input_size: usize,
    pub model_conf: f32,
    pub model_iou: f32,
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
    pub intra_threads: usize,

    pub makcu_port: String,
    pub makcu_baud: u32,
    pub mouse_dpi: f64,
    pub game_sens: f64,
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
        let scale_min_zone = var("SCALE_MIN_ZONE")
            .unwrap_or("0.85".to_string())
            .parse::<f32>()
            .expect("SCALE_MIN_ZONE is not a number");
        let sleep_interval_capture = var("SLEEP_INTERVAL_CAPTURE").ok().and_then(|v| {
            Some(std::time::Duration::from_millis(
                v.parse().expect("SLEEP_INTERVAL_CAPTURE is not a number"),
            ))
        });
        let model_provider = var("MODEL_PROVIDER").unwrap_or("cpu".to_string());
        let model_path = PathBuf::from(var("MODEL_PATH").expect("No MODEL_PATH specified"));
        if !model_path.is_file() {
            panic!("Model path is not a file");
        }
        let model_input_size = var("MODEL_INPUT_SIZE")
            .expect("No MODEL_INPUT_SIZE specified")
            .parse::<usize>()
            .expect("MODEL_INPUT_SIZE is not a number");
        let model_conf = var("MODEL_CONF")
            .expect("No MODEL_CONF specified")
            .parse::<f32>()
            .expect("MODEL_CONF is not a number");
        let model_iou = var("MODEL_IOU")
            .expect("No MODEL_IOU specified")
            .parse::<f32>()
            .expect("MODEL_IOU is not a number");
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
        let intra_threads = var("INTRA_THREADS")
            .unwrap_or("3".to_string())
            .parse::<usize>()
            .expect("INTRA_THREADS is not a number");
        let makcu_port = var("MAKCU_PORT").expect("No MAKCU_PORT specified");
        let makcu_baud = var("MAKCU_BAUD")
            .unwrap_or("115200".to_string())
            .parse::<u32>()
            .expect("MAKCU_BAUD is not an integer");
        let mouse_dpi = var("MOUSE_DPI")
            .unwrap_or("1000.".to_string())
            .parse::<f64>()
            .expect("MOUSE_DPI is not a number");
        let game_sens = var("GAME_SENS")
            .unwrap_or("1.".to_string())
            .parse::<f64>()
            .expect("GAME_SENS is not a number");
        Self {
            event_listener_port,
            source_stream,
            ndi_source_name,
            ndi_timeout,
            screen_width,
            screen_height,
            region_top,
            region_left,
            region_width,
            region_height,
            scale_min_zone,
            sleep_interval_capture,
            model_provider,
            model_path,
            model_input_size,
            model_conf,
            model_iou,
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
            intra_threads,
            makcu_port,
            makcu_baud,
            mouse_dpi,
            game_sens,
        }
    }
}
