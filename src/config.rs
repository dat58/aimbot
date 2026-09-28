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

/// Report cadence for the smooth-aim playback, derived from the link speed.
///
/// A `km.move(-12,-7)` line is about 18 bytes and 8N1 puts ten bits on the
/// wire per byte, so 115200 baud carries roughly 640 commands a second — at
/// 250 Hz the link stays near 39% busy even in the pathological case where
/// every report is non-zero. At 2M and above the same arithmetic puts 1000 Hz
/// under 10%. The 1000 Hz ceiling is deliberate rather than a throughput
/// limit: no real mouse reports faster, so a higher cadence would itself be a
/// signature.
pub fn poll_hz_for_baud(baud: u32) -> u32 {
    if baud >= 2_000_000 { 1000 } else { 250 }
}

/// Numeric tuning for the human-flick model behind
/// [`crate::mouse::MouseVirtual::move_smooth`].
///
/// Whether that model is used at all is not decided here. `MOVE_SMOOTH` picks
/// it over `move_bezier` for the ordinary aim modes, and
/// [`crate::aim::AimMode::aim_smooth`] — the ESP button 2 path — always uses
/// it regardless.
#[derive(Debug, Clone, Copy)]
pub struct SmoothConfig {
    /// Report cadence. `0` derives it from `MAKCU_BAUD`; see
    /// [`poll_hz_for_baud`].
    pub poll_hz: u32,
    /// Fitts intercept, milliseconds.
    pub fitts_a_ms: f64,
    /// Fitts slope, milliseconds per bit of index of difficulty.
    pub fitts_b_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    /// Lognormal spread applied to the movement time, as a fraction. 0.12
    /// gives a coefficient of variation near 12%, which is what human
    /// trial-to-trial variability looks like.
    pub mt_jitter: f64,
    /// Fraction of the distance the ballistic phase commits to. The rest is
    /// left for the next frame's corrective submovement.
    pub gain: f64,
    pub gain_sd: f64,
    /// How often a flick overshoots instead of falling short.
    pub overshoot_p: f64,
    /// Where peak speed sits, as a fraction of the movement time.
    pub peak_min: f64,
    pub peak_max: f64,
    /// Maximum perpendicular deviation from the straight line, as a fraction
    /// of the amplitude.
    pub bow: f64,
    /// How strongly the bow direction follows the direction of travel. `0` is
    /// a coin flip, `0.5` is fully handed.
    pub curve_bias: f64,
    /// Skews where along the path the bow peaks.
    pub kappa_min: f64,
    pub kappa_max: f64,
    /// Harris & Wolpert coefficient: noise standard deviation as a fraction of
    /// the per-report displacement.
    pub noise: f64,
    /// Physiological tremor amplitude, in mouse counts. `0` disables it.
    pub tremor: f64,
    /// Reaction latency on a fresh acquisition. Off by default, because the
    /// ESP trigger button already supplies a human reaction time — the command
    /// only leaves this process while that button is held.
    pub react_min_ms: u64,
    pub react_max_ms: u64,
    /// Silence longer than this means the old target is gone, so the leftover
    /// sub-count fraction describes a move that no longer exists.
    pub gap_ms: u64,
    /// Ceiling on a single report, in counts. Excess is deferred to the next
    /// report rather than dropped.
    pub max_counts: i64,
    /// Hand-speed envelope, in inches per second of physical mouse travel.
    ///
    /// Fitts's index of difficulty is a *ratio* of distance to target width, so
    /// it is dimensionless and says nothing about how far the hand actually
    /// moves. Counts do: `counts / MOUSE_DPI` is inches. Without this bound the
    /// same 250 ms stroke is asked of 500 counts and of 5, and the second one
    /// implies a hand creeping at hundredths of an inch per second — a
    /// deliberate crawl, not an aim, and the reason this visibly failed to move
    /// at a low counts-per-pixel.
    ///
    /// `min_speed_ips` is not merely a rail: below a few hundred counts it is
    /// what actually sets the duration, and Fitts takes over above that. This
    /// is the right way round. Fitts describes *aimed* reaches, where landing
    /// accurately is the cost; a correction far too small for that just travels
    /// at the hand's comfortable speed, which makes its duration proportional
    /// to its length. Fitts's law is known to flatten out at very low indices
    /// of difficulty for the same reason.
    ///
    /// Expressing both in inches rather than counts is what makes them hold at
    /// any `MOUSE_DPI` and `GAME_SENS`, instead of needing a retune whenever
    /// the counts-per-pixel changes — which is not a tuning knob but a fact
    /// about the mouse and the game.
    pub min_speed_ips: f64,
    pub max_speed_ips: f64,
    /// Send each report as `km.move_now` instead of `km.move`.
    ///
    /// MAKCU firmware V4.028+ interpolates a plain `km.move` internally, over
    /// roughly 4-42 ms depending on magnitude. This driver does its own
    /// interpolation and times every report itself, so that second layer only
    /// smears the profile and lets commands queue up behind each other.
    /// `km.move_now` puts the counts in the next report untouched, which is
    /// what a per-report driver wants.
    ///
    /// Off by default because the command does not exist on older firmware,
    /// where enabling it would stop the mouse moving at all.
    pub move_now: bool,
    /// Mean counts a report should carry, which sets how many reports a stroke
    /// of a given amplitude is worth planning.
    ///
    /// Reports are the resolution limit of the whole model: counts are
    /// integers, so a report holding less than half a count rounds to zero and
    /// moves nothing. Fitts's law sizes a stroke in *time*, and at 1 kHz a
    /// 250 ms stroke is 250 reports — fine for the hundreds of counts a
    /// full-screen flick carries, useless for the handful of counts an
    /// in-region correction carries, where it leaves 98% of the reports empty
    /// and the mouse visibly crawling.
    ///
    /// A real mouse mid-flick reports tens of counts at a time, so anything at
    /// or above ~1 keeps every report doing work. Raise it for fewer, larger
    /// reports; below 1 the zero-report problem starts to come back.
    pub counts_per_report: f64,
}

impl Default for SmoothConfig {
    fn default() -> Self {
        Self {
            poll_hz: 0,
            fitts_a_ms: 35.,
            fitts_b_ms: 55.,
            min_ms: 45.,
            max_ms: 320.,
            mt_jitter: 0.12,
            gain: 0.90,
            gain_sd: 0.045,
            overshoot_p: 0.15,
            peak_min: 0.30,
            peak_max: 0.45,
            bow: 0.03,
            curve_bias: 0.30,
            kappa_min: 0.85,
            kappa_max: 1.25,
            noise: 0.022,
            tremor: 0.35,
            react_min_ms: 0,
            react_max_ms: 0,
            gap_ms: 250,
            max_counts: 127,
            move_now: false,
            counts_per_report: 2.0,
            // A slow deliberate correction and a hard flick, respectively.
            min_speed_ips: 0.6,
            max_speed_ips: 40.0,
        }
    }
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
    /// `MOVE_SMOOTH`: move through the human-flick model instead of the
    /// firmware bezier in the ordinary aim modes. The ESP button 2 path uses
    /// it whatever this says.
    pub move_smooth: bool,
    /// Tuning for the human-flick model; see [`SmoothConfig`].
    pub smooth: SmoothConfig,

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
        // `mouse_counts` divides by both. A zero there yields an infinite
        // delta, and `NaN as i32` saturates to zero rather than panicking, so
        // the failure would be a silently frozen aim instead of a crash.
        if !(game_sens > 0.) {
            panic!("GAME_SENS must be greater than zero");
        }
        if !(mouse_dpi > 0.) {
            panic!("MOUSE_DPI must be greater than zero");
        }
        let move_smooth = var("MOVE_SMOOTH")
            .unwrap_or("false".to_string())
            .parse::<bool>()
            .expect("MOVE_SMOOTH is not a bool");
        let default_smooth = SmoothConfig::default();
        let smooth_f = |name: &str, default: f64| -> f64 {
            var(name)
                .unwrap_or(default.to_string())
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("{name} is not a number"))
        };
        let smooth_u64 = |name: &str, default: u64| -> u64 {
            var(name)
                .unwrap_or(default.to_string())
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("{name} is not an integer"))
        };
        let smooth = SmoothConfig {
            poll_hz: var("MOVE_SMOOTH_POLL_HZ")
                .unwrap_or("0".to_string())
                .parse::<u32>()
                .expect("MOVE_SMOOTH_POLL_HZ is not a number"),
            fitts_a_ms: smooth_f("MOVE_SMOOTH_FITTS_A_MS", default_smooth.fitts_a_ms),
            fitts_b_ms: smooth_f("MOVE_SMOOTH_FITTS_B_MS", default_smooth.fitts_b_ms),
            min_ms: smooth_f("MOVE_SMOOTH_MIN_MS", default_smooth.min_ms),
            max_ms: smooth_f("MOVE_SMOOTH_MAX_MS", default_smooth.max_ms),
            gain: smooth_f("MOVE_SMOOTH_GAIN", default_smooth.gain),
            overshoot_p: smooth_f("MOVE_SMOOTH_OVERSHOOT_P", default_smooth.overshoot_p),
            peak_min: smooth_f("MOVE_SMOOTH_PEAK_MIN", default_smooth.peak_min),
            peak_max: smooth_f("MOVE_SMOOTH_PEAK_MAX", default_smooth.peak_max),
            bow: smooth_f("MOVE_SMOOTH_BOW", default_smooth.bow),
            noise: smooth_f("MOVE_SMOOTH_NOISE", default_smooth.noise),
            tremor: smooth_f("MOVE_SMOOTH_TREMOR", default_smooth.tremor),
            react_min_ms: smooth_u64("MOVE_SMOOTH_REACT_MIN_MS", default_smooth.react_min_ms),
            react_max_ms: smooth_u64("MOVE_SMOOTH_REACT_MAX_MS", default_smooth.react_max_ms),
            move_now: var("MOVE_SMOOTH_MOVE_NOW")
                .unwrap_or(default_smooth.move_now.to_string())
                .parse::<bool>()
                .expect("MOVE_SMOOTH_MOVE_NOW is not a bool"),
            min_speed_ips: smooth_f("MOVE_SMOOTH_MIN_SPEED_IPS", default_smooth.min_speed_ips),
            max_speed_ips: smooth_f("MOVE_SMOOTH_MAX_SPEED_IPS", default_smooth.max_speed_ips),
            counts_per_report: smooth_f(
                "MOVE_SMOOTH_COUNTS_PER_REPORT",
                default_smooth.counts_per_report,
            ),
            ..default_smooth
        };
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
            move_smooth,
            smooth,
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
        for source in [
            "ndi://192.168.2.3",
            "udp://127.0.0.1:4200",
            "assets/clip.mp4",
        ] {
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
            let delta = (
                crosshair.0 - frame_w as f32 / 2.,
                crosshair.1 - frame_h as f32 / 2.,
            );
            assert_eq!(mouse_counts(delta, scale, 1.0, 1000.), (0., 0.));
        }
    }

    /// A `km.move` line is about 18 bytes and 8N1 spends ten bits per byte, so
    /// the cadence has to leave headroom even at the slowest allowed link —
    /// otherwise one stalled write pushes every later report past its
    /// deadline and the velocity profile smears.
    #[test]
    fn the_report_rate_stays_inside_the_serial_byte_budget_at_every_baud() {
        const LINE_BYTES: f64 = 18.;
        const BITS_PER_BYTE: f64 = 10.;
        for (baud, expected) in [(115_200u32, 250u32), (2_000_000, 1000), (4_000_000, 1000)] {
            let hz = poll_hz_for_baud(baud);
            assert_eq!(hz, expected, "{baud}");
            let duty = hz as f64 * LINE_BYTES * BITS_PER_BYTE / baud as f64;
            assert!(duty < 0.5, "{baud} would run the link {duty:.2} busy");
        }
    }
}
