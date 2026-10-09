use crate::aim::Mode;
use crate::stream::elgato::OutputFormat;
use crate::stream::v4l2::parse_fourcc;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Deserializer};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};

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
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SmoothConfig {
    /// Report cadence. `0` derives it from `MAKCU_BAUD`; see
    /// [`poll_hz_for_baud`].
    #[serde(rename = "move_smooth_poll_hz")]
    pub poll_hz: u32,
    /// Fitts intercept, milliseconds.
    #[serde(rename = "move_smooth_fitts_a_ms")]
    pub fitts_a_ms: f64,
    /// Fitts slope, milliseconds per bit of index of difficulty.
    #[serde(rename = "move_smooth_fitts_b_ms")]
    pub fitts_b_ms: f64,
    #[serde(rename = "move_smooth_min_ms")]
    pub min_ms: f64,
    #[serde(rename = "move_smooth_max_ms")]
    pub max_ms: f64,
    /// Lognormal spread applied to the movement time, as a fraction. 0.12
    /// gives a coefficient of variation near 12%, which is what human
    /// trial-to-trial variability looks like.
    #[serde(rename = "move_smooth_mt_jitter")]
    pub mt_jitter: f64,
    /// Fraction of the distance the ballistic phase commits to. The rest is
    /// left for the next frame's corrective submovement.
    #[serde(rename = "move_smooth_gain")]
    pub gain: f64,
    #[serde(rename = "move_smooth_gain_sd")]
    pub gain_sd: f64,
    /// How often a flick overshoots instead of falling short.
    #[serde(rename = "move_smooth_overshoot_p")]
    pub overshoot_p: f64,
    /// Where peak speed sits, as a fraction of the movement time.
    #[serde(rename = "move_smooth_peak_min")]
    pub peak_min: f64,
    #[serde(rename = "move_smooth_peak_max")]
    pub peak_max: f64,
    /// Maximum perpendicular deviation from the straight line, as a fraction
    /// of the amplitude.
    #[serde(rename = "move_smooth_bow")]
    pub bow: f64,
    /// How strongly the bow direction follows the direction of travel. `0` is
    /// a coin flip, `0.5` is fully handed.
    #[serde(rename = "move_smooth_curve_bias")]
    pub curve_bias: f64,
    /// Skews where along the path the bow peaks.
    #[serde(rename = "move_smooth_kappa_min")]
    pub kappa_min: f64,
    #[serde(rename = "move_smooth_kappa_max")]
    pub kappa_max: f64,
    /// Harris & Wolpert coefficient: noise standard deviation as a fraction of
    /// the per-report displacement.
    #[serde(rename = "move_smooth_noise")]
    pub noise: f64,
    /// Physiological tremor amplitude, in mouse counts. `0` disables it.
    #[serde(rename = "move_smooth_tremor")]
    pub tremor: f64,
    /// Reaction latency on a fresh acquisition. Off by default, because the
    /// ESP trigger button already supplies a human reaction time — the command
    /// only leaves this process while that button is held.
    #[serde(rename = "move_smooth_react_min_ms")]
    pub react_min_ms: u64,
    #[serde(rename = "move_smooth_react_max_ms")]
    pub react_max_ms: u64,
    /// Silence longer than this means the old target is gone, so the leftover
    /// sub-count fraction describes a move that no longer exists.
    #[serde(rename = "move_smooth_gap_ms")]
    pub gap_ms: u64,
    /// Ceiling on a single report, in counts. Excess is deferred to the next
    /// report rather than dropped.
    #[serde(rename = "move_smooth_max_counts")]
    pub max_counts: i64,
    /// Fire a left click once the flick has settled.
    ///
    /// Only the `move_smooth` path does this. It fires after *every* flick, and
    /// `move_smooth` is called once per frame while the trigger is held, so a
    /// held trigger produces a click roughly every
    /// `auto_click_lower_ms + hold` — a few per second, not one per target.
    #[serde(rename = "move_smooth_auto_click")]
    pub auto_click: bool,
    /// Delay between the flick ending and the click, in milliseconds, drawn
    /// uniformly. Stands in for the gap between settling on a target and
    /// deciding to shoot, so it should not be zero.
    #[serde(rename = "move_smooth_auto_click_lower_ms")]
    pub auto_click_lower_ms: u64,
    #[serde(rename = "move_smooth_auto_click_upper_ms")]
    pub auto_click_upper_ms: u64,
    /// Chance in `[0, 1]` that a click is fired without letting the aim settle
    /// first — a deliberate miss. `0` never misses, `1` always does.
    ///
    /// Both outcomes wait out the same `auto_click_lower_ms..upper_ms`. What
    /// separates them is whether the mouse keeps correcting during that wait.
    /// Holding still leaves the crosshair wherever the ballistic gain put it,
    /// which on a long flick is 12-20 screen pixels off a ~20 pixel target, so
    /// the shot misses. Tracking through the wait lets the corrections land
    /// first and the shot is on target.
    ///
    /// Worth having above zero: never missing is itself a signature. Note this
    /// is unrelated to [`Self::overshoot_p`], which decides whether a *movement*
    /// goes past its target rather than whether a *shot* does.
    #[serde(rename = "move_smooth_auto_click_miss_p")]
    pub auto_click_miss_p: f64,
    /// Smallest gap between two clicks, in milliseconds. `0` disables the
    /// limit.
    ///
    /// Without it a held trigger clicks once per flick, and a flick is planned
    /// once per frame, so the rate is whatever the frame rate and the click
    /// delay happen to multiply out to. A limited click is skipped outright
    /// rather than deferred: the worker does not even wait out the delay, so
    /// the aim keeps tracking instead of stalling on a click it will not fire.
    #[serde(rename = "move_smooth_auto_click_rate_limit_ms")]
    pub auto_click_rate_limit_ms: u64,
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
    #[serde(rename = "move_smooth_min_speed_ips")]
    pub min_speed_ips: f64,
    #[serde(rename = "move_smooth_max_speed_ips")]
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
    #[serde(rename = "move_smooth_move_now")]
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
    #[serde(rename = "move_smooth_counts_per_report")]
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
            auto_click: false,
            auto_click_lower_ms: 100,
            auto_click_upper_ms: 130,
            auto_click_miss_p: 0.0,
            auto_click_rate_limit_ms: 0,
            counts_per_report: 2.0,
            // A slow deliberate correction and a hard flick, respectively.
            min_speed_ips: 0.6,
            max_speed_ips: 40.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mover {
    #[default]
    Smooth,
    Bezier,
}

impl std::fmt::Display for Mover {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mover::Smooth => write!(f, "smooth"),
            Mover::Bezier => write!(f, "bezier"),
        }
    }
}

/// One set of aim and movement tuning, selected by which ESP button is held.
#[derive(Debug, Clone, Deserialize)]
pub struct Profile {
    pub name: String,
    /// condition to trigger, if fov <= L2 distance -> allow trigger.
    /// `0` or absent means the whole frame.
    #[serde(default)]
    pub fov: f32,
    pub scale_min_zone: f32,
    #[serde(default)]
    pub mover: Mover,
    /// Pins the aim mode for this profile. Absent follows the runtime
    /// [`crate::aim::AimMode`], which the event listener can change.
    #[serde(default)]
    pub aim_mode: Option<Mode>,
    #[serde(flatten)]
    smooth_keys: toml::Table,
    /// Tuning for the human-flick model; see [`SmoothConfig`].
    #[serde(skip)]
    pub smooth: SmoothConfig,
}

fn de_millis<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Duration, D::Error> {
    Ok(Duration::from_millis(u64::deserialize(d)?))
}

fn de_fourcc<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Option<u32>, D::Error> {
    let raw = Option::<String>::deserialize(d)?;
    match raw {
        Some(s) if !s.trim().is_empty() => {
            parse_fourcc(&s).map(Some).map_err(serde::de::Error::custom)
        }
        _ => Ok(None),
    }
}

fn de_output_format<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<OutputFormat, D::Error> {
    OutputFormat::parse(&String::deserialize(d)?).map_err(serde::de::Error::custom)
}

fn default_event_listener_port() -> u16 {
    10000
}
fn default_ndi_timeout() -> Duration {
    Duration::from_millis(1000)
}
fn default_screen_width() -> u32 {
    1920
}
fn default_screen_height() -> u32 {
    1080
}
fn default_model_provider() -> String {
    String::from("cpu")
}
fn default_capture_device() -> String {
    String::from("/dev/video0")
}
fn default_capture_fps() -> u32 {
    60
}
fn default_capture_buffers() -> u32 {
    3
}
fn default_capture_timeout() -> Duration {
    Duration::from_millis(1000)
}
fn default_output_format() -> OutputFormat {
    OutputFormat::Bgr
}
fn default_openvino_device_type() -> String {
    String::from("CPU")
}
fn default_intra_threads() -> usize {
    1
}
fn default_makcu_baud() -> u32 {
    115_200
}
fn default_mouse_dpi() -> f64 {
    1000.
}
fn default_game_sens() -> f64 {
    1.
}
fn enabled() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default = "default_event_listener_port")]
    pub event_listener_port: u16,

    pub source_stream: String,
    #[serde(default)]
    pub ndi_source_name: Option<String>,
    #[serde(
        default = "default_ndi_timeout",
        rename = "ndi_timeout_ms",
        deserialize_with = "de_millis"
    )]
    pub ndi_timeout: Duration,
    #[serde(default = "default_screen_width")]
    pub screen_width: u32,
    #[serde(default = "default_screen_height")]
    pub screen_height: u32,
    /// Geometry of the frames that actually arrive, which is what `REGION_*`
    /// and every detection are measured in. See [`frame_geometry`].
    #[serde(skip)]
    pub frame_width: u32,
    #[serde(skip)]
    pub frame_height: u32,
    /// Frame pixel -> screen pixel, per axis. `(1.0, 1.0)` when the capture
    /// already matches the screen.
    #[serde(skip)]
    pub frame_to_screen: (f64, f64),
    #[serde(default)]
    pub region_top: u32,
    #[serde(default)]
    pub region_left: u32,
    #[serde(default)]
    pub region_width: u32,
    #[serde(default)]
    pub region_height: u32,

    pub esp_button_1_profile: usize,
    pub esp_button_2_profile: usize,
    #[serde(rename = "profile")]
    pub profiles: Vec<Profile>,

    #[serde(default = "default_model_provider")]
    pub model_provider: String,
    pub model_path: PathBuf,
    pub model_input_size: usize,
    pub model_conf_body: f32,
    pub model_conf_head: f32,
    pub model_iou: f32,
    #[serde(default)]
    pub build_head_iou: Option<f32>,

    /// True for the `v_light_*` models, which need the nearest-neighbour
    /// stretch preprocess and the raw multi-class output decoder.
    #[serde(skip)]
    pub light_model: bool,

    #[serde(default = "default_capture_device")]
    pub capture_device: String,
    #[serde(default = "default_screen_width")]
    pub capture_width: u32,
    #[serde(default = "default_screen_height")]
    pub capture_height: u32,
    #[serde(default = "default_capture_fps")]
    pub capture_fps: u32,
    #[serde(default, deserialize_with = "de_fourcc")]
    pub capture_fourcc: Option<u32>,
    #[serde(default = "default_capture_buffers")]
    pub capture_buffers: u32,
    #[serde(default = "enabled")]
    pub capture_drop_stale: bool,
    #[serde(
        default = "default_capture_timeout",
        rename = "capture_timeout_ms",
        deserialize_with = "de_millis"
    )]
    pub capture_timeout: Duration,
    #[serde(default = "default_output_format", deserialize_with = "de_output_format")]
    pub capture_output: OutputFormat,
    /// Hand only `REGION_*` downstream instead of the whole frame. The
    /// detection pipeline never looks outside that region, so converting or
    /// JPEG-decoding the rest is pure latency. Turn it off for the `debug`
    /// feature's whole-frame bbox overlay.
    #[serde(default = "enabled")]
    pub capture_roi: bool,

    #[serde(default)]
    pub gpu_id: Option<i32>,
    #[serde(default)]
    pub gpu_mem_limit: Option<usize>,
    #[serde(default)]
    pub trt_min_shapes: String,
    #[serde(default)]
    pub trt_opt_shapes: String,
    #[serde(default)]
    pub trt_max_shapes: String,
    #[serde(default)]
    pub trt_fp16: Option<bool>,
    #[serde(default)]
    pub trt_max_partition_iterations: Option<u32>,
    #[serde(default)]
    pub trt_builder_optimization_level: Option<u8>,
    #[serde(default)]
    pub trt_dla_enable: Option<bool>,
    #[serde(default)]
    pub trt_dla_core: Option<u32>,
    #[serde(default)]
    pub trt_auxiliary_streams: Option<i8>,
    #[serde(default)]
    pub trt_cache_dir: String,

    #[serde(default)]
    pub openvino_cache_dir: String,
    #[serde(default = "default_openvino_device_type")]
    pub openvino_device_type: String,
    #[serde(default = "default_intra_threads")]
    pub intra_threads: usize,

    pub makcu_port: String,
    #[serde(default = "default_makcu_baud")]
    pub makcu_baud: u32,
    #[serde(default)]
    pub makcu_listen: bool,
    #[serde(default = "default_mouse_dpi")]
    pub mouse_dpi: f64,
    #[serde(default = "default_game_sens")]
    pub game_sens: f64,

    #[serde(default)]
    pub esp_port: Option<String>,
    #[serde(default)]
    pub default_aim_mode: Option<u8>,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read config file {}", path.display()))?;
        let mut config: Self = toml::from_str(&text)
            .with_context(|| format!("{} is not a valid config", path.display()))?;
        config.resolve()?;
        Ok(config)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let mut config: Self = toml::from_str(text)?;
        config.resolve()?;
        Ok(config)
    }

    fn resolve(&mut self) -> Result<()> {
        let ((frame_width, frame_height), frame_to_screen) = frame_geometry(
            &self.source_stream,
            (self.screen_width, self.screen_height),
            (self.capture_width, self.capture_height),
        );
        self.frame_width = frame_width;
        self.frame_height = frame_height;
        self.frame_to_screen = frame_to_screen;
        if (frame_width, frame_height) != (self.screen_width, self.screen_height) {
            tracing::info!(
                "[Config] frames arrive at {}x{} while the game renders at {}x{}; \
                 region_* and detections are in frame pixels and mouse deltas are \
                 scaled by {:.4}x/{:.4}y",
                frame_width,
                frame_height,
                self.screen_width,
                self.screen_height,
                frame_to_screen.0,
                frame_to_screen.1,
            );
        }

        self.light_model = self
            .model_path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.to_ascii_lowercase().starts_with("v_light"));
        if !self.model_path.is_file() {
            bail!("model_path {} is not a file", self.model_path.display());
        }

        if !self.game_sens.is_finite() || self.game_sens <= 0. {
            bail!("game_sens must be greater than zero");
        }
        if !self.mouse_dpi.is_finite() || self.mouse_dpi <= 0. {
            bail!("mouse_dpi must be greater than zero");
        }

        if self.profiles.is_empty() {
            bail!("at least one [[profile]] is required");
        }
        for (label, index) in [
            ("esp_button_1_profile", self.esp_button_1_profile),
            ("esp_button_2_profile", self.esp_button_2_profile),
        ] {
            if index >= self.profiles.len() {
                bail!(
                    "{label} = {index} but only {} profile(s) are defined",
                    self.profiles.len()
                );
            }
        }
        for profile in &mut self.profiles {
            let keys = std::mem::take(&mut profile.smooth_keys);
            profile.smooth = SmoothConfig::deserialize(toml::Value::Table(keys))
                .with_context(|| format!("profile {:?}", profile.name))?;
            if profile.fov <= 0. {
                profile.fov = frame_width as f32;
            }
        }
        Ok(())
    }

    pub fn profile(&self, index: usize) -> &Profile {
        &self.profiles[index]
    }

    pub fn profile_for(&self, esp_button_2_pressed: bool) -> (usize, &Profile) {
        let index = if esp_button_2_pressed {
            self.esp_button_2_profile
        } else {
            self.esp_button_1_profile
        };
        (index, &self.profiles[index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with(extra: &str) -> String {
        format!(
            "source_stream = \"elgato://\"\n\
             model_path = \"Cargo.toml\"\n\
             model_input_size = 640\n\
             model_conf_body = 0.5\n\
             model_conf_head = 0.5\n\
             model_iou = 0.5\n\
             makcu_port = \"/dev/ttyACM0\"\n\
             esp_button_1_profile = 0\n\
             esp_button_2_profile = 1\n\
             {extra}\n"
        )
    }

    const TWO_PROFILES: &str = r#"
[[profile]]
name = "tracking"
fov = 35.0
scale_min_zone = 0.5
move_smooth_gain = 0.75
move_smooth_auto_click = true

[[profile]]
name = "flick"
fov = 300.0
scale_min_zone = 0.8
mover = "bezier"
aim_mode = "horizon"
"#;

    #[test]
    fn a_profile_carries_its_own_aim_and_movement_tuning() {
        let config = Config::parse(&with(TWO_PROFILES)).unwrap();
        assert_eq!(config.profiles.len(), 2);

        let tracking = config.profile(config.esp_button_1_profile);
        assert_eq!(tracking.name, "tracking");
        assert_eq!(tracking.fov, 35.);
        assert_eq!(tracking.scale_min_zone, 0.5);
        assert_eq!(tracking.mover, Mover::Smooth);
        assert_eq!(tracking.aim_mode, None);
        assert_eq!(tracking.smooth.gain, 0.75);
        assert!(tracking.smooth.auto_click);

        let flick = config.profile(config.esp_button_2_profile);
        assert_eq!(flick.mover, Mover::Bezier);
        assert_eq!(flick.aim_mode, Some(Mode::Horizon));
        assert_eq!(flick.smooth.gain, SmoothConfig::default().gain);
    }

    /// ESP button 2 picks its own profile; everything else — button 1, the
    /// MAKCU side-4 button and no-trigger auto aim — shares button 1's.
    #[test]
    fn esp_button_two_is_the_only_thing_that_switches_profile() {
        let config = Config::parse(&with(TWO_PROFILES)).unwrap();
        let (held, _) = config.profile_for(true);
        let (idle, _) = config.profile_for(false);
        assert_eq!(held, config.esp_button_2_profile);
        assert_eq!(idle, config.esp_button_1_profile);
        assert_eq!(config.profile_for(true).1.name, "flick");
        assert_eq!(config.profile_for(false).1.name, "tracking");

        // Both buttons may point at the same profile, which is how you turn
        // the split off without deleting one.
        let same = with(TWO_PROFILES).replace("esp_button_2_profile = 1", "esp_button_2_profile = 0");
        let config = Config::parse(&same).unwrap();
        assert_eq!(config.profile_for(true).0, config.profile_for(false).0);
    }

    #[test]
    fn a_profile_that_names_no_tuning_gets_the_defaults() {
        let src = with("[[profile]]\nname = \"a\"\nscale_min_zone = 0.5\n\n\
                        [[profile]]\nname = \"b\"\nscale_min_zone = 0.5\n");
        let config = Config::parse(&src).unwrap();
        assert_eq!(config.profiles[0].smooth, SmoothConfig::default());
        assert_eq!(config.profiles[0].mover, Mover::Smooth);
        assert_eq!(config.profiles[0].aim_mode, None);
    }

    /// `fov` is compared against a frame-pixel distance, so "unset" has to mean
    /// the whole frame rather than zero, or nothing would ever be in range.
    #[test]
    fn an_unset_fov_covers_the_whole_frame() {
        let src = with(
            "screen_width = 2560\nscreen_height = 1440\n\
             capture_width = 1920\ncapture_height = 1080\n\n\
             [[profile]]\nname = \"a\"\nscale_min_zone = 0.5\n\n\
             [[profile]]\nname = \"b\"\nscale_min_zone = 0.5\nfov = 0.0\n",
        );
        let config = Config::parse(&src).unwrap();
        assert_eq!(config.frame_width, 1920);
        assert_eq!(config.profiles[0].fov, 1920.);
        assert_eq!(config.profiles[1].fov, 1920.);
    }

    #[test]
    fn a_button_pointing_past_the_last_profile_is_refused() {
        let src = with("[[profile]]\nname = \"only\"\nscale_min_zone = 0.5\n");
        let err = Config::parse(&src).unwrap_err().to_string();
        assert!(err.contains("esp_button_2_profile"), "{err}");
        assert!(err.contains("1 profile"), "{err}");
    }

    #[test]
    fn a_config_with_no_profiles_is_refused() {
        let err = Config::parse(&with("profile = []")).unwrap_err().to_string();
        assert!(err.contains("profile"), "{err}");
    }

    #[test]
    fn a_zero_sensitivity_is_refused_rather_than_freezing_the_aim() {
        for (key, value) in [("game_sens", "0.0"), ("mouse_dpi", "0.0")] {
            let src = with(&format!("{key} = {value}\n{TWO_PROFILES}"));
            let err = Config::parse(&src).unwrap_err().to_string();
            assert!(err.contains(key), "{err}");
        }
    }

    /// The `.env` this replaces had exactly this bug: it set `SCALE_MIN_ZONE`
    /// while the code read `SCALE_MIN_ZONE1`, so the setting did nothing and
    /// nothing said so. A mistyped key has to be an error.
    #[test]
    fn a_mistyped_key_is_refused_instead_of_silently_doing_nothing() {
        let in_profile = with(
            "[[profile]]\nname = \"a\"\nscale_min_zone = 0.5\nmove_smoth_gain = 0.5\n\n\
             [[profile]]\nname = \"b\"\nscale_min_zone = 0.5\n",
        );
        assert!(
            Config::parse(&in_profile).is_err(),
            "a typo inside [[profile]] was accepted"
        );

        let at_root = with(&format!("screen_wdith = 2560\n{TWO_PROFILES}"));
        assert!(
            Config::parse(&at_root).is_err(),
            "a typo in the common section was accepted"
        );
    }

    /// Stops the shipped example rotting away from the schema it documents.
    #[test]
    fn the_shipped_example_config_parses() {
        let text = std::fs::read_to_string("config.example.toml")
            .unwrap()
            .replace(
                "model_path = \"assets/v_light_192_fp16_onnx.onnx\"",
                "model_path = \"Cargo.toml\"",
            );
        let config = Config::parse(&text).unwrap();
        assert_eq!(config.profiles.len(), 2);
        assert!(config.esp_button_1_profile < config.profiles.len());
        assert!(config.esp_button_2_profile < config.profiles.len());
    }

    #[test]
    fn an_unknown_mover_or_aim_mode_is_refused() {
        for bad in ["mover = \"teleport\"", "aim_mode = \"elbow\""] {
            let src = with(&format!(
                "[[profile]]\nname = \"a\"\nscale_min_zone = 0.5\n{bad}\n\n\
                 [[profile]]\nname = \"b\"\nscale_min_zone = 0.5\n"
            ));
            assert!(Config::parse(&src).is_err(), "{bad} was accepted");
        }
    }

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
