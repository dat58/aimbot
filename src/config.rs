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

/// How the game turns mouse counts into a view rotation.
///
/// Everything here is a fact about the game, not about the mouse. The mouse's
/// DPI is deliberately absent: the game never sees it. It receives counts and
/// multiplies them by its own per-count yaw constant, so how many counts a
/// given turn is worth cannot depend on the hardware.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Optics {
    /// Width of the space the delta is measured in — the frame, which covers
    /// the whole screen and therefore the whole field of view. Both models
    /// normalise by it, and they only agree with each other when they use the
    /// same one.
    pub width: f64,
    /// Horizontal field of view in degrees, at the rendered aspect ratio.
    pub hfov: f64,
    /// Degrees of view rotation per mouse count at sensitivity 1.
    pub yaw: f64,
    /// The in-game sensitivity setting, as typed into the game.
    pub sens: f64,
    /// Correction on top of [`Self::yaw`], applied last.
    ///
    /// `1` for the game this was calibrated against, since its `game_yaw` is
    /// the measured one. Another game whose yaw constant is not known can
    /// leave `game_yaw` alone and bisect this instead: halve it while the aim
    /// sails past, double it while the aim falls short, and the value it
    /// settles on is the ratio between the two games' constants.
    pub scale: f64,
}

impl Optics {
    /// Pinhole focal length, in the same pixels as [`Self::width`].
    pub fn focal(&self) -> f64 {
        (self.width / 2.) / (self.hfov.to_radians() / 2.).tan()
    }

    /// Degrees of turn one count is worth.
    pub fn deg_per_count(&self) -> f64 {
        self.yaw * self.sens
    }

}

/// Mouse counts for a target `frame_delta` pixels from the crosshair.
///
/// Pixels are a perspective projection of the view angle, so the relation is
/// an arctangent rather than a scale factor. The bend is small close in — at
/// a 103° field of view and a 1920-wide frame it is 0.14% at 50 px — and only
/// becomes significant further out: 2.2% at 200 px, 23% at 800 px.
///
/// `mouse.dpi` is deliberately absent. The game never sees it, so how many
/// counts a turn is worth cannot depend on the hardware; its one legitimate
/// use is the hand-speed envelope in [`crate::mouse::mover::smooth`].
///
/// # Which width, and why `aim_scale` exists
///
/// The delta is in frame pixels and [`Optics::width`] is the frame width.
/// That pairing is the self-consistent one: the capture carries the whole
/// screen, so the frame spans the whole field of view, and the projection
/// only cares that the delta and the width describe the same space.
///
/// What the projection cannot supply is the *constant*. `game_yaw` has to
/// match the engine, and a wrong value scales every turn by a fixed factor.
/// [`Optics::scale`] is the measured correction for exactly that, and it is
/// the only number here that can be established in the game being played
/// rather than looked up.
pub fn mouse_counts(frame_delta: (f32, f32), optics: Optics) -> (f64, f64) {
    let dx = frame_delta.0 as f64;
    let dy = frame_delta.1 as f64;
    let focal = optics.focal();
    let yaw_deg = dx.atan2(focal).to_degrees();
    // The pitch is measured off the already-yawed ray, so its lever arm is
    // `hypot(focal, dx)` rather than `focal`. Using `focal` would
    // under-rotate y for any target also off centre horizontally.
    let pitch_deg = dy.atan2(focal.hypot(dx)).to_degrees();
    let deg_per_count = optics.deg_per_count();
    (
        yaw_deg / deg_per_count * optics.scale,
        pitch_deg / deg_per_count * optics.scale,
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
/// [`crate::mouse::mover::smooth::SmoothMover`].
///
/// Whether that model is used at all is not decided here: each [`Profile`]
/// picks its own [`MoverKind`].
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

/// Which movement a profile is played with. The name here selects the struct
/// that gets loaded; see [`crate::mouse::mover::build`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoverKind {
    #[default]
    Smooth,
    Bezier,
    /// Sends the delta as it arrives, with no model over the top. Still reads
    /// the `move_smooth_auto_click_*` keys, which is all it takes from
    /// [`SmoothConfig`].
    Direct,
}

impl std::fmt::Display for MoverKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoverKind::Smooth => write!(f, "smooth"),
            MoverKind::Bezier => write!(f, "bezier"),
            MoverKind::Direct => write!(f, "direct"),
        }
    }
}

/// Which device actually holds the mouse. The name here selects the struct
/// that gets loaded; see [`crate::mouse::connect`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    #[default]
    Makcu,
    Medius,
}

impl std::fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceKind::Makcu => write!(f, "makcu"),
            DeviceKind::Medius => write!(f, "medius"),
        }
    }
}

/// How a Medius box renders injected motion. Ignored by every other device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderMode {
    /// Even fill at the paced rate, with the model out of the path.
    Off,
    /// The bit-exact triangular smoother.
    Stock,
    /// The smoother with a ramped onset. What the box itself boots in.
    #[default]
    Despiked,
    /// No smoother: the model gets the raw injection.
    Unsmoothed,
}

/// Everything about the link to the physical mouse, which is a fact about the
/// hardware rather than about the game. Kept in its own table so a device that
/// is not a MAKCU does not have to read keys named after one.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MouseConfig {
    #[serde(default)]
    pub device: DeviceKind,
    pub port: String,
    #[serde(default = "default_mouse_baud")]
    pub baud: u32,
    /// Read the physical mouse's own button reports back off the link, which
    /// is what makes [`crate::mouse::MouseDevice::is_pressing`] answer with
    /// anything.
    #[serde(default)]
    pub listen: bool,
    /// Counts per inch of physical travel. Not a tuning knob: it is what the
    /// mouse is set to, and the movement model needs it to turn an amplitude
    /// into a hand speed.
    #[serde(default = "default_mouse_dpi")]
    pub dpi: f64,
    /// Medius only; see [`RenderMode`].
    #[serde(default)]
    pub render: RenderMode,
    /// Render the operator's own motion too, instead of relaying it
    /// untouched. The box boots with this off. Medius only.
    #[serde(default)]
    pub render_full: bool,
    /// Blank the operator's own motion for the frame an injected move lands
    /// in, so the two cannot add up. Medius only.
    #[serde(default)]
    pub lock_move: bool,
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
    pub mover: MoverKind,
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
fn default_mouse_baud() -> u32 {
    115_200
}
fn default_mouse_dpi() -> f64 {
    1000.
}
fn default_game_sens() -> f64 {
    1.
}
fn default_game_hfov() -> f64 {
    103.
}
fn default_game_yaw() -> f64 {
    1.
}
fn default_aim_scale() -> f64 {
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
    ///
    /// Nothing in the pipeline reads this any more: [`mouse_counts`] projects
    /// in frame space, and Fitts reads only a ratio, so both cancel the
    /// factor out. Kept as a description of the geometry.
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

    pub mouse: MouseConfig,
    #[serde(default = "default_game_sens")]
    pub game_sens: f64,
    /// Horizontal field of view the game renders at, in degrees. Must match
    /// the game, and must be the *horizontal* one at the actual aspect ratio.
    #[serde(default = "default_game_hfov")]
    pub game_hfov: f64,
    /// Degrees of view rotation one mouse count is worth at sensitivity 1 —
    /// the engine's own constant (`m_yaw` in Source-derived engines). Must
    /// match the game; getting it wrong scales every turn by a fixed factor.
    ///
    /// The default of `1` was measured in game rather than looked up: one
    /// count turns [`Self::game_sens`] degrees, so `deg_per_count == sens`.
    #[serde(default = "default_game_yaw")]
    pub game_yaw: f64,
    /// Correction on top of [`Self::game_yaw`]; see [`Optics::scale`]. `1`
    /// unless the game's yaw constant differs from the measured default.
    #[serde(default = "default_aim_scale")]
    pub aim_scale: f64,

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
                 region_* and detections are in frame pixels",
                frame_width,
                frame_height,
                self.screen_width,
                self.screen_height,
            );
        }

        // The delta arrives in frame pixels and both aim models scale x and y
        // by the same factor. That is only right when a frame pixel is the
        // same fraction of the screen on both axes, which is equal aspect
        // ratios.
        let frame_aspect = frame_width as f64 / frame_height as f64;
        let screen_aspect = self.screen_width as f64 / self.screen_height as f64;
        if (frame_aspect - screen_aspect).abs() > 1e-3 {
            tracing::warn!(
                "[Config] frames are {frame_width}x{frame_height} ({frame_aspect:.4}) \
                 but the game renders {}x{} ({screen_aspect:.4}); a non-uniform \
                 scale makes the vertical aim off by {:.1}%",
                self.screen_width,
                self.screen_height,
                (screen_aspect / frame_aspect - 1.) * 100.,
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
        // A field of view at or past 180 degrees has no finite focal length,
        // and a non-positive yaw would make every turn infinite.
        if !self.game_hfov.is_finite() || self.game_hfov <= 0. || self.game_hfov >= 180. {
            bail!("game_hfov must be between zero and 180 degrees");
        }
        if !self.game_yaw.is_finite() || self.game_yaw <= 0. {
            bail!("game_yaw must be greater than zero");
        }
        if !self.aim_scale.is_finite() || self.aim_scale <= 0. {
            bail!("aim_scale must be greater than zero");
        }
        if !self.mouse.dpi.is_finite() || self.mouse.dpi <= 0. {
            bail!("mouse.dpi must be greater than zero");
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

    /// How this game maps a *frame* offset onto a view rotation; see
    /// [`mouse_counts`]. Only meaningful after [`Self::resolve`] has filled in
    /// the frame geometry.
    pub fn optics(&self) -> Optics {
        Optics {
            width: self.frame_width as f64,
            hfov: self.game_hfov,
            yaw: self.game_yaw,
            sens: self.game_sens,
            scale: self.aim_scale,
        }
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

    /// `[mouse]` is emitted *after* `extra` on purpose: a TOML header swallows
    /// every key that follows it, so a table in the middle would capture
    /// whatever root keys a caller appended.
    fn with(extra: &str) -> String {
        format!(
            "source_stream = \"elgato://\"\n\
             model_path = \"Cargo.toml\"\n\
             model_input_size = 640\n\
             model_conf_body = 0.5\n\
             model_conf_head = 0.5\n\
             model_iou = 0.5\n\
             esp_button_1_profile = 0\n\
             esp_button_2_profile = 1\n\
             {extra}\n\
             \n[mouse]\n\
             port = \"/dev/ttyACM0\"\n"
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
        assert_eq!(tracking.mover, MoverKind::Smooth);
        assert_eq!(tracking.aim_mode, None);
        assert_eq!(tracking.smooth.gain, 0.75);
        assert!(tracking.smooth.auto_click);

        let flick = config.profile(config.esp_button_2_profile);
        assert_eq!(flick.mover, MoverKind::Bezier);
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
        assert_eq!(config.profiles[0].mover, MoverKind::Smooth);
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
        let src = with(&format!("game_sens = 0.0\n{TWO_PROFILES}"));
        let err = Config::parse(&src).unwrap_err().to_string();
        assert!(err.contains("game_sens"), "{err}");

        let src = with(TWO_PROFILES).replace("[mouse]", "[mouse]\ndpi = 0.0");
        let err = Config::parse(&src).unwrap_err().to_string();
        assert!(err.contains("dpi"), "{err}");
    }

    /// The device name picks which struct is loaded, so a name nothing answers
    /// to has to fail at load rather than at the first move.
    #[test]
    fn an_unknown_mouse_device_is_refused() {
        let src = with(TWO_PROFILES).replace("[mouse]", "[mouse]\ndevice = \"razer\"");
        assert!(Config::parse(&src).is_err(), "an unknown device was accepted");

        let src = with(TWO_PROFILES).replace("[mouse]", "[mouse]\ndevice = \"makcu\"");
        let config = Config::parse(&src).unwrap();
        assert_eq!(config.mouse.device, DeviceKind::Makcu);
        assert_eq!(config.mouse.baud, 115_200);
        assert_eq!(config.mouse.dpi, 1000.);
        assert!(!config.mouse.listen);
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

    fn optics(sens: f64) -> Optics {
        Optics {
            width: 1920.,
            hfov: 103.,
            yaw: 0.07,
            sens,
            scale: 1.,
        }
    }

    /// The one knob that can be measured in the game actually being played,
    /// so nothing may swallow it.
    #[test]
    fn aim_scale_multiplies_every_move() {
        let one = optics(0.38);
        let quarter = Optics { scale: 0.25, ..one };
        let (ax, ay) = mouse_counts((137., -42.), one);
        let (bx, by) = mouse_counts((137., -42.), quarter);
        assert!((bx - ax * 0.25).abs() < 1e-9, "{bx} vs {ax}");
        assert!((by - ay * 0.25).abs() < 1e-9, "{by} vs {ay}");
    }

    /// `game_yaw` was measured in game: one count turns `game_sens` degrees,
    /// which is `yaw == 1`. It replaced a looked-up `0.07` that needed
    /// `aim_scale = 0.07` to cancel it back out. Both say the same thing, and
    /// this pins that they do — the measurement is the reason for the
    /// default, so it should fail loudly if someone changes it back.
    #[test]
    fn the_measured_yaw_and_the_scale_that_cancelled_it_agree() {
        let measured = Optics {
            yaw: 1.0,
            scale: 1.0,
            ..optics(0.38)
        };
        let cancelled = Optics {
            yaw: 0.07,
            scale: 0.07,
            ..optics(0.38)
        };
        for px in [10.0f32, 35., 50., 200.] {
            let a = mouse_counts((px, 0.), measured);
            let b = mouse_counts((px, 0.), cancelled);
            assert!(
                (a.0 - b.0).abs() <= 1e-9 * a.0.abs(),
                "{px} px: {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn one_count_turns_the_sensitivity_in_degrees_by_default() {
        let config = Config::parse(&with(&format!("game_sens = 0.38\n{TWO_PROFILES}"))).unwrap();
        assert_eq!(config.game_yaw, 1.);
        assert_eq!(config.aim_scale, 1.);
        let per_count = config.optics().deg_per_count();
        assert!(
            (per_count - config.game_sens).abs() < 1e-12,
            "{per_count} vs {}",
            config.game_sens
        );
    }

    #[test]
    fn a_non_positive_aim_scale_is_refused() {
        for bad in ["aim_scale = 0.0", "aim_scale = -2.0"] {
            let src = with(&format!("{bad}\n{TWO_PROFILES}"));
            assert!(Config::parse(&src).is_err(), "{bad} was accepted");
        }
    }

    /// Reference values from the projection this is a port of. Nothing about
    /// the mouse enters here, which is the whole point: the game receives
    /// counts and converts them with its own yaw constant.
    #[test]
    fn mouse_counts_match_the_projective_reference() {
        assert!((optics(1.).focal() - 763.6184800011).abs() < 1e-9);

        for (delta, sens, want) in [
            ((100.0f32, -50.0f32), 2.0, (53.2909920090, -26.5330230069)),
            ((30., 40.), 1.5, (21.4266794049, 28.5355246879)),
            ((800., 0.), 0.38, (1741.8379715151, 0.)),
            ((20., 0.), 0.38, (56.4021015477, 0.)),
        ] {
            let (dx, dy) = mouse_counts(delta, optics(sens));
            assert!((dx - want.0).abs() < 1e-6, "{delta:?} x: {dx} vs {}", want.0);
            assert!((dy - want.1).abs() < 1e-6, "{delta:?} y: {dy} vs {}", want.1);
        }
    }

    /// The reason the old scale-factor form was wrong: screen pixels are a
    /// perspective projection of the view angle, so they compress away from
    /// the centre. A linear conversion over-turns on every long flick.
    #[test]
    fn counts_grow_slower_than_the_pixel_offset() {
        let near = mouse_counts((400., 0.), optics(1.)).0;
        let far = mouse_counts((800., 0.), optics(1.)).0;
        let ratio = far / near;
        assert!(
            (ratio - 1.6759).abs() < 1e-3,
            "twice the offset is {ratio:.4} times the turn, not 2"
        );
        assert!(ratio < 2., "a linear conversion would give exactly 2");
    }

    /// `mouse.dpi` describes the hand, not the game. It must not reach this.
    #[test]
    fn counts_do_not_depend_on_the_mouse_dpi() {
        let src = with(TWO_PROFILES);
        let a = Config::parse(&src.replace("[mouse]", "[mouse]\ndpi = 400.0")).unwrap();
        let b = Config::parse(&src.replace("[mouse]", "[mouse]\ndpi = 1600.0")).unwrap();
        assert_ne!(a.mouse.dpi, b.mouse.dpi);
        assert_eq!(
            mouse_counts((137., -42.), a.optics()),
            mouse_counts((137., -42.), b.optics())
        );
    }

    #[test]
    fn mouse_counts_are_symmetric_and_zero_at_the_crosshair() {
        assert_eq!(mouse_counts((0., 0.), optics(1.5)), (0., 0.));
        let (px, py) = mouse_counts((30., 40.), optics(1.5));
        let (nx, ny) = mouse_counts((-30., -40.), optics(1.5));
        assert!((px + nx).abs() < 1e-12 && (py + ny).abs() < 1e-12);
    }

    /// The projection is scale-free: measuring the delta in another pixel
    /// space and taking the width from that same space gives the identical
    /// angle, because `atan2` sees only the ratio and both sides carry the
    /// same factor. What that does **not** license is mixing the two — a
    /// frame delta against a screen width is a different answer, which is the
    /// bug this note exists to warn about.
    #[test]
    fn the_projection_is_free_of_the_pixel_space_when_both_sides_agree() {
        let (frame, scale) = frame_geometry("elgato://", (2560, 1440), (1920, 1080));
        let in_frame = Optics {
            width: frame.0 as f64,
            hfov: 103.,
            yaw: 0.07,
            sens: 1.5,
            scale: 1.,
        };
        let in_screen = Optics {
            width: 2560.,
            ..in_frame
        };
        // The identity is exact in real arithmetic; what is left is the f32
        // rounding of the scaled delta, which is relative, so the tolerance
        // has to be too.
        let near = |a: f64, b: f64| (a - b).abs() <= 1e-6 * a.abs().max(1.);
        for d in [(1.0f32, 0.5f32), (137., -42.), (800., 300.), (-60., 0.)] {
            let here = mouse_counts(d, in_frame);
            let there = mouse_counts(
                (d.0 * scale.0 as f32, d.1 * scale.1 as f32),
                in_screen,
            );
            assert!(near(here.0, there.0), "{d:?}: {here:?} vs {there:?}");
            assert!(near(here.1, there.1), "{d:?}: {here:?} vs {there:?}");
        }
    }

    #[test]
    fn a_degenerate_field_of_view_or_yaw_is_refused() {
        for bad in ["game_hfov = 0.0", "game_hfov = 180.0", "game_yaw = 0.0"] {
            let src = with(&format!("{bad}\n{TWO_PROFILES}"));
            assert!(Config::parse(&src).is_err(), "{bad} was accepted");
        }
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
            let ((frame_w, frame_h), _) = frame_geometry("elgato://", screen, capture);
            let crosshair = (frame_w as f32 / 2., frame_h as f32 / 2.);
            let delta = (
                crosshair.0 - frame_w as f32 / 2.,
                crosshair.1 - frame_h as f32 / 2.,
            );
            let mut o = optics(1.0);
            o.width = frame_w as f64;
            assert_eq!(mouse_counts(delta, o), (0., 0.));
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
