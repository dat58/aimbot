//! Movement driven by [`abcurves`](https://github.com/dat58/abcurves-rs), a
//! neural model trained on real human mouse data.
//!
//! The split against [`crate::mouse`]'s `move_smooth` is deliberate. abcurves
//! *is* the motion model, so everything about the path — duration, the speed
//! profile, curvature, tremor, how many reports a stroke is worth — comes from
//! it. What stays on this side is everything that is about the *target* rather
//! than the path: the ballistic undershoot that leaves the next frame something
//! to correct, the reaction window, and the whole auto-click family, which it
//! shares with `move_smooth` through [`crate::mouse::AutoClick`].
//!
//! abcurves is a continuous stream: you push targets at it and pull reports on
//! a 1 ms grid, and it is built to be retargeted in flight. So it is driven as
//! one: a single pipeline lives for the whole session and the worker pumps it
//! on the sample grid, emitting each report as it falls due. A new aim is a
//! `retarget`, which lands within a millisecond instead of waiting out whatever
//! was already planned.
//!
//! Cutting finite flicks out of the stream instead — planning a window, playing
//! it, then re-planning — was the first shape this took, and it forced a
//! 240 ms window: strokes run 110-200 ms and one that emits nothing inside its
//! window has to be abandoned, so anything shorter stalled small corrections.
//! That window was 240 ms during which the worker never looked at the queue.
//! Pumping has no such window.

use crate::mouse::{AutoClick, ClickCfg, MouseVirtual, gate_reaction, gauss};
use abcurves::continuous::{ContinuousPipeline, CountTransform, PipelineOptions};
use anyhow::{Context, Result, bail};
use rand::prelude::*;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// abcurves plans on a fixed 1 ms grid (`SAMPLE_US` in the crate).
const SAMPLE_US: i64 = 1000;
/// The renderer profile is this many reports, and the crate rejects any other
/// length outright.
const PROFILE_REPORTS: usize = 256;
/// Consecutive all-zero reports that count as the hand having come to rest.
///
/// This is only the fallback for a stroke that stops short of where it was
/// aimed; arriving is detected from the counts still owed, which is the signal
/// that actually works. It has to be generous because abcurves' reports are
/// sparse: measured on this hardware a 15-count stroke emits 20 non-zero
/// reports across 110 ms — 18% density — with runs of 17 zeros mid-stroke
/// while the sub-count residue accumulates. At 8 a stroke was called finished
/// after one count.
const REST_RUN: usize = 32;
/// Windows API DPI reference, the same constant `mouse_counts` divides by.
const WIN_DPI: f64 = 96.;

/// Tuning for [`AbAim`]. Everything here is either an abcurves input or one of
/// the target-level behaviours carried over from `move_smooth`; nothing
/// describes the path, because the model owns that.
#[derive(Debug, Clone)]
pub struct AbConfig {
    /// The game's horizontal field of view, degrees. Required: it is the only
    /// thing that turns our counts into the angular units abcurves thinks in.
    pub fov_deg: f64,
    pub profile_path: PathBuf,
    pub model_dir: PathBuf,
    /// How long one target may stay unreached before the stroke is given up on.
    ///
    /// A rail, not a tuning knob. The stream is pumped continuously and the aim
    /// loop retargets at frame rate, so a stroke that never arrives is a fault
    /// rather than a slow movement — measured strokes run 110-200 ms. It exists
    /// so a wedged planner cannot hold the auto click off forever.
    pub max_stroke_ms: u64,
    /// Fraction of the *remaining* distance the planner is pointed at when a
    /// reach begins, held for that reach.
    ///
    /// The aim loop re-measures every frame, so anything below 1 mostly damps
    /// how fast the loop closes rather than leaving a lasting undershoot;
    /// 1.0 is the fastest. The ballistic character comes from
    /// [`AbConfig::overshoot_p`] instead, which does leave the pointer past
    /// the target often enough to need a correction back.
    pub gain: f64,
    pub gain_sd: f64,
    /// How often a flick overshoots instead of falling short.
    /// Chance a reach is rolled with a gain above 1, so it carries the pointer
    /// past the target and the next reach has to come back.
    ///
    /// Only works because the gain is latched for the reach: re-rolling it per
    /// frame cancelled every overshoot 7 ms later, which is what once made
    /// this knob look inert. Measured over 40 reaches of 60 counts, counting
    /// those that crossed the target: 42% at 0.0, 48% at 0.5, 62% at 1.0, with
    /// the furthest going 1.23x, 1.38x and 1.57x the ask. The 42% floor is
    /// abcurves' own endpoint spread — this adds to it rather than creating it.
    pub overshoot_p: f64,
    pub react_min_ms: u64,
    pub react_max_ms: u64,
    /// Silence longer than this means the old target is gone.
    pub gap_ms: u64,
    /// Ceiling on a single report, in counts. Effectively off by default.
    ///
    /// `move_smooth` caps at 127 because that is an 8-bit mouse's report
    /// ceiling, but abcurves' renderer already emits device-realistic `i16`
    /// reports. Clamping them would flatten the peak of every fast stroke —
    /// reshaping the trajectory this model exists to get right — so the cap is
    /// kept only as a rail for firmware that chokes on large values.
    pub max_counts: i64,
    pub move_now: bool,
    pub auto_click: bool,
    pub auto_click_lower_ms: u64,
    pub auto_click_upper_ms: u64,
    pub auto_click_miss_p: f64,
    pub auto_click_rate_limit_ms: u64,
}

impl Default for AbConfig {
    fn default() -> Self {
        Self {
            fov_deg: 0., // no sane default; `AbAim::new` rejects it
            profile_path: PathBuf::from("assets/abcurves_profile.i16"),
            model_dir: PathBuf::from("assets/abcurves"),
            max_stroke_ms: 600,
            // 1.0 closes the loop as fast as the model allows. Lower only
            // damps it; see `gain_sets_the_closing_rate_not_the_endpoint`.
            gain: 1.0,
            gain_sd: 0.045,
            overshoot_p: 0.15,
            react_min_ms: 0,
            react_max_ms: 0,
            gap_ms: 250,
            max_counts: i16::MAX as i64,
            move_now: false,
            auto_click: false,
            auto_click_lower_ms: 100,
            auto_click_upper_ms: 130,
            auto_click_miss_p: 0.0,
            auto_click_rate_limit_ms: 0,
        }
    }
}

/// The per-session state behind [`AbAim::retarget`] and [`AbAim::pump`].
pub struct AbAim {
    cfg: AbConfig,
    pipeline: ContinuousPipeline,
    /// counts -> common units, per axis, from `CountTransform::common_per_native`.
    scale: [f64; 2],
    /// Origin for the microsecond clock `advance` is driven on.
    origin: Instant,
    /// Last timestamp handed to `advance`. Never goes backwards.
    planned_us: i64,
    last_flick: Option<Instant>,
    ready_at: Option<Instant>,
    last_click: Option<Instant>,
    /// The target being driven to, in common units, or `None` when the hand is
    /// at rest and there is nothing to pump.
    target: Option<[f64; 2]>,
    /// When the current target was set, for the stuck-stroke bound.
    started: Option<Instant>,
    /// The ballistic gain this reach was rolled with, held for its duration.
    gain: f64,
    settle: Settle,
}

/// What one turn of the pump did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pumped {
    /// No target: nothing to drive, so the caller may park.
    Idle,
    /// The sample is not due yet, or reports went out. Either way, come back.
    Moving,
    /// The aim arrived — the counts owed fell under one, or the hand came to
    /// rest. This is what a shot is fired behind.
    Arrived,
}

impl AbAim {
    /// Build the pipeline, or fail loudly saying what is missing.
    ///
    /// Everything abcurves needs is checked here rather than on the first
    /// flick: a mover that silently does nothing because a model directory is
    /// absent is far worse to diagnose than a process that refuses to start.
    pub fn new(cfg: AbConfig, screen_width: u32, game_sens: f64, mouse_dpi: f64) -> Result<Self> {
        if !(cfg.fov_deg.is_finite() && cfg.fov_deg > 0. && cfg.fov_deg < 180.) {
            bail!(
                "[AbCurves] MOVE_AB_FOV_DEG must be set to the game's horizontal FOV in \
                 degrees, between 0 and 180; got {}",
                cfg.fov_deg
            );
        }
        if screen_width == 0 {
            bail!("[AbCurves] SCREEN_WIDTH must be non-zero");
        }

        // abcurves thinks in angles, we think in counts, and only the FOV
        // bridges them: GAME_SENS and MOUSE_DPI relate pixels to counts but
        // carry nothing about how far the view turns. This is the inverse of
        // `config::mouse_counts` composed with radians-per-pixel.
        let rad_per_count = cfg.fov_deg.to_radians() * game_sens * mouse_dpi
            / (screen_width as f64 * WIN_DPI);
        if !(rad_per_count.is_finite() && rad_per_count > 0.) {
            bail!(
                "[AbCurves] derived radians-per-count is {rad_per_count}; check \
                 MOVE_AB_FOV_DEG, GAME_SENS, MOUSE_DPI and SCREEN_WIDTH"
            );
        }

        let profile = read_profile(&cfg.profile_path)?;
        if !cfg.model_dir.is_dir() {
            bail!(
                "[AbCurves] model directory {} not found. It ships with the abcurves \
                 repository under models/ (4.7 MB); copy it there or point \
                 MOVE_AB_MODEL_DIR somewhere else.",
                cfg.model_dir.display()
            );
        }

        // y_down because our dy grows downward like the screen does; the crate
        // flips the rendered report back for us.
        let transform = CountTransform::new(rad_per_count, true)
            .map_err(|e| anyhow::anyhow!("[AbCurves] count transform: {e}"))?;
        let scale = transform.common_per_native();

        // Seeded from the thread RNG rather than a constant: a fixed seed would
        // replay the same stream on every run, which is the repetition the
        // whole model exists to avoid.
        let mut random = rand::rng();
        let mut options = PipelineOptions::new(transform)
            .seed(random.random::<u64>())
            .renderer_seed(random.random::<u64>());
        options.model_dir = Some(cfg.model_dir.clone());

        let pipeline = ContinuousPipeline::load(&profile, options)
            .map_err(|e| anyhow::anyhow!("[AbCurves] loading the pipeline: {e}"))?;

        tracing::info!(
            "[AbCurves] ready: FOV {}°, {:.3e} rad/count ({:.2}x the model's reference), \
             profile {}, models {}",
            cfg.fov_deg,
            rad_per_count,
            rad_per_count / 0.000_350_948_708_328_061_8,
            cfg.profile_path.display(),
            cfg.model_dir.display(),
        );
        Ok(Self {
            cfg,
            pipeline,
            scale,
            origin: Instant::now(),
            planned_us: 0,
            last_flick: None,
            ready_at: None,
            last_click: None,
            target: None,
            started: None,
            gain: 1.,
            settle: Settle::default(),
        })
    }

    pub fn config(&self) -> &AbConfig {
        &self.cfg
    }

    /// Put the pipeline back in a usable state after a failed `advance`.
    ///
    /// The crate latches the failure — every later call returns the same error
    /// until a reset — so without this one numerical fault would silence the
    /// mover for the rest of the session.
    fn recover(&mut self, why: &str) {
        tracing::warn!("[AbCurves] {why}; resetting the pipeline");
        // Reseeds the renderer as well as the planner and clears the latched
        // failure, so the next flick starts from a clean, differently-seeded
        // stream rather than replaying whatever produced the fault.
        let mut random = rand::rng();
        if let Err(e) = self.pipeline.reset(
            Some(random.random::<u64>()),
            None,
            None,
            Some(random.random::<u64>()),
            None,
        ) {
            tracing::error!("[AbCurves] reset failed: {e}");
        }
        self.planned_us = self.origin.elapsed().as_micros() as i64;
    }
}

impl AutoClick for AbAim {
    fn click_cfg(&self) -> ClickCfg {
        ClickCfg {
            auto_click: self.cfg.auto_click,
            lower_ms: self.cfg.auto_click_lower_ms,
            upper_ms: self.cfg.auto_click_upper_ms,
            miss_p: self.cfg.auto_click_miss_p,
            rate_limit_ms: self.cfg.auto_click_rate_limit_ms,
        }
    }
    fn last_click(&self) -> Option<Instant> {
        self.last_click
    }
    fn set_last_click(&mut self, at: Instant) {
        self.last_click = Some(at);
    }
}

/// Read exactly [`PROFILE_REPORTS`] little-endian `[i16; 2]` pairs.
///
/// The length is a hard contract in the crate, so checking it here buys a
/// message that names the file instead of one about an inference contract.
fn read_profile(path: &Path) -> Result<Vec<[i16; 2]>> {
    let bytes = std::fs::read(path).with_context(|| {
        format!(
            "[AbCurves] cannot read the renderer profile {}. It ships with the abcurves \
             repository as examples/data/human_start/profile_hardware.i16",
            path.display()
        )
    })?;
    let want = PROFILE_REPORTS * 4;
    if bytes.len() != want {
        bail!(
            "[AbCurves] profile {} is {} bytes; it must be exactly {want} \
             ({PROFILE_REPORTS} reports of two little-endian i16)",
            path.display(),
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| {
            [
                i16::from_le_bytes([c[0], c[1]]),
                i16::from_le_bytes([c[2], c[3]]),
            ]
        })
        .collect())
}

/// Decides when one pull has run its course.
///
/// Split out and kept pure so the rule can be tested without model files, which
/// is where this is easiest to get wrong.
#[derive(Debug, Clone, Copy, Default)]
struct Settle {
    /// Consecutive all-zero reports.
    zeros: usize,
    /// Has anything moved at all yet?
    moved: bool,
}

impl Settle {
    fn observe(&mut self, report: [i16; 2]) {
        if report == [0, 0] {
            self.zeros += 1;
        } else {
            self.zeros = 0;
            self.moved = true;
        }
    }

    /// Has the stroke finished, given `left` counts still owed on each axis?
    ///
    /// The zero run is conjoined with having moved, and that conjunction is the
    /// whole point: speed starts at zero as well as ending there, so a stroke
    /// that is merely slow to get going opens with a run of zero reports. A
    /// zero run on its own would call that finished before the mouse had moved
    /// a single count.
    ///
    /// The counts still owed are the signal that actually works — the model
    /// reaches what it was asked for, measured to within a count. The rest run
    /// is only the fallback for a stroke that stops short: a hand stops where
    /// it stops, and abcurves has its own endpoint bias, so the remainder is
    /// simply what the next frame corrects.
    fn arrived(&self, left: [f64; 2]) -> bool {
        // Under a count on both axes is below the integer grid, so no further
        // report could move the pointer anyway.
        if left[0].abs() < 1. && left[1].abs() < 1. {
            return true;
        }
        self.moved && self.zeros >= REST_RUN
    }
}

/// Roll the Woodworth gain for one reach.
///
/// `move_smooth` plans a flick whole, so it draws this once per flick: a
/// ballistic movement stops short on purpose, and the minority that overshoot
/// are what put a direction reversal in the next correction.
///
/// Pumping made it easy to draw per *frame* instead, which is a different and
/// much worse thing. Re-anchoring on `rendered_xy` with a fixed gain is fine —
/// the target converges on 90% of the reach, which is the undershoot working
/// as intended. Re-rolling it is not: with `overshoot_p` at 0.15 the target
/// jumps between `0.9r` and `1.06r` every 7 ms, so the planner is handed a
/// destination that never holds still and can never commit to a reach. On
/// hardware that showed up as 5 counts taking 587 ms over 83 retargets while
/// 120 counts took 489 ms over 69 — a small correction slower than a long
/// flick, which is backwards.
///
/// So the caller rolls this when a reach begins and holds it until the reach
/// ends, which is the granularity `move_smooth` has always applied it at.
fn reach_gain(cfg: &AbConfig, random: &mut impl Rng) -> f64 {
    if random.random::<f64>() < cfg.overshoot_p {
        (1.06 + 0.04 * gauss(random).clamp(-3., 3.)).clamp(1., 1.25)
    } else {
        (cfg.gain + cfg.gain_sd * gauss(random).clamp(-3., 3.)).clamp(0.05, 1.05)
    }
}

/// Where a relative ask lands, absolutely.
///
/// `from` and the result are common units, `delta` is mouse counts, and
/// `scale` is `CountTransform::common_per_native` — which is where the y flip
/// lives, as a negative second element, and nowhere else in this module.
///
/// Pure, so the one piece of unit arithmetic between our counts and the
/// model's angles can be tested without the model files.
fn target_from(from: [f64; 2], delta: (f64, f64), gain: f64, scale: [f64; 2]) -> [f64; 2] {
    [
        from[0] + delta.0 * gain * scale[0],
        from[1] + delta.1 * gain * scale[1],
    ]
}

impl AbAim {
    /// Aim somewhere new, now.
    ///
    /// `delta` is relative, in mouse counts; abcurves targets are absolute, in
    /// common units. The pipeline tracks its own position, so successive deltas
    /// accumulate without us keeping a sub-count remainder — that is the
    /// renderer's job, which is why there is no `carry` here.
    ///
    /// Retargeting mid-stroke is the designed use of a continuous planner: the
    /// trajectory bends toward the new aim instead of stopping and starting
    /// again. Returns false when the reaction window has not elapsed.
    pub fn retarget(&mut self, delta: (f64, f64), random: &mut impl Rng) -> bool {
        self.retarget_at(delta, random, Instant::now())
    }

    /// [`Self::retarget`] with the clock injected, so the reaction window is
    /// testable without sleeping.
    pub fn retarget_at(&mut self, delta: (f64, f64), random: &mut impl Rng, now: Instant) -> bool {
        // Both components finite makes the hypotenuse finite and non-negative,
        // so `== 0.` is exactly "nothing was asked for" with no NaN left to
        // sneak through the comparison.
        if !delta.0.is_finite() || !delta.1.is_finite() || delta.0.hypot(delta.1) == 0. {
            return false;
        }
        let cfg = &self.cfg;
        // `None` means still watching: the hand has not started yet.
        let Some(_stale) = gate_reaction(
            self.last_flick,
            &mut self.ready_at,
            cfg.gap_ms,
            (cfg.react_min_ms, cfg.react_max_ms),
            now,
            random,
        ) else {
            return false;
        };

        // Everything below belongs to the *reach*, not to the frame. The aim
        // loop retargets every 7 ms, so resetting any of it per call destroys
        // it: the rest detector never counts 32 zeros in a row, and the
        // stuck-stroke clock never reaches its bound. A reach that came to
        // rest a fraction of a count short then had no way at all to end, and
        // the pump sat emitting nothing — measured as a 2255 ms dead gap in
        // the middle of a 2714 ms tracking run.
        let fresh = self.target.is_none();
        if fresh {
            self.gain = reach_gain(cfg, random);
        }
        let gain = self.gain;

        // Anchored on where the emitted counts actually put the cursor, not on
        // `current_xy()` — that is the planner's ideal position and leads the
        // wire by whatever the renderer has in flight. `rendered_xy` is
        // accumulated inside the crate as `+= to_common(report)`, which is
        // exactly where the wire is.
        let target = target_from(self.pipeline.rendered_xy(), delta, gain, self.scale);

        let ts = (self.origin.elapsed().as_micros() as i64).max(self.planned_us);
        if let Err(e) = self.pipeline.update_target(target, ts) {
            self.recover(&format!("update_target failed: {e}"));
            return false;
        }
        self.planned_us = ts;
        self.target = Some(target);
        if fresh {
            self.started = Some(now);
            self.settle = Settle::default();
        }
        self.last_flick = Some(now);
        true
    }

    /// Stop driving. The hand lets go when the reason to aim does.
    pub fn release(&mut self) {
        self.target = None;
        self.started = None;
    }

    /// Where the emitted counts have taken the cursor, in native counts.
    /// Diagnostics only: the aim loop measures the real crosshair instead.
    pub fn wire_counts(&self) -> (f64, f64) {
        let r = self.pipeline.rendered_xy();
        (r[0] / self.scale[0], r[1] / self.scale[1])
    }

    /// How long until the next sample falls due.
    ///
    /// The stream runs on a 1 ms grid but the worker loop turns over in
    /// nanoseconds, so without this it would spin a core flat for the length
    /// of every stroke. Zero when a sample is already owed, or when there is
    /// no target and the caller should park on its own terms instead.
    pub fn due_in(&self) -> Duration {
        if self.target.is_none() {
            return Duration::ZERO;
        }
        let now_us = self.origin.elapsed().as_micros() as i64;
        Duration::from_micros((self.planned_us + SAMPLE_US - now_us).max(0) as u64)
    }

    pub fn is_driving(&self) -> bool {
        self.target.is_some()
    }

    /// Advance the stream to now and put whatever fell due on the wire.
    ///
    /// Called in a tight loop by the worker. It paces itself: nothing is due
    /// until the next sample boundary, and when the worker was held up
    /// elsewhere several samples come back at once and are coalesced into one
    /// command, the same way `play_flick` folds a late report into the next.
    pub fn pump(&mut self, mouse: &MouseVirtual, keep_going: &impl Fn() -> bool) -> Pumped {
        let (state, (dx, dy)) = self.step(keep_going);
        if (dx, dy) != (0, 0) {
            if let Err(e) = mouse.emit_move(dx, dy, self.cfg.move_now) {
                tracing::error!("[AbCurves] {e}");
            }
        }
        state
    }

    /// Decide what this turn owes the wire, without touching it.
    ///
    /// Split out of [`Self::pump`] so the whole mover can be driven with no
    /// MAKCU attached: the counts and the millisecond they fall on are the
    /// interesting part, and a serial port contributes nothing to a test.
    pub fn step(&mut self, keep_going: &impl Fn() -> bool) -> (Pumped, (i64, i64)) {
        const NOTHING: (i64, i64) = (0, 0);
        let Some(target) = self.target else {
            return (Pumped::Idle, NOTHING);
        };
        // A stroke is abandoned when the reason for it goes away, exactly as
        // `play_flick` abandons one between reports.
        if !keep_going() {
            self.release();
            return (Pumped::Idle, NOTHING);
        }

        let now_us = self.origin.elapsed().as_micros() as i64;
        if now_us < self.planned_us + SAMPLE_US {
            return (Pumped::Moving, NOTHING); // not due yet
        }
        let advance = match self.pipeline.advance(now_us) {
            Ok(advance) => advance,
            Err(e) => {
                self.recover(&format!("advance failed: {e}"));
                self.release();
                return (Pumped::Idle, NOTHING);
            }
        };
        self.planned_us = now_us;

        let cap = self.cfg.max_counts.clamp(1, i32::MAX as i64);
        let (mut dx, mut dy) = (0i64, 0i64);
        for report in &advance.reports {
            self.settle.observe(*report);
            dx += report[0] as i64;
            dy += report[1] as i64;
        }
        let owed = (dx.clamp(-cap, cap), dy.clamp(-cap, cap));

        let rendered = self.pipeline.rendered_xy();
        let left = [
            (target[0] - rendered[0]) / self.scale[0].abs(),
            (target[1] - rendered[1]) / self.scale[1].abs(),
        ];
        // The stuck bound is a rail, not a tuning knob: with the aim loop
        // retargeting at frame rate a stroke that never arrives is a fault,
        // not a slow movement.
        let stuck = self
            .started
            .is_some_and(|at| at.elapsed() >= Duration::from_millis(self.cfg.max_stroke_ms));
        if self.settle.arrived(left) || stuck {
            if stuck {
                tracing::debug!("[AbCurves] stroke gave up after {} ms", self.cfg.max_stroke_ms);
            }
            self.release();
            self.last_flick = Some(Instant::now());
            return (Pumped::Arrived, owed);
        }
        (Pumped::Moving, owed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A FOV outside (0, 180) is a typo, and a zero one would make the derived
    /// radians-per-count zero — which the crate would then reject with a
    /// message about transforms rather than about configuration.
    #[test]
    fn a_nonsense_fov_is_refused_before_anything_is_loaded() {
        for fov in [0., -5., 180., 400., f64::NAN] {
            let cfg = AbConfig {
                fov_deg: fov,
                ..Default::default()
            };
            // Not `unwrap_err`: AbAim holds a pipeline that is not Debug.
            let err = match AbAim::new(cfg, 2560, 0.38, 1000.) {
                Ok(_) => panic!("FOV {fov} should have been refused"),
                Err(e) => e.to_string(),
            };
            assert!(err.contains("MOVE_AB_FOV_DEG"), "fov {fov}: {err}");
        }
    }

    /// The published derivation, pinned so a later edit to the formula has to
    /// be deliberate.
    #[test]
    fn the_derived_sensitivity_matches_the_documented_formula() {
        // The worked example in the README: 103 deg horizontal, which is the
        // 70.53 deg vertical this is usually quoted as, at 16:9.
        let (fov_deg, game_sens, mouse_dpi, screen_width) = (103.0f64, 0.38, 1000.0, 2560.0);
        let got = fov_deg.to_radians() * game_sens * mouse_dpi / (screen_width * WIN_DPI);
        assert!((got - 0.002_780).abs() < 1e-6, "{got}");
        // And it is a multiple of the model's own reference, not a fraction.
        assert!((got / 0.000_350_948_708_328_061_8 - 7.92).abs() < 0.01);
    }

    const FAR: [f64; 2] = [50., 50.];

    fn feed(settle: &mut Settle, pairs: &[(i16, i16)]) {
        for &(x, y) in pairs {
            settle.observe([x, y]);
        }
    }

    /// The bug this guards. Speed starts at zero as well as ending there, so a
    /// stroke that is slow to get going opens with a run of zero reports. A
    /// zero run on its own calls that finished before the mouse has moved one
    /// count — and the flick comes back empty.
    #[test]
    fn a_slow_start_is_not_mistaken_for_having_arrived() {
        let mut settle = Settle::default();
        feed(&mut settle, &[(0, 0); 40]);
        assert!(
            !settle.arrived(FAR),
            "forty zero reports before any motion is a slow start, not an arrival"
        );
        // And once it does move and then rests, it is an arrival.
        feed(&mut settle, &[(3, 1)]);
        feed(&mut settle, &[(0, 0); REST_RUN]);
        assert!(settle.arrived(FAR));
    }

    #[test]
    fn coming_to_rest_ends_the_stroke_even_with_counts_still_owed() {
        // abcurves has its own endpoint bias and may stop short of where we
        // aimed; the remainder is what the next frame corrects.
        let mut settle = Settle::default();
        feed(&mut settle, &[(5, 0), (4, 0), (1, 0)]);
        feed(&mut settle, &[(0, 0); REST_RUN]);
        assert!(settle.arrived(FAR), "still owed a lot, but the hand stopped");
    }

    #[test]
    fn a_pause_mid_stroke_does_not_end_it() {
        let mut settle = Settle::default();
        feed(&mut settle, &[(4, 0)]);
        feed(&mut settle, &[(0, 0); REST_RUN - 1]);
        assert!(!settle.arrived(FAR), "one short of the run is still moving");
        // Motion resets the run, so the next pause starts counting afresh.
        feed(&mut settle, &[(2, 0)]);
        feed(&mut settle, &[(0, 0); REST_RUN - 1]);
        assert!(!settle.arrived(FAR));
    }

    /// The y flip belongs to the transform and must not be applied a second
    /// time on the way in, or down-aims would walk the crosshair upward.
    #[test]
    fn a_relative_ask_becomes_an_absolute_target_with_y_flipped_once() {
        // The shape `CountTransform::new(rad, /*y_down*/ true)` hands back.
        let scale = [2., -2.];
        let got = target_from([10., 10.], (3., 4.), 1., scale);
        assert_eq!(got, [16., 2.], "x adds, y subtracts");

        // Relative to where the wire already is, not to the origin.
        let moved = target_from([100., 100.], (3., 4.), 1., scale);
        assert_eq!(moved, [106., 92.]);
        assert_eq!(
            [moved[0] - 90., moved[1] - 90.],
            got,
            "the same ask displaces by the same amount wherever it starts"
        );
    }

    fn tuned() -> AbConfig {
        AbConfig {
            gain: 0.9,
            overshoot_p: 0.15,
            ..Default::default()
        }
    }

    /// The undershoot has to survive, or corrections stop reversing direction
    /// and the movement reads as a machine servoing onto a point.
    #[test]
    fn a_reach_still_falls_short_on_average() {
        let mut r = rand::rng();
        let cfg = tuned();
        let mean: f64 = (0..4000).map(|_| reach_gain(&cfg, &mut r)).sum::<f64>() / 4000.;
        assert!(mean > 0.88 && mean < 0.98, "mean reach gain {mean}");
    }

    /// Below the integer grid nothing further could move the pointer, so this
    /// ends the stroke whatever the report stream is doing.
    #[test]
    fn being_inside_one_count_ends_the_stroke_immediately() {
        let settle = Settle::default();
        assert!(settle.arrived([0.4, -0.9]), "under a count on both axes");
        assert!(!settle.arrived([0.4, -1.8]), "still a count out on y");
        assert!(!settle.arrived([3.0, 0.1]), "still three counts out on x");
    }
}
