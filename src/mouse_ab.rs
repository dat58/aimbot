//! Movement driven by [`abcurves`](https://github.com/dat58/abcurves-rs), a
//! neural model trained on real human mouse data.
//!
//! abcurves *is* the motion model, so everything about the path comes from it:
//! duration, the speed profile, curvature, tremor, how many reports a stroke is
//! worth. What stays on this side is everything about the *target* rather than
//! the path — the ballistic undershoot, the reaction window, and the whole
//! auto-click family, which it shares with `move_smooth` through
//! [`crate::mouse::AutoClick`].
//!
//! Both of the crate's planners end up in the same shape here: a finite list of
//! one report per millisecond, played on the wire against its own deadlines.
//! That is deliberately the same shape `move_smooth` already has, so the worker
//! has one way of playing a stroke rather than two.
//!
//! The static planner is a *continuation* model: it expects the hand to be
//! already moving toward the target, and lands 43-80% short from a standing
//! start. Since the hardware cannot stream the real mouse, the approach that
//! precedes the handoff is constructed here, stopping at a distance the crate's
//! own seam rules accept. [`check_prefix`] is the gate that decides whether a
//! reach is in that regime at all.

use crate::config::{AbConfig, Fallback, Planner, SmoothConfig};
use crate::mouse::{AutoClick, ClickCfg, MouseVirtual, MoveStep, gate_reaction, gauss};
use crate::mouse_input::CursorTrack;
use abcurves::{
    ContinuousPipeline, CountTransform, Intent, Movement, PipelineOptions, Policy,
    RendererProfile, StaticPipeline,
};
use anyhow::{Context, Result, bail};
use rand::prelude::*;
use std::thread::sleep;
use std::time::{Duration, Instant};

/// abcurves plans on a fixed 1 ms grid.
const SAMPLE_US: i64 = 1000;
/// The renderer profile is this many reports, and the crate rejects any other
/// length outright.
const PROFILE_REPORTS: usize = 256;
/// Below this the crate refuses to continue a movement under any policy.
const MIN_REMAINING: f64 = 8.;
/// The radius range the model was trained over starts here.
const MIN_RADIUS: f64 = 2.;
/// Consecutive all-zero reports that count as the hand having come to rest.
/// Only the fallback for a stroke that stops short; arriving is detected from
/// the counts still owed. Generous because abcurves' reports are sparse.
const REST_RUN: usize = 32;
/// Only the last 160 bins reach the static planner.
const PREFIX_BINS: usize = 160;
/// Windows API DPI reference, the same constant `mouse_counts` divides by.
const WIN_DPI: f64 = 96.;
/// Below this a sleep costs more than it saves.
const SLEEP_FLOOR: Duration = Duration::from_micros(150);

/// Whether a prefix puts a reach inside the static planner's regime.
///
/// Every variant but [`Self::Static`] routes to the fallback mover, and is
/// named so the log says which rule turned it away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixUse {
    /// Real movement with speed at the handoff.
    Static,
    /// Nothing moved. The planner would extrapolate from standstill, which is
    /// measured at 43-80% short.
    Standstill,
    /// Moved, but the last bin is empty, so the ProDMP boundary velocity is
    /// zero and the finish starts from rest.
    NoBoundaryVelocity,
    /// Under the eight counts the crate needs to continue anything.
    TooClose,
    /// Empty, or carrying a non-finite value.
    Invalid,
}

impl PrefixUse {
    pub fn is_static(self) -> bool {
        self == PrefixUse::Static
    }
}

pub fn check_prefix(prefix: &[[f32; 2]], remaining: [f64; 2]) -> PrefixUse {
    if prefix.is_empty() {
        return PrefixUse::Invalid;
    }
    let finite = remaining[0].is_finite()
        && remaining[1].is_finite()
        && prefix.iter().all(|p| p[0].is_finite() && p[1].is_finite());
    if !finite {
        return PrefixUse::Invalid;
    }
    if remaining[0].hypot(remaining[1]) < MIN_REMAINING {
        return PrefixUse::TooClose;
    }
    if prefix.iter().all(|p| p == &[0., 0.]) {
        return PrefixUse::Standstill;
    }
    if prefix[prefix.len() - 1] == [0., 0.] {
        return PrefixUse::NoBoundaryVelocity;
    }
    PrefixUse::Static
}

/// A straight constant-velocity approach that stops `bc` counts short of the
/// target, which is what the handoff is measured against.
///
/// The last bin is the planner's initial-velocity boundary condition, so every
/// bin has to carry motion; a prefix that ends at rest is exactly the
/// standstill case [`check_prefix`] turns away.
pub fn approach_prefix(delta: (f64, f64), bc: f64, bins: usize) -> Vec<[f32; 2]> {
    let distance = delta.0.hypot(delta.1);
    if !distance.is_finite() || distance <= 0. || bins == 0 {
        return Vec::new();
    }
    let share = (distance - bc).max(0.) / distance / bins as f64;
    vec![[(delta.0 * share) as f32, (delta.1 * share) as f32]; bins]
}

/// Sort the sixteen plans by duration, drop the ones past the ceiling, and draw
/// uniformly from what is left.
///
/// The heads are the model's own distribution of human finishes. Always taking
/// the shortest would replace that distribution with its own tail, which is
/// itself a signature, so the ceiling filters and the draw stays uniform.
pub fn pick_head(
    mut intents: Vec<Intent>,
    max_duration_ms: usize,
    random: &mut impl Rng,
) -> Option<Intent> {
    intents.sort_by_key(|intent| intent.duration_ms);
    let fit = intents
        .iter()
        .position(|intent| intent.duration_ms > max_duration_ms)
        .unwrap_or(intents.len());
    if fit == 0 {
        return None;
    }
    Some(intents.swap_remove(random.random_range(0..fit)))
}

/// Which planner produced a stroke, for the log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Static { head: usize },
    Continuous,
}

/// A finite stroke, one report per millisecond.
#[derive(Debug, Clone)]
pub struct Stroke {
    pub reports: Vec<[i16; 2]>,
    pub source: Source,
}

/// What the aim loop should do with a request.
#[derive(Debug)]
pub enum Plan {
    Play(Stroke),
    /// Play `approach` with the flick model, then come back through
    /// [`AbAim::finish`] with the reports it actually emitted. That prefix is
    /// real movement, which is the thing the static planner is asking for.
    Approach {
        /// What to aim the flick at, which is past the target by
        /// [`crate::config::AbConfig::lead`] so it is still moving at the cut.
        aimed: (f64, f64),
        /// The whole reach, relative to where the approach starts, x-right and
        /// y-up as the planner reads it.
        target: [f64; 2],
        radius: f64,
        /// The counts-still-owed band the seam rules accept.
        window: (f64, f64),
        /// Where inside that band to aim the cut.
        bc: f64,
    },
    /// Hand this reach to `move_smooth` instead.
    Smooth,
    /// Nothing to play: a zero delta, or the reaction window has not elapsed.
    Nothing,
}

/// How many steps of a flick to play before handing over, or `None` when no
/// cut lands inside the window.
///
/// The handoff has to land while the hand is still moving — a flick played to
/// its end stops, and a prefix that ends at rest gives the planner a zero
/// boundary velocity — and it has to land inside the seam window, which for a
/// long reach is only a few counts wide while the steps crossing it are
/// several counts each. So every cut is measured and the closest legal one
/// wins, rather than the first one past a threshold.
pub fn cut_at(
    steps: &[MoveStep],
    distance: f64,
    window: (f64, f64),
    bc: f64,
) -> Option<usize> {
    let mut moved = (0i64, 0i64);
    let mut best: Option<(f64, usize)> = None;
    for (index, step) in steps.iter().enumerate() {
        moved.0 += step.dx as i64;
        moved.1 += step.dy as i64;
        if step.dx == 0 && step.dy == 0 {
            continue; // a cut here would hand over a zero boundary velocity
        }
        let remaining = distance - (moved.0 as f64).hypot(moved.1 as f64);
        if remaining < window.0 || remaining > window.1 {
            continue;
        }
        let off = (remaining - bc).abs();
        if best.is_none_or(|(closest, _)| off < closest) {
            best = Some((off, index + 1));
        }
    }
    best.map(|(_, keep)| keep)
}

/// Fold a planned flick onto the 1 ms grid both planners read.
///
/// A bin that no step landed in is a real `[0, 0]`, not a missing sample.
/// Flips to the planner's y-up convention on the way.
pub fn bins_from_steps(steps: &[MoveStep]) -> Vec<[f32; 2]> {
    let Some(last) = steps.last() else {
        return Vec::new();
    };
    let len = last.at.as_millis() as usize + 1;
    let mut bins = vec![[0f32; 2]; len];
    for step in steps {
        let slot = (step.at.as_millis() as usize).min(len - 1);
        bins[slot][0] += step.dx as f32;
        bins[slot][1] -= step.dy as f32;
    }
    bins
}

#[derive(Debug, Default, Clone, Copy)]
struct Settle {
    zeros: usize,
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

    fn arrived(&self, left: [f64; 2]) -> bool {
        if left[0].abs() < 1. && left[1].abs() < 1. {
            return true;
        }
        self.moved && self.zeros >= REST_RUN
    }
}

/// Holds the physical mouse axes still for as long as it lives.
///
/// Never touches the buttons: a held axis must not stop a shot. Releasing on
/// `Drop` is the point — a stroke abandoned part-way, or one whose renderer
/// errors, would otherwise leave the mouse unusable.
struct AxisLock<'a> {
    mouse: Option<&'a MouseVirtual>,
}

impl<'a> AxisLock<'a> {
    fn engage(mouse: &'a MouseVirtual, wanted: bool) -> Self {
        if !wanted {
            return Self { mouse: None };
        }
        if let Err(e) = mouse.lock_mx().and_then(|()| mouse.lock_my()) {
            tracing::error!("[AbCurves] could not hold the axes: {e}");
            let _ = mouse.unlock_mx();
            let _ = mouse.unlock_my();
            return Self { mouse: None };
        }
        Self { mouse: Some(mouse) }
    }
}

impl Drop for AxisLock<'_> {
    fn drop(&mut self) {
        let Some(mouse) = self.mouse else { return };
        if let Err(e) = mouse.unlock_mx() {
            tracing::error!("[AbCurves] could not release the x axis: {e}");
        }
        if let Err(e) = mouse.unlock_my() {
            tracing::error!("[AbCurves] could not release the y axis: {e}");
        }
    }
}

pub struct AbAim {
    cfg: AbConfig,
    smooth: SmoothConfig,
    statics: Option<(StaticPipeline, RendererProfile)>,
    continuous: Option<ContinuousPipeline>,
    scale: [f64; 2],
    origin: Instant,
    planned_us: i64,
    last_flick: Option<Instant>,
    ready_at: Option<Instant>,
    last_click: Option<Instant>,
    track: Option<CursorTrack>,
}

impl AutoClick for AbAim {
    fn click_cfg(&self) -> ClickCfg {
        ClickCfg::from(&self.smooth)
    }
    fn last_click(&self) -> Option<Instant> {
        self.last_click
    }
    fn set_last_click(&mut self, at: Instant) {
        self.last_click = Some(at);
    }
}

impl AbAim {
    /// Load whichever pipelines this profile can reach, or fail saying what is
    /// missing.
    ///
    /// Both of the expensive one-off costs land here rather than mid-reach: the
    /// model files, and the renderer profile, which is 2.7 ms and nine times
    /// the plan it would otherwise precede.
    pub fn new(
        cfg: AbConfig,
        smooth: SmoothConfig,
        screen_width: u32,
        game_sens: f64,
        mouse_dpi: f64,
    ) -> Result<Self> {
        let profile_reports = read_profile(&cfg.profile)?;

        let wants_static = matches!(
            cfg.planner,
            Planner::Static | Planner::Hybrid | Planner::GetPos
        );
        let wants_continuous =
            cfg.planner == Planner::Continuous || cfg.fallback == Fallback::Continuous;

        let statics = if wants_static {
            let pipeline =
                StaticPipeline::from_pretrained(cfg.model_seed, Some(&cfg.model_dir), true)
                    .map_err(|e| anyhow::anyhow!("[AbCurves] static planner: {e}"))?;
            let profile = pipeline
                .prepare_renderer_profile(&profile_reports)
                .map_err(|e| anyhow::anyhow!("[AbCurves] renderer profile: {e}"))?;
            Some((pipeline, profile))
        } else {
            None
        };

        let mut scale = [1., -1.];
        let continuous = if wants_continuous {
            let rad = cfg.fov_deg.to_radians() * game_sens * mouse_dpi
                / (screen_width as f64 * WIN_DPI);
            let transform = CountTransform::new(rad, true)
                .map_err(|e| anyhow::anyhow!("[AbCurves] count transform: {e}"))?;
            scale = transform.common_per_native();
            let mut options = PipelineOptions::new(transform)
                .seed(rand::rng().random::<u64>())
                .renderer_seed(rand::rng().random::<u64>());
            options.model_dir = Some(cfg.model_dir.clone());
            let pipeline = ContinuousPipeline::load(&profile_reports, options)
                .map_err(|e| anyhow::anyhow!("[AbCurves] continuous planner: {e}"))?;
            tracing::info!(
                "[AbCurves] continuous ready: FOV {}°, {:.3e} rad/count",
                cfg.fov_deg,
                rad
            );
            Some(pipeline)
        } else {
            None
        };

        tracing::info!(
            "[AbCurves] planner {}, fallback {}, head ceiling {} ms, axis lock {}",
            cfg.planner,
            cfg.fallback,
            cfg.max_duration_ms,
            cfg.lock_axis
        );
        Ok(Self {
            cfg,
            smooth,
            statics,
            continuous,
            scale,
            origin: Instant::now(),
            planned_us: 0,
            last_flick: None,
            ready_at: None,
            last_click: None,
            track: None,
        })
    }

    pub fn config(&self) -> &AbConfig {
        &self.cfg
    }

    /// Hand this mover the cursor feed it needs under `planner = "getpos"`.
    ///
    /// Without one that planner has no prefix and every reach falls back, so
    /// the absence is logged rather than left to be inferred from the aim.
    pub fn track(&mut self, track: Option<CursorTrack>) {
        if track.is_none() && self.cfg.planner == Planner::GetPos {
            tracing::warn!(
                "[AbCurves] planner is \"getpos\" but the pointer cannot be read back; \
                 every reach will fall back to {}",
                self.cfg.fallback
            );
        }
        self.track = track;
    }

    pub fn wants_cursor(&self) -> bool {
        self.cfg.planner == Planner::GetPos
    }

    pub fn plan(&mut self, delta: (f64, f64), radius: f64, random: &mut impl Rng) -> Plan {
        self.plan_at(delta, radius, random, Instant::now())
    }

    /// [`Self::plan`] with the clock injected, so the reaction window is
    /// testable without sleeping.
    pub fn plan_at(
        &mut self,
        delta: (f64, f64),
        radius: f64,
        random: &mut impl Rng,
        now: Instant,
    ) -> Plan {
        if !delta.0.is_finite() || !delta.1.is_finite() || delta.0.hypot(delta.1) == 0. {
            return Plan::Nothing;
        }
        if gate_reaction(
            self.last_flick,
            &mut self.ready_at,
            self.smooth.gap_ms,
            (self.smooth.react_min_ms, self.smooth.react_max_ms),
            now,
            random,
        )
        .is_none()
        {
            return Plan::Nothing;
        }

        let gain = reach_gain(&self.smooth, random);
        let reach = (delta.0 * gain, delta.1 * gain);
        let radius = radius.max(MIN_RADIUS);

        // Everything the static planner sees is x-right and y-up; the aim loop
        // is y-down, so the flip happens here and is undone on every report.
        let target = [reach.0, -reach.1];
        let outcome = match self.cfg.planner {
            // The hand did the approach; read it back and finish it.
            Planner::GetPos => {
                let Some(track) = self.track.as_mut() else {
                    return self.fall_back(reach);
                };
                let prefix = track.recent(PREFIX_BINS);
                Ok(self.finish(&prefix, target, radius, random))
            }
            Planner::Hybrid => {
                let Some((window, bc)) = self.handoff_window(target, radius) else {
                    tracing::debug!("[AbCurves] no handoff window for this geometry");
                    return self.fall_back(reach);
                };
                return Plan::Approach {
                    aimed: (reach.0 * self.cfg.lead, reach.1 * self.cfg.lead),
                    target,
                    radius,
                    window,
                    bc,
                };
            }
            Planner::Static => Ok(self.plan_static(target, radius, random)),
            Planner::Continuous => self.plan_continuous(reach),
        };
        let stroke = match outcome {
            Ok(Some(stroke)) => stroke,
            Ok(None) => return self.fall_back(reach),
            Err(e) => {
                tracing::debug!("[AbCurves] {e}");
                return self.fall_back(reach);
            }
        };
        self.last_flick = Some(now + Duration::from_millis(stroke.reports.len() as u64));
        Plan::Play(stroke)
    }

    fn fall_back(&mut self, reach: (f64, f64)) -> Plan {
        match self.cfg.fallback {
            Fallback::Smooth => Plan::Smooth,
            Fallback::Continuous => match self.plan_continuous(reach) {
                Ok(Some(stroke)) => {
                    self.last_flick =
                        Some(Instant::now() + Duration::from_millis(stroke.reports.len() as u64));
                    Plan::Play(stroke)
                }
                Ok(None) => Plan::Smooth,
                Err(e) => {
                    tracing::debug!("[AbCurves] continuous fallback: {e}");
                    Plan::Smooth
                }
            },
        }
    }

    /// Where the handoff lands, in counts still owed at B.
    ///
    /// `move_ab_approach` says where it would like to sit, but the seam rules
    /// own the answer: outside the window the crate refuses the continuation
    /// outright, so the ask is clamped into it.
    /// The counts-still-owed band the seam rules accept, and where inside it
    /// this profile aims to hand over.
    pub fn handoff_window(&self, target: [f64; 2], radius: f64) -> Option<((f64, f64), f64)> {
        let window =
            Movement::eligible_remaining([0., 0.], target, radius.max(MIN_RADIUS), Policy::Auto)?;
        let (lower, upper) = (*window.start(), *window.end());
        let bc = upper - (upper - lower) * self.cfg.handoff.clamp(0., 1.);
        Some(((lower, upper), bc))
    }

    /// The static planner over an approach that is built here and never sent,
    /// so only the finish reaches the wire.
    fn plan_static(
        &mut self,
        target: [f64; 2],
        radius: f64,
        random: &mut impl Rng,
    ) -> Option<Stroke> {
        let (_, bc) = self.handoff_window(target, radius)?;
        let prefix = approach_prefix((target[0], target[1]), bc, self.cfg.approach_bins);
        self.finish(&prefix, target, radius, random)
    }

    /// Plan and render the continuation that follows `prefix`.
    ///
    /// `None` is a routing decision, not a failure: the reach is outside the
    /// regime this planner was trained for and belongs to another mover.
    pub fn finish(
        &mut self,
        prefix: &[[f32; 2]],
        target: [f64; 2],
        radius: f64,
        random: &mut impl Rng,
    ) -> Option<Stroke> {
        let radius = radius.max(MIN_RADIUS);
        let movement = Movement::from_counts([0., 0.], target, radius, prefix);
        let verdict = check_prefix(movement.prefix(), movement.remaining());
        if !verdict.is_static() {
            tracing::debug!("[AbCurves] prefix is {verdict:?}, not the static regime");
            return None;
        }
        if let Err(e) = movement.handoff() {
            tracing::debug!("[AbCurves] handoff refused: {e}");
            return None;
        }
        let Some((pipeline, profile)) = self.statics.as_mut() else {
            tracing::debug!("[AbCurves] no static pipeline is loaded for this profile");
            return None;
        };
        let intents = match pipeline.plan_movement_all(&movement) {
            Ok(intents) => intents,
            Err(e) => {
                tracing::debug!("[AbCurves] plan: {e}");
                return None;
            }
        };
        let intent = pick_head(intents, self.cfg.max_duration_ms, random).or_else(|| {
            tracing::debug!(
                "[AbCurves] every head runs past the {} ms ceiling",
                self.cfg.max_duration_ms
            );
            None
        })?;
        let head = intent.head;
        let mut stream = match pipeline.begin(profile, intent, random.random::<u64>()) {
            Ok(stream) => stream,
            Err(e) => {
                tracing::debug!("[AbCurves] render: {e}");
                return None;
            }
        };
        let model = pipeline.model();
        let cap = self.smooth.max_counts.clamp(1, i32::MAX as i64);
        let mut reports = Vec::with_capacity(stream.duration_ms());
        while !stream.complete() {
            match stream.step(model) {
                // Back to the aim loop's y-down convention.
                Ok(report) => {
                    reports.push([clamp_count(report[0], cap), clamp_count(-report[1], cap)])
                }
                Err(e) => {
                    tracing::debug!("[AbCurves] step: {e}");
                    return None;
                }
            }
        }
        Some(Stroke {
            reports,
            source: Source::Static { head },
        })
    }

    fn plan_continuous(&mut self, reach: (f64, f64)) -> Result<Option<Stroke>> {
        let cap = self.smooth.max_counts.clamp(1, i32::MAX as i64);
        let rail = Duration::from_millis(self.cfg.max_stroke_ms);
        let scale = self.scale;
        let Some(pipeline) = self.continuous.as_mut() else {
            return Ok(None);
        };

        let from = pipeline.rendered_xy();
        let target = [
            from[0] + reach.0 * scale[0],
            from[1] + reach.1 * scale[1],
        ];
        let ts = (self.origin.elapsed().as_micros() as i64).max(self.planned_us);
        pipeline
            .update_target(target, ts)
            .map_err(|e| anyhow::anyhow!("update_target: {e}"))?;
        self.planned_us = ts;

        let mut reports = Vec::new();
        let mut settle = Settle::default();
        let budget = rail.as_millis() as usize;
        while reports.len() < budget {
            self.planned_us += SAMPLE_US;
            let advance = pipeline
                .advance(self.planned_us)
                .map_err(|e| anyhow::anyhow!("advance: {e}"))?;
            for report in &advance.reports {
                settle.observe(*report);
                reports.push([clamp_count(report[0], cap), clamp_count(report[1], cap)]);
            }
            let rendered = pipeline.rendered_xy();
            let left = [
                (target[0] - rendered[0]) / scale[0].abs(),
                (target[1] - rendered[1]) / scale[1].abs(),
            ];
            if settle.arrived(left) {
                break;
            }
        }
        if reports.is_empty() {
            return Ok(None);
        }
        Ok(Some(Stroke {
            reports,
            source: Source::Continuous,
        }))
    }

    /// Put a stroke on the wire, one report per millisecond against its own
    /// deadlines.
    ///
    /// Zero-motion reports are elided — a real mouse does send them but the
    /// firmware gains nothing, and the deadline is still honoured, so the shape
    /// of the profile survives.
    pub fn play(
        &self,
        mouse: &MouseVirtual,
        stroke: &Stroke,
        keep_going: impl Fn() -> bool,
    ) -> Result<()> {
        let _lock = AxisLock::engage(mouse, self.cfg.lock_axis);
        let start = Instant::now();
        let last = stroke.reports.len().saturating_sub(1);
        let (mut dx, mut dy) = (0i64, 0i64);
        for (index, report) in stroke.reports.iter().enumerate() {
            if !keep_going() {
                break;
            }
            dx += report[0] as i64;
            dy += report[1] as i64;
            let due = start + Duration::from_millis(index as u64);
            match due.checked_duration_since(Instant::now()) {
                Some(wait) if wait > SLEEP_FLOOR => sleep(wait),
                Some(_) => {}
                None if index < last => continue,
                None => {}
            }
            if (dx, dy) != (0, 0) {
                mouse.emit_move(dx, dy, self.smooth.move_now)?;
                dx = 0;
                dy = 0;
            }
        }
        Ok(())
    }
}

/// The Woodworth gain for one reach: a ballistic movement stops short on
/// purpose, and the minority that overshoot are what put a direction reversal
/// in the next correction.
fn reach_gain(cfg: &SmoothConfig, random: &mut impl Rng) -> f64 {
    if random.random::<f64>() < cfg.overshoot_p {
        (1.06 + 0.04 * gauss(random).clamp(-3., 3.)).clamp(1., 1.25)
    } else {
        (cfg.gain + cfg.gain_sd * gauss(random).clamp(-3., 3.)).clamp(0.05, 1.05)
    }
}

fn clamp_count(value: i16, cap: i64) -> i16 {
    (value as i64).clamp(-cap, cap) as i16
}

fn read_profile(path: &std::path::Path) -> Result<Vec<[i16; 2]>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("[AbCurves] cannot read the renderer profile {}", path.display()))?;
    if bytes.len() != PROFILE_REPORTS * 4 {
        bail!(
            "[AbCurves] {} is {} bytes; a renderer profile is exactly {} ({PROFILE_REPORTS} reports)",
            path.display(),
            bytes.len(),
            PROFILE_REPORTS * 4
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

#[cfg(test)]
mod tests {
    use super::*;

    fn moving(bins: usize) -> Vec<[f32; 2]> {
        vec![[2., -1.]; bins]
    }

    const FAR: [f64; 2] = [120., 0.];

    #[test]
    fn a_still_prefix_is_not_the_static_regime() {
        assert_eq!(check_prefix(&vec![[0., 0.]; 160], FAR), PrefixUse::Standstill);
    }

    /// The last bin is the planner's initial-velocity boundary condition, so a
    /// prefix that ends at rest starts its finish from standstill.
    #[test]
    fn a_prefix_that_ends_at_rest_is_turned_away() {
        let mut prefix = moving(160);
        prefix[159] = [0., 0.];
        assert_eq!(check_prefix(&prefix, FAR), PrefixUse::NoBoundaryVelocity);
    }

    #[test]
    fn a_reach_under_eight_counts_cannot_be_continued() {
        for remaining in [[7.9, 0.], [0., 7.9], [5., 5.]] {
            assert_eq!(
                check_prefix(&moving(160), remaining),
                PrefixUse::TooClose,
                "{remaining:?}"
            );
        }
        assert_eq!(check_prefix(&moving(160), [8.1, 0.]), PrefixUse::Static);
    }

    #[test]
    fn an_empty_or_non_finite_prefix_is_invalid() {
        assert_eq!(check_prefix(&[], FAR), PrefixUse::Invalid);
        assert_eq!(check_prefix(&[[f32::NAN, 0.]], FAR), PrefixUse::Invalid);
        assert_eq!(check_prefix(&[[1., f32::INFINITY]], FAR), PrefixUse::Invalid);
        assert_eq!(
            check_prefix(&moving(10), [f64::NAN, 0.]),
            PrefixUse::Invalid
        );
    }

    #[test]
    fn the_synthetic_approach_stops_where_it_was_asked_to() {
        let delta = (300., -120.);
        let prefix = approach_prefix(delta, 40., 100);
        assert_eq!(prefix.len(), 100);
        let total: (f64, f64) = prefix.iter().fold((0., 0.), |sum, step| {
            (sum.0 + step[0] as f64, sum.1 + step[1] as f64)
        });
        let distance = delta.0.hypot(delta.1);
        assert!(
            (total.0.hypot(total.1) - (distance - 40.)).abs() < 0.5,
            "travelled {:.2}, wanted {:.2}",
            total.0.hypot(total.1),
            distance - 40.
        );
        // Same direction as the ask, and still moving at the handoff.
        assert!(total.0 > 0. && total.1 < 0.);
        assert_ne!(prefix[99], [0., 0.]);
    }

    /// If the prefix this module builds for itself were ever turned away, the
    /// static planner could never run at all.
    #[test]
    fn the_synthetic_approach_is_always_in_the_static_regime() {
        for delta in [(300.0f64, -120.0f64), (-40., 40.), (9., 0.), (0., -250.)] {
            let distance = delta.0.hypot(delta.1);
            let bc = (distance * 0.10).max(MIN_REMAINING + 0.5);
            let prefix = approach_prefix(delta, bc, 100);
            let unit = (delta.0 / distance, delta.1 / distance);
            let remaining = [unit.0 * bc, unit.1 * bc];
            assert_eq!(
                check_prefix(&prefix, remaining),
                PrefixUse::Static,
                "{delta:?}"
            );
        }
    }

    fn steps(counts: &[(i32, i32)]) -> Vec<MoveStep> {
        counts
            .iter()
            .enumerate()
            .map(|(i, (dx, dy))| MoveStep {
                dx: *dx,
                dy: *dy,
                at: Duration::from_millis(i as u64),
            })
            .collect()
    }

    /// The cut has to land inside the seam band, and the band is only a few
    /// counts wide while the steps crossing it are several counts each — so
    /// the first step past a threshold is the wrong answer.
    #[test]
    fn the_cut_lands_inside_the_window_not_past_it() {
        // Travels 10 counts a step; remaining goes 90, 80, ... 10, 0.
        let flick = steps(&[(10, 0); 10]);
        let keep = cut_at(&flick, 100., (18., 26.), 22.).expect("20 is inside the band");
        let moved: i32 = flick[..keep].iter().map(|s| s.dx).sum();
        assert_eq!(100 - moved, 20);
    }

    #[test]
    fn a_flick_that_steps_over_the_window_has_no_legal_cut() {
        // Remaining jumps 100 -> 50 -> 0, straight past a band at 18..26.
        let flick = steps(&[(50, 0), (50, 0)]);
        assert_eq!(cut_at(&flick, 100., (18., 26.), 22.), None);
    }

    /// A cut on an empty report would hand the planner a zero boundary
    /// velocity, which is exactly what the static regime rejects.
    #[test]
    fn the_cut_never_lands_on_a_report_that_moved_nothing() {
        let flick = steps(&[(10, 0), (10, 0), (10, 0), (10, 0), (0, 0), (0, 0)]);
        let keep = cut_at(&flick, 100., (55., 65.), 60.).unwrap();
        assert_ne!((flick[keep - 1].dx, flick[keep - 1].dy), (0, 0));
    }

    #[test]
    fn a_widened_window_picks_the_closest_approach() {
        let flick = steps(&[(10, 0); 10]);
        let keep = cut_at(&flick, 100., (0., f64::MAX), 0.).unwrap();
        assert_eq!(keep, flick.len());
    }

    /// The flick model is y-down and the planner y-up, so the prefix flips
    /// exactly once on the way in.
    #[test]
    fn folding_a_flick_into_bins_flips_y_once() {
        let flick = steps(&[(3, 4), (1, -2)]);
        let bins = bins_from_steps(&flick);
        assert_eq!(bins, vec![[3., -4.], [1., 2.]]);
    }

    #[test]
    fn a_millisecond_no_step_landed_in_is_a_real_zero() {
        let flick = vec![
            MoveStep { dx: 2, dy: 0, at: Duration::from_millis(0) },
            MoveStep { dx: 3, dy: 0, at: Duration::from_millis(3) },
        ];
        assert_eq!(
            bins_from_steps(&flick),
            vec![[2., 0.], [0., 0.], [0., 0.], [3., 0.]]
        );
    }

    fn intent(head: usize, duration_ms: usize) -> Intent {
        Intent {
            smooth_dxdy: vec![[1., 0.]; duration_ms.max(1)],
            duration_ms,
            head,
        }
    }

    fn sixteen() -> Vec<Intent> {
        (0..16).map(|k| intent(k, 40 + k * 30)).collect()
    }

    #[test]
    fn a_head_past_the_ceiling_is_dropped() {
        let mut r = rand::rng();
        for _ in 0..200 {
            let picked = pick_head(sixteen(), 160, &mut r).expect("some head fits 160 ms");
            assert!(picked.duration_ms <= 160, "{}", picked.duration_ms);
        }
    }

    #[test]
    fn a_ceiling_below_every_head_leaves_nothing() {
        let mut r = rand::rng();
        assert!(pick_head(sixteen(), 10, &mut r).is_none());
        assert!(pick_head(Vec::new(), 1000, &mut r).is_none());
    }

    /// The sixteen heads are the model's distribution of human finishes, so the
    /// draw has to reach all of them. Always taking the shortest would replace
    /// that distribution with its own tail.
    #[test]
    fn every_head_inside_the_ceiling_gets_drawn() {
        let mut r = rand::rng();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..2000 {
            seen.insert(pick_head(sixteen(), 250, &mut r).unwrap().head);
        }
        let fit: Vec<usize> = sixteen()
            .iter()
            .filter(|i| i.duration_ms <= 250)
            .map(|i| i.head)
            .collect();
        assert!(fit.len() > 1, "the fixture must leave a real choice");
        for head in fit {
            assert!(seen.contains(&head), "head {head} was never drawn");
        }
    }

    #[test]
    fn a_report_is_clamped_to_the_configured_ceiling() {
        assert_eq!(clamp_count(5000, 127), 127);
        assert_eq!(clamp_count(-5000, 127), -127);
        assert_eq!(clamp_count(42, i16::MAX as i64), 42);
    }

    #[test]
    fn coming_to_rest_ends_a_stroke_that_stopped_short() {
        let mut settle = Settle::default();
        assert!(!settle.arrived([50., 0.]), "nothing has moved yet");
        settle.observe([4, 0]);
        for _ in 0..REST_RUN {
            settle.observe([0, 0]);
        }
        assert!(settle.arrived([50., 0.]));
    }

    /// Speed starts at zero as well as ending there, so a slow opening is a run
    /// of empty reports and must not read as having arrived.
    #[test]
    fn a_slow_start_is_not_mistaken_for_having_arrived() {
        let mut settle = Settle::default();
        for _ in 0..REST_RUN * 2 {
            settle.observe([0, 0]);
        }
        assert!(!settle.arrived([50., 0.]));
    }

    #[test]
    fn being_inside_one_count_ends_the_stroke_whatever_the_reports_say() {
        let settle = Settle::default();
        assert!(settle.arrived([0.4, -0.9]));
        assert!(!settle.arrived([0.4, -1.8]));
    }
}
