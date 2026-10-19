//! How a delta gets turned into movement.
//!
//! One struct per movement, each implementing [`Mover`], each selected by the
//! `mover = "..."` name in a profile. Adding one is a file and a match arm in
//! [`build`]; nothing about the device changes.

pub mod bezier;
pub mod constant;
pub mod smooth;

use crate::config::{MoverKind, Profile};
use crate::mouse::MouseDevice;
use anyhow::Result;
use bezier::BezierMover;
use rand::RngCore;
use smooth::SmoothMover;
use std::time::{Duration, Instant};

/// What the aim loop asks for: a delta in mouse counts, plus the *visual*
/// difficulty of the shot in screen pixels.
///
/// The two are not interchangeable. How long a hand takes depends on what it
/// can see, not on the mouse's DPI, so a movement that wants a duration needs
/// pixels — which counts do not carry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aim {
    pub delta: (f64, f64),
    /// Distance to the target, in screen pixels.
    pub reach_px: f32,
    /// The target's full width, in screen pixels.
    pub width_px: f32,
}

/// One request on the worker's queue: an aim, plus the profile whose mover,
/// tuning and carried state it belongs to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveRequest {
    pub profile: usize,
    pub aim: Aim,
}

/// What should happen once a movement has been played.
///
/// Both outcomes wait out the same delay; the difference is what the mouse
/// does during it, which is what decides whether the shot lands.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClickPlan {
    /// Do not click.
    None,
    /// Hold still for this long, then fire. The crosshair stays where the
    /// movement left it, so the shot carries the whole ballistic error — on a
    /// long flick that is 12-20 screen pixels off a ~20 pixel target, and it
    /// misses.
    Miss(Duration),
    /// Keep taking aim for this long, then fire. The corrections land inside
    /// the wait, so the shot is on target.
    Settle(Duration),
}

/// One way of moving the mouse.
///
/// These four are what [`crate::mouse::handle_mouse`] needs of a movement: a
/// name to log it under, the move itself, whether a click follows, and a note
/// of the click that did. A movement with nothing to say about clicking
/// implements the first two and inherits the rest.
pub trait Mover: Send {
    /// The `mover = "..."` name this movement is loaded under.
    fn name(&self) -> &'static str;

    /// Move by `aim.delta` mouse counts.
    ///
    /// Blocks for the length of the movement. `keep_going` is re-read once per
    /// report, because the alternative is not re-reading the trigger for the
    /// length of the movement: a hand abandons a flick when the reason for it
    /// goes away, so this does too.
    fn apply(
        &mut self,
        mouse: &dyn MouseDevice,
        aim: Aim,
        random: &mut dyn RngCore,
        keep_going: &dyn Fn() -> bool,
    ) -> Result<()>;

    /// What should follow the movement.
    ///
    /// The worker owns the wait rather than the mover, because settling means
    /// playing the corrections that are still arriving on a queue the mover
    /// cannot see — and because deciding here would let every correction plan
    /// a click of its own.
    fn after_move(&mut self, _random: &mut dyn RngCore) -> ClickPlan {
        ClickPlan::None
    }

    /// A click was pressed at `at`, for whatever rate limit the mover keeps.
    fn clicked(&mut self, _at: Instant) {}
}

/// Load the movement a profile names, with that profile's own tuning and its
/// own carried state.
pub fn build(profile: &Profile, baud: u32, dpi: f64) -> Box<dyn Mover> {
    match profile.mover {
        MoverKind::Smooth => Box::new(SmoothMover::new(profile.smooth, baud, dpi)),
        MoverKind::Bezier => Box::new(BezierMover),
    }
}
