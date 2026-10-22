//! The plain path: hand the whole delta to the device in one command and let
//! it walk the curve itself.
//!
//! The counterpart of [`super::smooth`], which plans and times every report
//! here instead. This one keeps no state between frames — there is no carried
//! remainder and no auto click, because nothing is being modelled.

use super::constant::*;
use super::{Aim, Mover};
use crate::mouse::MouseDevice;
use anyhow::Result;
use rand::prelude::*;

pub struct BezierMover;

impl BezierMover {
    /// The `mover = "..."` name this movement is loaded under.
    pub const NAME: &'static str = "bezier";

    /// Step count and control-point offsets for a `delta` in mouse counts.
    ///
    /// The step count rides the amplitude, which is what makes the gesture
    /// take longer the further it goes — linearly, unlike [`super::smooth`].
    pub fn plan(delta: (f64, f64), random: &mut impl Rng) -> (i32, (i32, i32)) {
        let (dx, dy) = delta;
        let pixel = (dx * dx + dy * dy).sqrt();
        let lower = (pixel * BEZIER_STEPS_FRAC.0) as i32;
        let upper = (pixel * BEZIER_STEPS_FRAC.1) as i32 + 1;
        let steps = random.random_range(lower..=upper);
        let ref_x = random.random_range(BEZIER_CTRL.0..=BEZIER_CTRL.1);
        let ref_y = random.random_range(BEZIER_CTRL.0..=BEZIER_CTRL.1);
        (steps, (ref_x, ref_y))
    }
}

impl Mover for BezierMover {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    /// `keep_going` is not read: the whole gesture leaves in one write, so
    /// there is no point between reports at which it could be abandoned.
    fn apply(
        &mut self,
        mouse: &dyn MouseDevice,
        aim: Aim,
        mut random: &mut dyn RngCore,
        _keep_going: &dyn Fn() -> bool,
    ) -> Result<()> {
        // Counts are integers. This used to hand the firmware the raw `f64`,
        // so the wire carried `km.move(12.345,-5.6,...)`; the device trait
        // speaks counts, which is what the rest of the driver has always sent.
        let (dx, dy) = (aim.delta.0 as i32, aim.delta.1 as i32);
        if !mouse.caps().curve {
            return mouse.move_to(dx, dy);
        }
        let (steps, ctrl) = Self::plan(aim.delta, &mut random);
        mouse.move_curve(dx, dy, steps, ctrl)
    }
}
