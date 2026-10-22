use super::smooth::plan_click_for;
use super::{Aim, ClickPlan, Mover};
use crate::config::SmoothConfig;
use crate::mouse::MouseDevice;
use anyhow::Result;
use rand::RngCore;
use std::time::Instant;

pub struct DirectMover {
    cfg: SmoothConfig,
    last_click: Option<Instant>,
}

impl DirectMover {
    pub const NAME: &'static str = "direct";

    pub fn new(cfg: SmoothConfig) -> Self {
        Self {
            cfg,
            last_click: None,
        }
    }
}

impl Mover for DirectMover {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn apply(
        &mut self,
        mouse: &dyn MouseDevice,
        aim: Aim,
        _random: &mut dyn RngCore,
        _keep_going: &dyn Fn() -> bool,
    ) -> Result<()> {
        mouse.move_to(aim.delta.0 as i32, aim.delta.1 as i32)
    }

    fn after_move(&mut self, mut random: &mut dyn RngCore) -> ClickPlan {
        plan_click_for(&self.cfg, self.last_click, &mut random)
    }

    fn clicked(&mut self, at: Instant) {
        self.last_click = Some(at);
    }
}
