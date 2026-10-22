use super::{Button, Caps, MouseDevice};
use crate::config::{MouseConfig, RenderMode};
use anyhow::Result;
use medius::{Axis, Blanket, Device, Direction};
use std::thread::sleep;
use std::time::Duration;

pub struct Medius {
    device: Device,
    lock_move: bool,
}

impl Medius {
    pub const NAME: &'static str = "medius";

    pub fn new(cfg: &MouseConfig) -> Result<Self> {
        let device = if cfg.port.trim().is_empty() {
            Device::find()?
        } else {
            Device::open(&cfg.port)?
        };
        device.set_render(render(cfg.render), cfg.render_full)?;
        Ok(Self {
            device,
            lock_move: cfg.lock_move,
        })
    }

    fn send(&self, dx: i32, dy: i32, now: bool) -> Result<()> {
        let dx = dx.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        let dy = dy.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        if !self.lock_move {
            return Ok(self.emit(dx, dy, now)?);
        }
        self.device.lock_all(Blanket::Aim, Direction::Both)?;
        let sent = self.emit(dx, dy, now);
        self.device.unlock_all(Blanket::Aim, Direction::Both)?;
        Ok(sent?)
    }

    fn emit(&self, dx: i16, dy: i16, now: bool) -> medius::Result<()> {
        if now {
            self.device.move_rel_now(dx, dy)
        } else {
            self.device.move_rel(dx, dy)
        }
    }
}

fn render(mode: RenderMode) -> medius::RenderMode {
    match mode {
        RenderMode::Off => medius::RenderMode::Off,
        RenderMode::Stock => medius::RenderMode::Stock,
        RenderMode::Despiked => medius::RenderMode::Despiked,
        RenderMode::Unsmoothed => medius::RenderMode::Unsmoothed,
    }
}

fn usage(button: Button) -> medius::Button {
    match button {
        Button::Left => medius::Button::LEFT,
        Button::Right => medius::Button::RIGHT,
        Button::Middle => medius::Button::MIDDLE,
        Button::Side4 => medius::Button::SIDE1,
        Button::Side5 => medius::Button::SIDE2,
    }
}

impl MouseDevice for Medius {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn caps(&self) -> Caps {
        Caps {
            move_immediate: true,
            curve: false,
            buttons: false,
            lock_axis: true,
        }
    }

    fn move_to(&self, dx: i32, dy: i32) -> Result<()> {
        self.send(dx, dy, false)
    }

    fn move_immediate(&self, dx: i32, dy: i32) -> Result<()> {
        self.send(dx, dy, true)
    }

    fn press(&self, button: Button, down: bool) -> Result<()> {
        let usage = usage(button);
        Ok(if down {
            self.device.press(usage)?
        } else {
            self.device.release(usage)?
        })
    }

    fn click(&self, button: Button, hold: Duration) -> Result<()> {
        self.press(button, true)?;
        sleep(hold);
        self.press(button, false)
    }

    fn lock_mx(&self, locked: bool) -> Result<()> {
        Ok(if locked {
            self.device.lock(Axis::X, Direction::Both)?
        } else {
            self.device.unlock(Axis::X, Direction::Both)?
        })
    }

    fn lock_my(&self, locked: bool) -> Result<()> {
        Ok(if locked {
            self.device.lock(Axis::Y, Direction::Both)?
        } else {
            self.device.unlock(Axis::Y, Direction::Both)?
        })
    }
}
