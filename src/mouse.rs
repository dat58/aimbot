use anyhow::Result;
use makcu_rs as makcu;
use rand::prelude::*;
use std::time::Duration;

pub struct MouseVirtual {
    device: makcu::Device,
    random: ThreadRng,
}

impl MouseVirtual {
    pub fn new(port: &str, baud: u32) -> Result<Self> {
        let device = makcu::Device::new(port, baud, Duration::from_millis(100));
        device.connect()?;
        let random = rand::rng();
        Ok(Self { device, random })
    }

    pub fn move_shift(&self, dx: i32, dy: i32) -> Result<()> {
        Ok(self.device.move_rel(dx, dy)?)
    }

    pub fn move_bezier(&mut self, dx: i32, dy: i32) -> Result<()> {
        let pixel = (dx * dx + dy * dy).isqrt();
        let steps = if pixel < 50 {
            1
        } else if pixel < 200 {
            self.random.random_range(3..8)
        } else if pixel < 500 {
            self.random.random_range(6..12)
        } else if pixel < 1200 {
            self.random.random_range(12..26)
        } else {
            self.random.random_range(26..33)
        };
        let ref_x = self.random.random_range(4..17);
        let ref_y = self.random.random_range(4..17);
        Ok(self.device.move_bezier(dx, dy, steps, ref_x, ref_y)?)
    }

    pub fn close(&self) {
        self.device.disconnect()
    }
}
