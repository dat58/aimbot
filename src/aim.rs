use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};

pub const AIM_MODE_LENGTH: u8 = 4;

#[derive(Clone)]
pub struct AimMode(Arc<AtomicU8>);

#[derive(Clone, Copy)]
pub enum Mode {
    Head,
    Neck,
    Chest,
    Abdomen,
}

impl AimMode {
    pub fn mode(&self) -> Mode {
        self.0.load(Ordering::Acquire).into()
    }

    pub fn set_mode(&self, mode: Mode) {
        self.0.store(mode.into(), Ordering::Release);
    }
}

impl Default for AimMode {
    fn default() -> Self {
        Self(Arc::new(AtomicU8::new(0)))
    }
}

impl From<Mode> for AimMode {
    fn from(mode: Mode) -> Self {
        Self(Arc::new(match mode {
            Mode::Head => AtomicU8::new(0),
            Mode::Neck => AtomicU8::new(1),
            Mode::Chest => AtomicU8::new(2),
            Mode::Abdomen => AtomicU8::new(3),
        }))
    }
}

impl From<u8> for AimMode {
    fn from(mode: u8) -> Self {
        Self(Arc::new(AtomicU8::new(mode % AIM_MODE_LENGTH)))
    }
}

impl From<u8> for Mode {
    fn from(mode: u8) -> Self {
        match mode % AIM_MODE_LENGTH {
            0 => Mode::Head,
            1 => Mode::Neck,
            2 => Mode::Chest,
            _ => Mode::Abdomen,
        }
    }
}

impl Into<u8> for Mode {
    fn into(self) -> u8 {
        match self {
            Mode::Head => 0,
            Mode::Neck => 1,
            Mode::Chest => 2,
            Mode::Abdomen => 3,
        }
    }
}
