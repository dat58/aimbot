use crate::{
    config::{SCALE_ABDOMEN_Y, SCALE_CHEST_Y, SCALE_HEAD_Y, SCALE_NECK_Y},
    model::{Bboxes, Point2f},
};
use rand::prelude::*;
use std::{
    fmt::Display,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
};

pub const AIM_MODE_LENGTH: u8 = 5;

#[derive(Clone, Debug)]
pub struct AimMode(Arc<AtomicU8>);

#[derive(Clone, Copy)]
pub enum Mode {
    Head,
    Neck,
    Chest,
    Abdomen,
    Horizon,
}

impl AimMode {
    pub fn mode(&self) -> Mode {
        self.0.load(Ordering::Acquire).into()
    }

    pub fn set_mode(&self, mode: Mode) {
        self.0.store(mode.into(), Ordering::Release);
    }

    pub fn aim(
        &self,
        bboxes: &Bboxes,
        crosshair: &Point2f,
        random: &mut ThreadRng,
    ) -> Option<(Point2f, f32)> {
        match self.mode() {
            Mode::Head => self.aim_head(bboxes),
            Mode::Neck => self.aim_neck(bboxes),
            Mode::Chest => self.aim_chest(bboxes),
            Mode::Abdomen => self.aim_abdomen(bboxes),
            Mode::Horizon => self.aim_horizon(bboxes, crosshair, random),
        }
    }

    pub fn aim_head(&self, bboxes: &Bboxes) -> Option<(Point2f, f32)> {
        match bboxes.class_1.first() {
            Some(bbox) => Some((
                Point2f::new(
                    (bbox.xmax() + bbox.xmin()) / 2.,
                    bbox.ymax() - bbox.height() / 3.,
                ),
                (bbox.width() / 2.).max(bbox.height() / 2.),
            )),
            _ => match bboxes.class_0.first() {
                Some(bbox) => Some((
                    bbox.cxcy_scale(None, Some(SCALE_HEAD_Y)),
                    (bbox.width() / 2.).max(bbox.height() / 2. * SCALE_HEAD_Y),
                )),
                _ => None,
            },
        }
    }

    pub fn aim_neck(&self, bboxes: &Bboxes) -> Option<(Point2f, f32)> {
        match bboxes.class_1.first() {
            Some(bbox) => Some((
                Point2f::new((bbox.xmax() + bbox.xmin()) / 2., bbox.ymax()),
                (bbox.width() / 2.).max(bbox.height() / 2.),
            )),
            _ => match bboxes.class_0.first() {
                Some(bbox) => Some((
                    bbox.cxcy_scale(None, Some(SCALE_NECK_Y)),
                    (bbox.width() / 2.).max(bbox.height() / 2. * SCALE_NECK_Y),
                )),
                _ => None,
            },
        }
    }

    pub fn aim_chest(&self, bboxes: &Bboxes) -> Option<(Point2f, f32)> {
        self.aim_low(bboxes, SCALE_CHEST_Y)
    }

    pub fn aim_abdomen(&self, bboxes: &Bboxes) -> Option<(Point2f, f32)> {
        self.aim_low(bboxes, SCALE_ABDOMEN_Y)
    }

    #[inline(always)]
    fn aim_low(&self, bboxes: &Bboxes, scale: f32) -> Option<(Point2f, f32)> {
        match bboxes.class_0.first() {
            Some(bbox) => Some((
                bbox.cxcy_scale(None, Some(scale)),
                (bbox.width() / 2.).max(scale * bbox.height() / 2.),
            )),
            _ => match bboxes.class_1.first() {
                Some(bbox) => Some((
                    bbox.cxcy_scale(None, Some(scale / SCALE_HEAD_Y)),
                    (bbox.width() / 2.).max(scale / SCALE_HEAD_Y * bbox.height() / 2.),
                )),
                _ => None,
            },
        }
    }

    #[inline(always)]
    pub fn aim_horizon(
        &self,
        bboxes: &Bboxes,
        crosshair: &Point2f,
        random: &mut ThreadRng,
    ) -> Option<(Point2f, f32)> {
        match bboxes.class_1.first() {
            Some(bbox) => {
                if crosshair.y() < bbox.ymin() {
                    let delta_y = random.random_range(0..11.min(bbox.height() as i32)) as f32;
                    let y_rand = crosshair.y() + delta_y;
                    Some((
                        Point2f::new((bbox.xmin() + bbox.xmax()) / 2., y_rand),
                        bbox.width() / 2.,
                    ))
                } else if crosshair.y() > bbox.ymax() {
                    let delta_y = random.random_range(0..11.min(bbox.height() as i32)) as f32;
                    let y_rand = crosshair.y() - delta_y;
                    Some((
                        Point2f::new((bbox.xmin() + bbox.xmax()) / 2., y_rand),
                        bbox.width() / 2.,
                    ))
                } else {
                    Some((
                        Point2f::new(
                            (bbox.xmax() + bbox.xmin()) / 2.,
                            bbox.ymax() - bbox.height() / 3.,
                        ),
                        (bbox.width() / 2.).max(bbox.height() / 2.),
                    ))
                }
            }
            _ => match bboxes.class_0.first() {
                Some(bbox) => {
                    if crosshair.y() < bbox.ymin() {
                        let delta_y = random.random_range(0..21.min(bbox.height() as i32)) as f32;
                        let y_rand = crosshair.y() + delta_y;
                        Some((
                            Point2f::new((bbox.xmin() + bbox.xmax()) / 2., y_rand),
                            bbox.width() / 2.,
                        ))
                    } else if crosshair.y() > bbox.ymax() {
                        let delta_y = random.random_range(0..21.min(bbox.height() as i32)) as f32;
                        let y_rand = crosshair.y() - delta_y;
                        Some((
                            Point2f::new((bbox.xmin() + bbox.xmax()) / 2., y_rand),
                            bbox.width() / 2.,
                        ))
                    } else {
                        let delta_y = random.random_range(
                            -((crosshair.y() - bbox.ymin()) as i32).min(10)
                                ..((bbox.ymax() - crosshair.y()) as i32).min(12),
                        ) as f32;
                        let y_rand = crosshair.y() + delta_y;
                        Some((
                            Point2f::new((bbox.xmin() + bbox.xmax()) / 2., y_rand),
                            bbox.width() / 2.,
                        ))
                    }
                }
                _ => None,
            },
        }
    }
}

impl Default for AimMode {
    fn default() -> Self {
        Self(Arc::new(AtomicU8::new(2)))
    }
}

impl From<Mode> for AimMode {
    fn from(mode: Mode) -> Self {
        Self(Arc::new(match mode {
            Mode::Head => AtomicU8::new(0),
            Mode::Neck => AtomicU8::new(1),
            Mode::Chest => AtomicU8::new(2),
            Mode::Abdomen => AtomicU8::new(3),
            Mode::Horizon => AtomicU8::new(4),
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
            3 => Mode::Abdomen,
            _ => Mode::Horizon,
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
            Mode::Horizon => 4,
        }
    }
}

impl Display for AimMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.mode() {
            Mode::Head => write!(f, "Head"),
            Mode::Neck => write!(f, "Neck"),
            Mode::Chest => write!(f, "Chest"),
            Mode::Abdomen => write!(f, "Abdomen"),
            Mode::Horizon => write!(f, "Horizon"),
        }
    }
}
