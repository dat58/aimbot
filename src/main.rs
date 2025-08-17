#![allow(unreachable_code)]
use aimbot::{
    aim::{AimMode, Mode},
    config::{
        Config, DISTANCE_SENSITIVITY, SCALE_ABDOMEN_Y, SCALE_CHEST_Y, SCALE_HEAD_Y, SCALE_MIN_ZONE,
        SCALE_NECK_Y,
    },
    event::start_event_listener,
    model::{Model, Point2f},
    mouse::MouseVirtual,
    stream::{UDP, handle_capture},
};
use anyhow::Result;
use crossbeam::queue::ArrayQueue;
use opencv::core::Mat;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;
use tracing_subscriber::{EnvFilter, Layer, fmt, layer::SubscriberExt, util::SubscriberInitExt};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() -> Result<()> {
    dotenv::dotenv().ok();
    tracing_subscriber::registry()
        .with(fmt::Layer::new().with_writer(std::io::stdout).with_filter(
            EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new("info"))?,
        ))
        .init();
    let config = Config::new();
    let crosshair = Point2f::new(
        config.screen_width as f32 / 2.,
        config.screen_height as f32 / 2.,
    );
    let serving_port_event_listener = config.serving_port;
    let udp_stream = UDP::new(config.url.as_str())?;
    let model = Model::new(config.clone())?;
    let frame_queue = Arc::new(ArrayQueue::<Mat>::new(1));
    let signal = Arc::new(AtomicBool::new(true));
    let aim_mode = AimMode::default();

    let capture_queue = frame_queue.clone();
    thread::spawn(move || {
        handle_capture(Box::new(udp_stream), capture_queue);
    });

    let turn_on = signal.clone();
    let aim = aim_mode.clone();
    thread::spawn(move || {
        #[cfg(feature = "debug")]
        let mut count = 0;
        #[cfg(feature = "debug")]
        const ROOT_PATH_DEBUG: &str = "assets/debug";
        #[cfg(feature = "debug")]
        {
            let path = std::path::Path::new(ROOT_PATH_DEBUG);
            if path.is_dir() {
                std::fs::remove_dir_all(path).unwrap();
            }
            std::fs::create_dir_all(path).unwrap();
        }
        let mut mouse = MouseVirtual::new(&config.makcu_port, config.makcu_baud)?;
        loop {
            if turn_on.load(Ordering::Relaxed) {
                if let Some(image) = frame_queue.pop() {
                    let mut bboxes = model.infer(&image)?;
                    bboxes.sort_by(|a, b| {
                        let dist_a = crosshair.l2_distance(&a.cxcy());
                        let dist_b = crosshair.l2_distance(&b.cxcy());
                        dist_a.partial_cmp(&dist_b).unwrap()
                    });
                    tracing::debug!("[Model] bboxes: {:?}", bboxes);
                    if bboxes.len() > 0 {
                        let bbox = bboxes[0];
                        let (destination, min_zone) = match aim.mode() {
                            Mode::Head => {
                                if bbox.class() == 1 {
                                    (bbox.cxcy(), (bbox.width() / 2.).max(bbox.height() / 2.))
                                } else {
                                    (
                                        bbox.cxcy_scale(None, Some(SCALE_HEAD_Y)),
                                        (bbox.width() / 2.).max(bbox.height() / 2. * SCALE_HEAD_Y),
                                    )
                                }
                            }
                            Mode::Neck => {
                                if bbox.class() == 1 {
                                    (
                                        Point2f::new((bbox.xmax() - bbox.xmin()) / 2., bbox.ymax()),
                                        (bbox.width() / 2.).max(bbox.height() / 2.),
                                    )
                                } else {
                                    (
                                        bbox.cxcy_scale(None, Some(SCALE_NECK_Y)),
                                        (bbox.width() / 2.).max(bbox.height() / 2. * SCALE_NECK_Y),
                                    )
                                }
                            }
                            Mode::Chest => {
                                if bbox.class() == 1 {
                                    let mut point =
                                        bbox.cxcy_scale(None, Some(SCALE_CHEST_Y / SCALE_HEAD_Y));
                                    let mut min_zone = (bbox.width() / 2.)
                                        .max(SCALE_CHEST_Y / SCALE_HEAD_Y * bbox.height() / 2.);
                                    for i in 1..bboxes.len() {
                                        let bbox = bboxes[i];
                                        if bbox.class() == 0 {
                                            point = bbox.cxcy_scale(None, Some(SCALE_CHEST_Y));
                                            min_zone = (bbox.width() / 2.)
                                                .max(SCALE_CHEST_Y * bbox.height() / 2.);
                                            break;
                                        }
                                    }
                                    (point, min_zone)
                                } else {
                                    (
                                        bbox.cxcy_scale(None, Some(SCALE_CHEST_Y)),
                                        (bbox.width() / 2.).max(SCALE_CHEST_Y * bbox.height() / 2.),
                                    )
                                }
                            }
                            Mode::Abdomen => {
                                if bbox.class() == 1 {
                                    let mut point =
                                        bbox.cxcy_scale(None, Some(SCALE_ABDOMEN_Y / SCALE_HEAD_Y));
                                    let mut min_zone = (bbox.width() / 2.)
                                        .max(SCALE_ABDOMEN_Y / SCALE_HEAD_Y * bbox.height() / 2.);
                                    for i in 1..bboxes.len() {
                                        let bbox = bboxes[i];
                                        if bbox.class() == 0 {
                                            point = bbox.cxcy_scale(None, Some(SCALE_ABDOMEN_Y));
                                            min_zone = (bbox.width() / 2.)
                                                .max(SCALE_ABDOMEN_Y * bbox.height() / 2.);
                                            break;
                                        }
                                    }
                                    (point, min_zone)
                                } else {
                                    (
                                        bbox.cxcy_scale(None, Some(SCALE_ABDOMEN_Y)),
                                        (bbox.width() / 2.)
                                            .max(SCALE_ABDOMEN_Y * bbox.height() / 2.),
                                    )
                                }
                            }
                        };
                        let min_zone = min_zone * SCALE_MIN_ZONE;
                        let dist = destination.l2_distance(&crosshair).sqrt();
                        if dist > min_zone {
                            let dx =
                                ((destination.x() - crosshair.x()) * DISTANCE_SENSITIVITY) as i32;
                            let dy =
                                ((destination.y() - crosshair.y()) * DISTANCE_SENSITIVITY) as i32;
                            mouse.move_bezier(dx, dy)?;
                        }

                        #[cfg(feature = "debug")]
                        {
                            let mut image = image;
                            tracing::info!("[Model] bboxes: {:?}", bboxes);
                            bboxes.iter().for_each(|b| {
                                opencv::imgproc::rectangle(
                                    &mut image,
                                    opencv::core::Rect::new(
                                        b.xmin() as i32,
                                        b.ymin() as i32,
                                        b.width() as i32,
                                        b.height() as i32,
                                    ),
                                    opencv::core::Scalar::new(255., 0., 255., 0.),
                                    2,
                                    -1,
                                    0,
                                )
                                .unwrap();
                            });
                            opencv::imgproc::circle(
                                &mut image,
                                opencv::core::Point::new(
                                    destination.x() as i32,
                                    destination.y() as i32,
                                ),
                                3,
                                opencv::core::Scalar::new(255., 0., 0., 0.),
                                2,
                                -1,
                                0,
                            )
                            .unwrap();
                            let filename = format!("{ROOT_PATH_DEBUG}/{count}.jpg");
                            opencv::imgcodecs::imwrite(&filename, &image, &Default::default())
                                .unwrap();
                            count += 1;
                        }
                    }
                }
            }
        }
        mouse.close();
        Ok::<(), anyhow::Error>(())
    });
    start_event_listener(signal, aim_mode, serving_port_event_listener)?;
    Ok(())
}
