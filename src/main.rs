#![allow(unreachable_code)]
use aimbot::{
    aim::{AimMode, Mode},
    config::{Config, SCALE_ABDOMEN_Y, SCALE_CHEST_Y, SCALE_HEAD_Y, SCALE_NECK_Y},
    event::start_event_listener,
    model::{Model, Point2f},
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
    let model = Model::new(config)?;
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
                            let filename = format!("assets/debug/{count}.jpg");
                            opencv::imgcodecs::imwrite(&filename, &image, &Default::default())
                                .unwrap();
                            count += 1;
                        }

                        let bbox = bboxes.pop().unwrap();
                        let destination = match aim.mode() {
                            Mode::Head => {
                                if bbox.class() == 1 {
                                    bbox.cxcy()
                                } else {
                                    bbox.cxcy_scale(None, Some(SCALE_HEAD_Y))
                                }
                            }
                            Mode::Neck => {
                                if bbox.class() == 1 {
                                    Point2f::new((bbox.xmax() - bbox.xmin()) / 2., bbox.ymax())
                                } else {
                                    bbox.cxcy_scale(None, Some(SCALE_NECK_Y))
                                }
                            }
                            Mode::Chest => {
                                if bbox.class() == 1 {
                                    let mut point =
                                        bbox.cxcy_scale(None, Some(SCALE_CHEST_Y / SCALE_HEAD_Y));
                                    for bbox in bboxes {
                                        if bbox.class() == 0 {
                                            point = bbox.cxcy_scale(None, Some(SCALE_CHEST_Y));
                                            break;
                                        }
                                    }
                                    point
                                } else {
                                    bbox.cxcy_scale(None, Some(SCALE_CHEST_Y))
                                }
                            }
                            Mode::Abdomen => {
                                if bbox.class() == 1 {
                                    let mut point =
                                        bbox.cxcy_scale(None, Some(SCALE_ABDOMEN_Y / SCALE_HEAD_Y));
                                    for bbox in bboxes {
                                        if bbox.class() == 0 {
                                            point = bbox.cxcy_scale(None, Some(SCALE_ABDOMEN_Y));
                                            break;
                                        }
                                    }
                                    point
                                } else {
                                    bbox.cxcy_scale(None, Some(SCALE_ABDOMEN_Y))
                                }
                            }
                        };
                    }
                }
            }
        }
        Ok::<(), anyhow::Error>(())
    });
    start_event_listener(signal, aim_mode, serving_port_event_listener)?;
    Ok(())
}
