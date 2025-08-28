#![allow(unused_variables)]
#![allow(unused_imports)]
use aimbot::{
    aim::AimMode,
    config::{Config, WIN_DPI_SCALE_FACTOR},
    event::start_event_listener,
    model::{Bbox, Model, Point2f},
    mouse::MouseVirtual,
    stream::{NDI, StreamCapture, UDP, handle_capture},
};
use anyhow::{Result, anyhow};
use crossbeam::queue::ArrayQueue;
use opencv::core::{Mat, MatTraitConst};
use std::{
    io::Write,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
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
    let serving_port_event_listener = config.event_listener_port;
    let makcu_port = config.makcu_port.clone();
    let makcu_baud = config.makcu_baud;
    let source_stream: Box<dyn StreamCapture> = if config.source_stream.starts_with("ndi://") {
        let source_stream = config
            .source_stream
            .trim()
            .split(',')
            .into_iter()
            .map(|source| source.trim_start_matches("ndi://"))
            .collect::<Vec<&str>>();
        let source_stream = source_stream.join(",");
        Box::new(NDI::new(
            &source_stream,
            config.ndi_source_name.clone(),
            config.ndi_timeout,
        )?)
    } else {
        Box::new(UDP::new(config.source_stream.as_str())?)
    };
    let model = Model::new(config.clone())?;
    let frame_queue = Arc::new(ArrayQueue::<Mat>::new(1));
    let signal = Arc::new(AtomicBool::new(true));
    let aim_mode = AimMode::default();
    let running = Arc::new(AtomicBool::new(true));

    let capture_queue = frame_queue.clone();
    let keep_running = running.clone();
    thread::spawn(move || {
        handle_capture(source_stream, capture_queue, 12, Duration::from_secs(5));
        tracing::error!("Capture stream stopped");
        keep_running.store(false, Ordering::Relaxed);
    });

    let turn_on = signal.clone();
    let aim = aim_mode.clone();
    #[cfg(not(feature = "disable-mouse"))]
    let keep_running = running.clone();
    thread::spawn(move || {
        let f = move || -> Result<(), anyhow::Error> {
            #[cfg(feature = "debug")]
            let (filename_png, filename_txt) = {
                let id = uuid::Uuid::new_v4().to_string();
                let filename_png = format!("d-{id}.png");
                let filename_txt = format!("d-{id}.txt");
                (filename_png, filename_txt)
            };
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

            #[cfg(not(feature = "disable-mouse"))]
            let mut mouse = {
                let mouse =
                    MouseVirtual::new(&config.makcu_port, config.makcu_baud).map_err(|err| {
                        anyhow!(format!("Mouse cannot not initialized due to {}", err))
                    })?;
                tracing::info!("Mouse initialized");
                mouse
            };

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
                            let (destination, min_zone) = aim.aim(&bboxes).unwrap();
                            let dist = destination.l2_distance(&crosshair).sqrt();

                            #[cfg(not(feature = "disable-mouse"))]
                            if dist > min_zone * config.scale_min_zone {
                                let dx = (destination.x() - crosshair.x()) as f64
                                    * WIN_DPI_SCALE_FACTOR
                                    / config.game_sens
                                    / config.mouse_dpi;
                                let dy = (destination.y() - crosshair.y()) as f64
                                    * WIN_DPI_SCALE_FACTOR
                                    / config.game_sens
                                    / config.mouse_dpi;
                                if config.makcu_mouse_lock_while_aim {
                                    mouse
                                        .batch()
                                        .lock_mx()
                                        .lock_my()
                                        .move_bezier(dx, dy)
                                        .unlock_mx()
                                        .unlock_my()
                                        .run()?;
                                } else {
                                    mouse.move_bezier(dx, dy)?;
                                };
                            }

                            #[cfg(feature = "debug")]
                            {
                                #[cfg(not(feature = "save-bbox"))]
                                let mut image = image;

                                #[cfg(not(feature = "save-bbox"))]
                                {
                                    bboxes.class_0.iter().for_each(|b| {
                                        opencv::imgproc::rectangle(
                                            &mut image,
                                            opencv::core::Rect::new(
                                                b.xmin() as i32,
                                                b.ymin() as i32,
                                                b.width() as i32,
                                                b.height() as i32,
                                            ),
                                            opencv::core::Scalar::new(255., 255., 0., 0.),
                                            2,
                                            -1,
                                            0,
                                        )
                                        .unwrap();
                                    });
                                    bboxes.class_1.iter().for_each(|b| {
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
                                }
                                #[cfg(feature = "save-bbox")]
                                {
                                    let mut txt = std::fs::File::create(format!(
                                        "{ROOT_PATH_DEBUG}/{filename_txt}"
                                    ))?;
                                    let mut f = |bboxes: &Vec<Bbox>, class: u32| {
                                        bboxes.iter().for_each(|bbox| {
                                            let center = bbox.cxcy();
                                            let (x, y) = (
                                                center.x() / image.cols() as f32,
                                                center.y() / image.rows() as f32,
                                            );
                                            let width = bbox.width() / image.cols() as f32;
                                            let height = bbox.height() / image.rows() as f32;
                                            txt.write_all(
                                                format!("{class} {x} {y} {width} {height}\r\n")
                                                    .as_bytes(),
                                            )
                                            .unwrap();
                                        });
                                    };
                                    f(&bboxes.class_0, 0);
                                    f(&bboxes.class_1, 1);
                                    txt.flush()?;
                                }
                                let filename = format!("{ROOT_PATH_DEBUG}/{filename_png}");
                                opencv::imgcodecs::imwrite(&filename, &image, &Default::default())
                                    .unwrap();
                            }
                        }
                    }
                }
            }
        };
        f().map_err(|err| {
            tracing::error!("Model inference stop due to {}", err);
            #[cfg(not(feature = "disable-mouse"))]
            keep_running.store(false, Ordering::Relaxed);
            err
        })?;
        Ok::<_, anyhow::Error>(())
    });

    let keep_running = running.clone();
    thread::spawn(move || {
        start_event_listener(signal, aim_mode, serving_port_event_listener).map_err(|err| {
            tracing::error!("Event listener stop due to {}", err);
            keep_running.store(false, Ordering::Relaxed);
            err
        })?;
        Ok::<_, anyhow::Error>(())
    });

    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })?;
    while running.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(1000));
    }
    if config.makcu_mouse_lock_while_aim {
        let mut mouse = MouseVirtual::new(&makcu_port, makcu_baud)?;
        mouse.batch().unlock_mx().unlock_my().run()?;
    }
    tracing::warn!("Server stopped.");
    Ok(())
}
