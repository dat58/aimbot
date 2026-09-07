#![allow(unused_variables)]
#![allow(unused_imports)]
use aimbot::{
    aim::{AimMode, Mode},
    config::{Config, mouse_counts},
    esp_button::EspButton,
    event::start_event_listener,
    model::{Bbox, Model, Point2f},
    mouse::MouseVirtual,
    stream::{Elgato, NDI, StreamCapture, UDP, handle_capture},
};
use anyhow::{Result, anyhow};
use crossbeam::queue::ArrayQueue;
use mimalloc::MiMalloc;
use opencv::core::{Mat, MatTraitConst};
use rand::prelude::*;
use std::{
    io::Write,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use tracing_subscriber::{EnvFilter, Layer, fmt, layer::SubscriberExt, util::SubscriberInitExt};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() -> Result<()> {
    dotenv::dotenv().ok();
    tracing_subscriber::registry()
        .with(fmt::Layer::new().with_writer(std::io::stdout).with_filter(
            EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new("info"))?,
        ))
        .init();
    ort::init().commit()?;
    let config = Config::new();
    // Detections and `REGION_*` are in frame pixels, so the crosshair has to be
    // the centre of the frame, not of the screen. They differ whenever the
    // capture does not match the screen resolution.
    let crosshair = Point2f::new(
        config.frame_width as f32 / 2.,
        config.frame_height as f32 / 2.,
    );
    let serving_port_event_listener = config.event_listener_port;
    let makcu_port = config.makcu_port.clone();
    let makcu_baud = config.makcu_baud;
    let esp_port = config.esp_port.clone();
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
    } else if let Some(device) = config
        .source_stream
        .trim()
        .strip_prefix("elgato://")
        .or_else(|| config.source_stream.trim().strip_prefix("v4l2://"))
    {
        // `elgato://` on its own falls back to CAPTURE_DEVICE.
        let device = (!device.is_empty()).then_some(device);
        Box::new(Elgato::new(&config, device)?)
    } else {
        Box::new(UDP::new(config.source_stream.as_str())?)
    };
    let model = Model::new(config.clone())?;
    let frame_queue = Arc::new(ArrayQueue::<Mat>::new(1));
    let use_trigger = Arc::new(AtomicBool::new(true));
    let use_auto_aim = Arc::new(AtomicBool::new(true));
    let aim_mode = match config.default_aim_mode {
        Some(mode) => AimMode::from(Mode::from(mode)),
        _ => AimMode::default(),
    };
    tracing::info!("[main] Start app with aim mode {aim_mode}");
    let esp_button1 = Arc::new(AtomicBool::new(false));
    let esp_button2 = Arc::new(AtomicBool::new(false));
    let running = Arc::new(AtomicBool::new(true));

    let capture_queue = frame_queue.clone();
    let keep_running = running.clone();
    thread::spawn(move || {
        handle_capture(
            source_stream,
            capture_queue,
            10000,
            Duration::from_millis(2),
        );
        tracing::error!("Capture stream stopped");
        keep_running.store(false, Ordering::Relaxed);
    });

    if let Some(esp_port) = esp_port {
        let mut button = EspButton::new(&esp_port, esp_button1.clone(), esp_button2.clone())?;
        let keep_running = running.clone();
        thread::spawn(move || {
            tracing::info!("Start listening esp button");
            button.listen();
            tracing::error!("Esp button stopped");
            keep_running.store(false, Ordering::Relaxed);
        });
    }

    let trigger = use_trigger.clone();
    let auto_aim = use_auto_aim.clone();
    let aim = aim_mode.clone();
    #[cfg(not(feature = "disable-mouse"))]
    let keep_running = running.clone();
    #[cfg(not(feature = "disable-mouse"))]
    let keep_running_clone = keep_running.clone();
    thread::spawn(move || {
        let f = move || -> Result<(), anyhow::Error> {
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
            let mouse = {
                let mouse =
                    MouseVirtual::new(&config.makcu_port, config.makcu_baud).map_err(|err| {
                        anyhow!(format!("Mouse cannot not initialized due to {}", err))
                    })?;
                tracing::info!("Mouse initialized");

                let mouse = Arc::new(mouse);

                if config.makcu_listen {
                    let m = mouse.clone();
                    let running = keep_running_clone.clone();
                    thread::spawn(move || {
                        tracing::info!("Start listening mouse button");
                        m.listen_button_presses();
                        running.store(false, Ordering::Relaxed);
                    });

                    let m = mouse.clone();
                    let trigger_clone = trigger.clone();
                    let auto_aim_clone = auto_aim.clone();
                    let running = keep_running_clone.clone();
                    thread::spawn(move || {
                        tracing::info!("Start handling the switch trigger/auto_aim button");
                        let f = Box::new(move || {
                            let mut last_value = trigger_clone.load(Ordering::Acquire);
                            last_value = !last_value;
                            trigger_clone.store(last_value, Ordering::Release);
                            // auto_aim always set to `true` every time trigger changed
                            auto_aim_clone.store(true, Ordering::Release);
                            tracing::info!("trigger modified to {}", last_value);
                        });
                        m.handle_right_holding(
                            // hold right mouse to switch
                            // between trigger & auto aim bot
                            Duration::from_millis(500),
                            Duration::from_millis(50),
                            f,
                        );
                        running.store(false, Ordering::Relaxed);
                    });

                    let m = mouse.clone();
                    let trigger_clone = trigger.clone();
                    let auto_aim_clone = auto_aim.clone();
                    let running = keep_running_clone.clone();
                    thread::spawn(move || {
                        tracing::info!("Start handling the switch auto aim button side4");
                        let f = Box::new(move || {
                            if !trigger_clone.load(Ordering::Acquire) {
                                auto_aim_clone.store(true, Ordering::Release);
                                tracing::info!("Auto aim modified to true");
                            }
                        });
                        m.handle_side4_holding(
                            // hold side4 mouse button ~20 millis second `turn on` auto aim
                            Duration::from_millis(0),
                            Duration::from_millis(50),
                            f,
                        );
                        running.store(false, Ordering::Relaxed);
                    });

                    let m = mouse.clone();
                    let trigger_clone = trigger.clone();
                    let auto_aim_clone = auto_aim.clone();
                    let running = keep_running_clone.clone();
                    thread::spawn(move || {
                        tracing::info!("Start handling the switch auto aim button side5");
                        let f = Box::new(move || {
                            if !trigger_clone.load(Ordering::Acquire) {
                                auto_aim_clone.store(false, Ordering::Release);
                                tracing::info!("Auto aim modified to false");
                            }
                        });
                        m.handle_side5_holding(
                            // hold side5 mouse button ~20 millis second `turn off` auto aim
                            Duration::from_millis(0),
                            Duration::from_millis(50),
                            f,
                        );
                        running.store(false, Ordering::Relaxed);
                    });
                }

                mouse
            };

            let mut random = rand::rng();

            loop {
                if auto_aim.load(Ordering::Relaxed) {
                    if let Some(image) = frame_queue.pop() {
                        let mut bboxes = model.infer(&image)?;
                        bboxes.sort_by(|a, b| {
                            let dist_a = crosshair.l2_distance(&a.cxcy());
                            let dist_b = crosshair.l2_distance(&b.cxcy());
                            dist_a.partial_cmp(&dist_b).unwrap()
                        });
                        if let Some(build_head_iou) = config.build_head_iou {
                            bboxes.build(build_head_iou);
                        }
                        tracing::debug!("[Model] bboxes: {:?}", bboxes);

                        if bboxes.len() > 0 {
                            // if esp button 2 is triggered it's always aim head
                            let esp_button2_pressed = esp_button2.load(Ordering::Acquire);
                            let (destination, min_zone) = if esp_button2_pressed {
                                let (destination, min_zone) = aim.aim_head(&bboxes).unwrap();
                                (destination, min_zone * config.scale_min_zone2)
                            } else {
                                let (destination, min_zone) =
                                    aim.aim(&bboxes, &crosshair, &mut random).unwrap();
                                (destination, min_zone * config.scale_min_zone1)
                            };
                            let dist = destination.l2_distance(&crosshair).sqrt();

                            #[cfg(not(feature = "disable-mouse"))]
                            if dist > min_zone && dist <= config.fov {
                                let (dx, dy) = mouse_counts(
                                    (
                                        destination.x() - crosshair.x(),
                                        destination.y() - crosshair.y(),
                                    ),
                                    config.frame_to_screen,
                                    config.game_sens,
                                    config.mouse_dpi,
                                );
                                let use_trigger = trigger.load(Ordering::Acquire);
                                if (use_trigger
                                    && (esp_button1.load(Ordering::Acquire)
                                        || esp_button2_pressed
                                        || mouse.is_side4_pressing()))
                                    || (!use_trigger)
                                {
                                    mouse.move_bezier(dx, dy, &mut random)?;
                                }
                            }

                            #[cfg(feature = "debug")]
                            {
                                #[cfg(not(feature = "save-bbox"))]
                                let mut image = image;

                                // Detections are screen-space; the frame may be
                                // only the region (CAPTURE_ROI), so shift them
                                // into frame-local coordinates before drawing.
                                #[cfg(not(feature = "save-bbox"))]
                                let draw_origin = model.frame_origin(&image)?;

                                #[cfg(not(feature = "save-bbox"))]
                                {
                                    bboxes.class_0.iter().for_each(|b| {
                                        opencv::imgproc::rectangle(
                                            &mut image,
                                            opencv::core::Rect::new(
                                                b.xmin() as i32 - draw_origin.x,
                                                b.ymin() as i32 - draw_origin.y,
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
                                                b.xmin() as i32 - draw_origin.x,
                                                b.ymin() as i32 - draw_origin.y,
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
                                            destination.x() as i32 - draw_origin.x,
                                            destination.y() as i32 - draw_origin.y,
                                        ),
                                        3,
                                        opencv::core::Scalar::new(255., 0., 0., 0.),
                                        2,
                                        -1,
                                        0,
                                    )
                                    .unwrap();
                                }

                                let id = uuid::Uuid::new_v4().to_string();
                                let filename_png = format!("d-{id}.png");
                                let filename_txt = format!("d-{id}.txt");

                                #[cfg(feature = "save-bbox")]
                                {
                                    let mut txt = std::fs::File::create(format!(
                                        "{ROOT_PATH_DEBUG}/{filename_txt}"
                                    ))?;
                                    // Labels have to be normalised against the
                                    // image that is actually written next to
                                    // them, which is the region under
                                    // CAPTURE_ROI and the whole frame otherwise.
                                    let origin = model.frame_origin(&image)?;
                                    let (img_w, img_h) = (image.cols() as f32, image.rows() as f32);
                                    let mut f = |bboxes: &Vec<Bbox>, class: u32| {
                                        bboxes.iter().for_each(|bbox| {
                                            let center = bbox.cxcy();
                                            let (x, y) = (
                                                (center.x() - origin.x as f32) / img_w,
                                                (center.y() - origin.y as f32) / img_h,
                                            );
                                            let width = bbox.width() / img_w;
                                            let height = bbox.height() / img_h;
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
        start_event_listener(
            use_trigger,
            use_auto_aim,
            aim_mode,
            serving_port_event_listener,
        )
        .map_err(|err| {
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
    tracing::warn!("Server stopped.");
    Ok(())
}
