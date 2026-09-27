use aimbot::{
    config::Config,
    mouse::{MouseVirtual, SmoothAim, plan_flick},
    stream::open_source,
};
use opencv::{
    core::{self, Mat, MatTraitConst, Rect},
    imgproc,
};
use std::{
    error::Error,
    sync::{
        Arc,
        mpsc::{self, Receiver, RecvTimeoutError},
    },
    thread,
    time::{Duration, Instant},
};
use tracing_subscriber::{EnvFilter, Layer, fmt, layer::SubscriberExt, util::SubscriberInitExt};

fn main() -> Result<(), Box<dyn Error>> {
    dotenv::dotenv().ok();
    tracing_subscriber::registry()
        .with(fmt::Layer::new().with_writer(std::io::stdout).with_filter(
            EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new("info"))?,
        ))
        .init();
    let config = Config::new();
    let mouse = Arc::new(MouseVirtual::new(&config.makcu_port, config.makcu_baud)?);

    // `mouse_test latency [counts] [samples]` measures the capture latency and
    // skips the interactive button checks.
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("latency") {
        let counts = match args.get(2) {
            Some(v) => v.parse::<i32>()?,
            None => 30,
        };
        let samples = match args.get(3) {
            Some(v) => v.parse::<usize>()?,
            None => 20,
        };
        return measure_latency(&config, &mouse, counts, samples);
    }

    let mouse_clone = mouse.clone();
    thread::spawn(move || {
        mouse_clone.listen_button_presses();
    });
    tracing::info!("[1] Testing for left mouse button presses");
    tracing::info!("[1] Please hold your left mouse button");
    while !mouse.is_left_pressing() {
        thread::sleep(Duration::from_millis(100));
    }
    tracing::info!("[1] Testing left mouse button presses successfully");
    tracing::info!("---------------------------------------------------");
    tracing::info!("[2] Testing for right mouse button presses");
    tracing::info!("[2] Please hold your right mouse button");
    while !mouse.is_right_pressing() {
        thread::sleep(Duration::from_millis(100));
    }
    tracing::info!("[2] Testing right mouse button presses successfully");
    tracing::info!("---------------------------------------------------");
    tracing::info!("[3] Testing for side4 mouse button presses");
    tracing::info!("[3] Please hold your side4 mouse button");
    while !mouse.is_side4_pressing() {
        thread::sleep(Duration::from_millis(100));
    }
    tracing::info!("[3] Testing side4 mouse button presses successfully");
    tracing::info!("---------------------------------------------------");
    tracing::info!("[4] Testing for side5 mouse button presses");
    tracing::info!("[4] Please hold your side5 mouse button");
    while !mouse.is_side5_pressing() {
        thread::sleep(Duration::from_millis(100));
    }
    tracing::info!("[4] Testing side5 mouse button presses successfully");
    tracing::info!("---------------------------------------------------");
    let mut random = rand::rng();
    let mut smooth = SmoothAim::new(config.smooth, config.makcu_baud);
    let mut use_smooth = true;
    tracing::info!("[5] Testing for mouse move");
    tracing::info!("[5] Input separate for dy, dy; type q to quit, s to switch path");
    tracing::info!(
        "[5] Smooth path reports at {} Hz; watch that the measured time tracks \
         the planned one, because a firmware that coalesces reports would \
         swallow the velocity profile without saying so",
        smooth.poll_hz()
    );
    loop {
        let mut value = String::new();
        tracing::info!("dx: ");
        std::io::stdin().read_line(&mut value)?;
        let v = value.trim();
        if v.trim() == "q" {
            break;
        }
        if v.trim() == "s" {
            use_smooth = !use_smooth;
            tracing::info!("path: {}", if use_smooth { "smooth" } else { "bezier" });
            continue;
        }
        let dx = v.parse::<i64>()?;

        let mut value = String::new();
        tracing::info!("dy: ");
        std::io::stdin().read_line(&mut value)?;
        let v = value.trim();
        if v.trim() == "q" {
            break;
        }
        let dy = v.parse::<i64>()?;
        if use_smooth {
            // Fitts wants screen pixels and a target width. There is no frame
            // here to measure against, so the count magnitude stands in for
            // the reach and 64 px for the width, which is about what a body
            // box gives the aim thread. Enough to feel the profile, not a
            // calibration.
            let reach = ((dx * dx + dy * dy) as f64).sqrt() as f32;
            match plan_flick(&mut smooth, (dx as f64, dy as f64), reach, 64., &mut random) {
                Some(flick) => {
                    let writes = flick
                        .steps
                        .iter()
                        .filter(|s| (s.dx, s.dy) != (0, 0))
                        .count();
                    let start = Instant::now();
                    mouse.play_flick(&flick, || true)?;
                    tracing::info!(
                        "moved {:?} over {} reports ({writes} on the wire), planned {:?}, took {:?}",
                        flick.total,
                        flick.steps.len(),
                        flick.duration,
                        start.elapsed()
                    );
                }
                _ => tracing::info!("nothing to move"),
            }
        } else {
            mouse.move_bezier(dx as f64, dy as f64, &mut random)?;
        }
        tracing::info!("-------------------------");
    }
    Ok(())
}

/// How long a mouse move takes to show up in the frames this process receives:
/// MAKCU, the game, rendering, capture and transport together.
///
/// It decides whether a smooth preset is stable. A frame shows the game as it
/// was this long ago, so a flick shorter than it is still invisible in the
/// frame read right after it, and the same error gets commanded twice.
///
/// Each sample waits for the view to be still, sends one instant `km.move`, and
/// times the first frame whose centre differs from the pre-move frame by more
/// than the scene's own noise.
fn measure_latency(
    config: &Config,
    mouse: &MouseVirtual,
    counts: i32,
    samples: usize,
) -> Result<(), Box<dyn Error>> {
    tracing::info!(
        "[latency] Stand still in game, facing a detailed, static surface — a \
         textured wall, not the sky — with the game focused and not in a menu. \
         Do not touch the mouse. The aimbot must not be running."
    );
    tracing::info!("[latency] Each sample turns the view {counts} counts sideways, then back.");

    let mut cap = open_source(config)?;
    let (tx, rx) = mpsc::channel::<(Instant, Mat)>();
    thread::spawn(move || {
        loop {
            match cap.capture() {
                Ok(frame) => {
                    // Stamp before any processing: the arrival time is the
                    // measurement.
                    let at = Instant::now();
                    match centre_patch(&frame) {
                        Ok(patch) => {
                            if tx.send((at, patch)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::error!("[latency] {e}");
                            break;
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("[latency] capture failed: {e}, reconnecting");
                    if cap.reconnect().is_err() {
                        thread::sleep(Duration::from_millis(200));
                    }
                }
            }
        }
    });

    // The scene's own frame-to-frame noise sets the bar a move has to clear.
    tracing::info!("[latency] Measuring the scene's own noise for 2 s...");
    let (mut noise, mut gaps, mut prev) = (Vec::new(), Vec::new(), None::<(Instant, Mat)>);
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(2) {
        let (at, patch) = rx.recv_timeout(Duration::from_secs(2))?;
        if let Some((pat, p)) = &prev {
            noise.push(mean_abs_diff(&patch, p)?);
            gaps.push(ms(at - *pat));
        }
        prev = Some((at, patch));
    }
    if noise.len() < 10 {
        return Err("fewer than 10 frames arrived in 2 s; is the stream running?".into());
    }
    noise.sort_by(f64::total_cmp);
    gaps.sort_by(f64::total_cmp);
    let frame_ms = percentile(&gaps, 0.5);
    let floor = percentile(&noise, 0.99);
    // Two still frames never match exactly — compression and sensor noise put a
    // steady level under the difference — so the bar sits a margin above how
    // far that level *wanders*, not a multiple of the level itself. A multiple
    // would climb above the change a real move makes on a noisy stream.
    let threshold = floor + (3. * (floor - percentile(&noise, 0.5))).max(1.5);
    tracing::info!(
        "[latency] {} frames, one every {frame_ms:.1} ms; scene noise {floor:.2}, \
         detection threshold {threshold:.2}",
        noise.len() + 1
    );
    if floor > 8. {
        tracing::warn!(
            "[latency] The view changes on its own (noise {floor:.1}). Find a stiller \
             spot, or the samples will be noisy."
        );
    }

    let (mut latencies, mut changes, mut missed) = (Vec::new(), Vec::new(), 0);
    let mut dir = 1;
    for i in 1..=samples {
        let Some(baseline) = settle(&rx, threshold, 8, Duration::from_secs(3))? else {
            tracing::warn!("[latency] sample {i:>2}: the view never settled, skipped");
            missed += 1;
            continue;
        };
        // `settle` returns the moment a frame arrives, which would send every
        // move at the same point of the game's frame clock and round every
        // sample up to the same frame boundary. A random wait spreads the
        // samples over a whole frame instead.
        thread::sleep(Duration::from_secs_f64(
            rand::random::<f64>() * frame_ms / 1000.,
        ));
        let t0 = Instant::now();
        mouse.move_shift((dir * counts) as f64, 0.)?;
        dir = -dir;
        match first_change(&rx, &baseline, t0, threshold, Duration::from_millis(1500))? {
            Some((latency, change)) => {
                tracing::info!(
                    "[latency] sample {i:>2}: {:>6.1} ms (change {change:.1})",
                    ms(latency)
                );
                latencies.push(ms(latency));
                changes.push(change);
            }
            None => {
                tracing::warn!("[latency] sample {i:>2}: no change within 1.5 s");
                missed += 1;
            }
        }
    }
    // An odd number of moves leaves the view turned; put it back.
    if dir == -1 {
        mouse.move_shift((dir * counts) as f64, 0.)?;
    }

    if latencies.is_empty() {
        return Err(
            "no sample saw the view change; check that the game has focus, \
                    is not in a menu, and the view is textured"
                .into(),
        );
    }
    latencies.sort_by(f64::total_cmp);
    let mean_change = changes.iter().sum::<f64>() / changes.len() as f64;
    tracing::info!("[latency] ==================================================");
    tracing::info!(
        "[latency] {} samples, {missed} missed, a frame every {frame_ms:.1} ms",
        latencies.len()
    );
    tracing::info!(
        "[latency] min {:.0} ms   median {:.0} ms   p90 {:.0} ms   max {:.0} ms",
        latencies[0],
        percentile(&latencies, 0.5),
        percentile(&latencies, 0.9),
        latencies[latencies.len() - 1]
    );
    tracing::info!(
        "[latency] Samples spread over one frame ({frame_ms:.0} ms): the min is \
         close to the pipeline delay itself, the median about half a frame above."
    );
    if mean_change < 2. * threshold {
        tracing::warn!(
            "[latency] Moves only just cleared the threshold (change {mean_change:.1} \
             against {threshold:.1}), so samples may read late. Face a more textured \
             surface or pass a larger count."
        );
    }
    tracing::info!("[latency] Report the median and the p90.");
    Ok(())
}

/// The grey centre of a frame, at most 256 px square. Turning the view moves
/// everything in it, so the middle is enough — and it is what `CAPTURE_ROI`
/// delivers anyway.
fn centre_patch(frame: &Mat) -> opencv::Result<Mat> {
    let side = 256.min(frame.cols()).min(frame.rows());
    let rect = Rect::new(
        (frame.cols() - side) / 2,
        (frame.rows() - side) / 2,
        side,
        side,
    );
    let view = Mat::roi(frame, rect)?.clone_pointee();
    let mut grey = Mat::default();
    match view.channels() {
        1 => view.copy_to(&mut grey)?,
        3 => imgproc::cvt_color_def(&view, &mut grey, imgproc::COLOR_BGR2GRAY)?,
        4 => imgproc::cvt_color_def(&view, &mut grey, imgproc::COLOR_BGRA2GRAY)?,
        n => {
            return Err(opencv::Error::new(
                core::StsBadArg,
                format!("{n}-channel frames are not supported; set CAPTURE_OUTPUT=bgr"),
            ));
        }
    }
    Ok(grey)
}

/// Mean absolute difference between two grey patches, 0 to 255.
fn mean_abs_diff(a: &Mat, b: &Mat) -> opencv::Result<f64> {
    let mut diff = Mat::default();
    core::absdiff(a, b, &mut diff)?;
    let pixels = (diff.rows() * diff.cols()).max(1) as f64;
    Ok(core::sum_elems(&diff)?[0] / pixels)
}

/// Wait for `need` consecutive frames that each differ from the one before by
/// less than `threshold`, and return the last of them as the baseline.
fn settle(
    rx: &Receiver<(Instant, Mat)>,
    threshold: f64,
    need: usize,
    timeout: Duration,
) -> Result<Option<Mat>, Box<dyn Error>> {
    let deadline = Instant::now() + timeout;
    let (mut prev, mut still) = (None::<Mat>, 0);
    loop {
        let (_, patch) = match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(frame) => frame,
            Err(RecvTimeoutError::Timeout) => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        if let Some(p) = &prev {
            still = if mean_abs_diff(&patch, p)? < threshold {
                still + 1
            } else {
                0
            };
        }
        prev = Some(patch);
        if still >= need {
            return Ok(prev);
        }
    }
}

/// Time from `t0` to the arrival of the first frame that differs from
/// `baseline` by more than `threshold`.
fn first_change(
    rx: &Receiver<(Instant, Mat)>,
    baseline: &Mat,
    t0: Instant,
    threshold: f64,
    timeout: Duration,
) -> Result<Option<(Duration, f64)>, Box<dyn Error>> {
    let deadline = t0 + timeout;
    loop {
        let (at, patch) = match rx.recv_timeout(deadline.saturating_duration_since(Instant::now()))
        {
            Ok(frame) => frame,
            Err(RecvTimeoutError::Timeout) => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        // Arrived before the move was even sent.
        if at < t0 {
            continue;
        }
        let change = mean_abs_diff(&patch, baseline)?;
        if change > threshold {
            return Ok(Some((at - t0, change)));
        }
    }
}

/// `q`-th quantile of an already sorted, non-empty slice.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.
}
