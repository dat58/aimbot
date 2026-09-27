use crate::config::{SmoothConfig, poll_hz_for_baud};
use anyhow::{Result, bail};
use rand::prelude::*;
use serialport::{self, SerialPort, available_ports};
use std::io::Write;
use std::{
    f64::consts::{PI, TAU},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::sleep,
    time::{Duration, Instant},
};

const BAUD_CHANGE_COMMAND: [u8; 9] = [0xDE, 0xAD, 0x05, 0x00, 0xA5, 0x00, 0x09, 0x3D, 0x00];
const VERIFY_COMMAND: &[u8] = b"km.version()\r\n";
const DEFAULT_BAUD_RATE: u32 = 115_200;
const ALLOWED_BAUD_RATE: [u32; 3] = [115_200, 2_000_000, 4_000_000];
const CRLF: &str = "\r\n";
/// `thread::sleep` on Linux overshoots by tens of microseconds, so below this
/// it is cheaper to fire a report a touch early than to oversleep it.
const SLEEP_FLOOR: Duration = Duration::from_micros(150);

pub struct MouseVirtual {
    serial: Mutex<Box<dyn SerialPort>>,
    pressed: [AtomicBool; 5],
}

impl MouseVirtual {
    pub fn new(port: &str, baud: u32) -> Result<Self> {
        tracing::debug!("All available serial port: {:?}", available_ports());
        if !ALLOWED_BAUD_RATE.contains(&baud) {
            bail!("Baud rate out of range, allowed: {:?}", ALLOWED_BAUD_RATE);
        }
        let mut serial = serialport::new(port, baud)
            .timeout(Duration::from_millis(300))
            .open()?;
        let serial = match Self::check_km_version_ok(&mut serial) {
            Ok(_) => {
                drop(serial);
                sleep(Duration::from_millis(300));
                let mut serial = serialport::new(port, baud)
                    .timeout(Duration::from_millis(100))
                    .open()?;
                serial.write_all(format!("km.buttons(1){CRLF}").as_bytes())?;
                serial
            }
            Err(_) => {
                {
                    drop(serial);
                    sleep(Duration::from_millis(300));
                    tracing::info!(
                        "Check KM version unavailable at baud: {}, try change baud rate...",
                        baud
                    );
                    let mut serial = serialport::new(port, DEFAULT_BAUD_RATE)
                        .timeout(Duration::from_millis(300))
                        .open()?;
                    sleep(Duration::from_millis(200));
                    serial.write_all(&BAUD_CHANGE_COMMAND)?;
                    drop(serial);
                    sleep(Duration::from_millis(200));
                }
                {
                    let mut serial = serialport::new(port, baud)
                        .timeout(Duration::from_millis(300))
                        .open()?;
                    sleep(Duration::from_millis(200));
                    Self::check_km_version_ok(&mut serial)?;
                    tracing::info!("Check KM version available at baud: {}", baud);
                }
                sleep(Duration::from_millis(200));
                let mut serial = serialport::new(port, baud)
                    .timeout(Duration::from_millis(100))
                    .open()?;
                serial.write_all(format!("km.buttons(1){CRLF}").as_bytes())?;
                serial
            }
        };
        tracing::info!("Mouse connected at baud rate: {:?}", serial.baud_rate());
        Ok(Self {
            serial: Mutex::new(serial),
            pressed: Default::default(),
        })
    }

    fn check_km_version_ok(serial: &mut Box<dyn SerialPort>) -> Result<()> {
        serial.clear(serialport::ClearBuffer::Input)?;
        serial.clear(serialport::ClearBuffer::Output)?;
        serial.write_all(VERIFY_COMMAND)?;
        let mut verification_response = String::new();
        let mut buffer = [0; 128];
        loop {
            match serial.read(&mut buffer) {
                Ok(bytes_read) => {
                    if bytes_read > 0 {
                        verification_response
                            .push_str(&String::from_utf8_lossy(&buffer[..bytes_read]));
                        if verification_response.contains("km.MAKCU") {
                            break;
                        }
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => {
                    tracing::error!(
                        "Verification MAKCU change baud rate timed out. Check the connection."
                    );
                    bail!(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "Timeout during verification"
                    ));
                }
                Err(e) => {
                    bail!(e);
                }
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn cmd(&self, command: &str) -> Result<()> {
        let mut serial = self.serial.lock().expect("Failed to lock serial port");
        Ok(serial.write_all(format!("{command}{CRLF}").as_bytes())?)
    }

    pub fn move_shift(&self, dx: f64, dy: f64) -> Result<()> {
        let dx = dx as i32;
        let dy = dy as i32;
        self.cmd(format!("km.move({dx},{dy})").as_str())
    }

    pub fn move_bezier(&self, dx: f64, dy: f64, random: &mut ThreadRng) -> Result<()> {
        let (steps, ref_x, ref_y) = self.find_bezier(dx, dy, random);
        self.cmd(format!("km.move({dx},{dy},{steps},{ref_x},{ref_y})").as_str())
    }

    /// Move toward `delta` (mouse counts) the way a hand does: plan a flick
    /// with [`plan_flick`], then play it with [`Self::play_flick`].
    ///
    /// The counterpart of [`Self::move_bezier`], which hands the whole delta to
    /// the firmware in one command. `reach_px` and `width_px` are the distance
    /// to the target and its width in screen pixels; Fitts's law needs the
    /// *visual* difficulty of the shot, which counts do not carry. Blocks for
    /// the length of the flick.
    pub fn move_smooth(
        &self,
        smooth: &mut SmoothAim,
        delta: (f64, f64),
        reach_px: f32,
        width_px: f32,
        random: &mut impl Rng,
        keep_going: impl Fn() -> bool,
    ) -> Result<()> {
        match plan_flick(smooth, delta, reach_px, width_px, random) {
            Some(flick) => self.play_flick(&flick, keep_going),
            None => Ok(()),
        }
    }

    /// Play a flick planned by [`plan_flick`], one `km.move` per report
    /// deadline.
    ///
    /// Deadlines are absolute offsets from the first report, so a `sleep` that
    /// overshoots never accumulates. A deadline that has already gone by folds
    /// its report into the next one rather than stretching the flick out; the
    /// counts are integers, so the total lands exactly either way.
    ///
    /// `keep_going` is checked once per report, because the alternative is not
    /// re-reading the trigger for the length of the movement. A hand abandons a
    /// flick when the reason for it goes away, so this does too.
    pub fn play_flick(&self, flick: &Flick, keep_going: impl Fn() -> bool) -> Result<()> {
        let start = Instant::now();
        let last = flick.steps.len().saturating_sub(1);
        let (mut dx, mut dy) = (0i64, 0i64);
        for (i, step) in flick.steps.iter().enumerate() {
            if !keep_going() {
                break;
            }
            dx += step.dx as i64;
            dy += step.dy as i64;
            match (start + step.at).checked_duration_since(Instant::now()) {
                Some(wait) if wait > SLEEP_FLOOR => sleep(wait),
                Some(_) => {}
                // Already late: roll this report into the next one instead of
                // pushing every later deadline back by the same amount.
                None if i < last => continue,
                None => {}
            }
            // A real mouse does send zero-motion reports, but the firmware
            // gains nothing from them and the wire time is better spent
            // elsewhere. The deadline was still honoured, so the slow head and
            // the long tail of the profile keep their shape.
            if (dx, dy) != (0, 0) {
                self.cmd(format!("km.move({dx},{dy})").as_str())?;
                dx = 0;
                dy = 0;
            }
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn find_bezier(&self, dx: f64, dy: f64, random: &mut ThreadRng) -> (i32, i32, i32) {
        let pixel = (dx * dx + dy * dy).sqrt();
        let lower = (pixel * 0.2) as i32;
        let upper = (pixel * 0.55) as i32 + 1;
        let steps = random.random_range(lower..=upper);
        let ref_x = random.random_range(1..6);
        let ref_y = random.random_range(1..6);
        (steps, ref_x, ref_y)
    }

    pub fn listen_button_presses(self: Arc<Self>) {
        let mut last_value = 0;
        let mut buf = [0; 8];
        loop {
            // The port carries a 100 ms read timeout and an idle one is the
            // normal case, so reading under the lock would hold it for the full
            // timeout and stall every `km.move` queued behind it. A timed
            // report profile cannot survive that, so ask first and only read
            // when there is something to read.
            let bytes_read = {
                let mut serial = self.serial.lock().expect("Could not acquire serial lock");
                match serial.bytes_to_read() {
                    Ok(0) | Err(_) => {
                        drop(serial);
                        sleep(Duration::from_millis(1));
                        continue;
                    }
                    _ => serial.read(&mut buf),
                }
            };
            match bytes_read {
                Ok(bytes_read) => {
                    if bytes_read > 0 {
                        buf[..bytes_read].iter().for_each(|v| {
                            let v = *v;
                            if v != 0x0A && v != 0x0D && v < 32 {
                                let changed = last_value ^ v;
                                if changed > 0 {
                                    for i in 0..self.pressed.len() {
                                        let m = 1 << i;
                                        if changed & m > 0 {
                                            self.pressed[i].store(v & m > 0, Ordering::Release);
                                        }
                                    }
                                    last_value = v;
                                }
                            }
                        });
                    }
                }
                _ => {}
            }
            sleep(Duration::from_millis(1));
        }
    }

    fn is_button_pressing(&self, button: usize) -> bool {
        self.pressed[button].load(Ordering::Acquire)
    }

    fn handle_button_holding(
        self: Arc<Self>,
        button: usize,
        hold_duration: Duration,
        interval: Duration,
        f: Box<dyn Fn() -> ()>,
    ) {
        loop {
            if self.pressed[button].load(Ordering::Acquire) {
                let time = Instant::now();
                loop {
                    sleep(interval);
                    if self.pressed[button].load(Ordering::Acquire) {
                        if time.elapsed() >= hold_duration {
                            f();
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }
            sleep(interval);
        }
    }

    pub fn is_left_pressing(&self) -> bool {
        self.is_button_pressing(0)
    }

    pub fn is_right_pressing(&self) -> bool {
        self.is_button_pressing(1)
    }

    pub fn is_middle_pressing(&self) -> bool {
        self.is_button_pressing(2)
    }

    pub fn is_side4_pressing(&self) -> bool {
        self.is_button_pressing(3)
    }

    pub fn is_side5_pressing(&self) -> bool {
        self.is_button_pressing(4)
    }

    pub fn handle_left_holding(
        self: Arc<Self>,
        hold_duration: Duration,
        interval: Duration,
        f: Box<dyn Fn() -> ()>,
    ) {
        self.handle_button_holding(0, hold_duration, interval, f);
    }

    pub fn handle_right_holding(
        self: Arc<Self>,
        hold_duration: Duration,
        interval: Duration,
        f: Box<dyn Fn() -> ()>,
    ) {
        self.handle_button_holding(1, hold_duration, interval, f);
    }

    pub fn handle_middle_holding(
        self: Arc<Self>,
        hold_duration: Duration,
        interval: Duration,
        f: Box<dyn Fn() -> ()>,
    ) {
        self.handle_button_holding(2, hold_duration, interval, f);
    }

    pub fn handle_side4_holding(
        self: Arc<Self>,
        hold_duration: Duration,
        interval: Duration,
        f: Box<dyn Fn() -> ()>,
    ) {
        self.handle_button_holding(3, hold_duration, interval, f);
    }

    pub fn handle_side5_holding(
        self: Arc<Self>,
        hold_duration: Duration,
        interval: Duration,
        f: Box<dyn Fn() -> ()>,
    ) {
        self.handle_button_holding(4, hold_duration, interval, f);
    }

    /// Lock physical mouse on X-axis direction
    pub fn lock_mx(&self) -> Result<()> {
        self.cmd("km.lock_mx(1)")
    }

    /// Unlock physical mouse on X-axis direction
    pub fn unlock_mx(&self) -> Result<()> {
        self.cmd("km.lock_mx(0)")
    }

    /// Lock physical mouse on Y-axis direction
    pub fn lock_my(&self) -> Result<()> {
        self.cmd("km.lock_my(1)")
    }

    /// Unlock physical mouse on Y-axis direction
    pub fn unlock_my(&self) -> Result<()> {
        self.cmd("km.lock_my(0)")
    }

    pub fn click_left(&self) -> Result<()> {
        self.cmd(format!("km.left(1){CRLF}km.left(0)").as_str())
    }

    pub fn click_right(&self) -> Result<()> {
        self.cmd(format!("km.right(1){CRLF}km.right(0)").as_str())
    }

    pub fn batch(&self) -> BatchCommands<'_> {
        BatchCommands::new(self)
    }
}

pub struct BatchCommands<'a> {
    mouse: &'a MouseVirtual,
    buf: String,
}

impl<'a> BatchCommands<'a> {
    pub fn new(mouse: &'a MouseVirtual) -> Self {
        Self {
            mouse,
            buf: String::new(),
        }
    }

    pub fn move_shift(mut self, dx: f64, dy: f64) -> Self {
        let dx = dx as i32;
        let dy = dy as i32;
        self.buf
            .push_str(format!("km.move({dx},{dy}){CRLF}").as_str());
        self
    }

    pub fn move_bezier(mut self, dx: f64, dy: f64, random: &mut ThreadRng) -> Self {
        let (steps, ref_x, ref_y) = self.mouse.find_bezier(dx, dy, random);
        self.buf
            .push_str(format!("km.move({dx},{dy},{steps},{ref_x},{ref_y}){CRLF}").as_str());
        self
    }

    pub fn lock_mx(mut self) -> Self {
        self.buf.push_str(format!("km.lock_mx(1){CRLF}").as_str());
        self
    }

    pub fn unlock_mx(mut self) -> Self {
        self.buf.push_str(format!("km.lock_mx(0){CRLF}").as_str());
        self
    }

    pub fn lock_my(mut self) -> Self {
        self.buf.push_str(format!("km.lock_my(1){CRLF}").as_str());
        self
    }

    pub fn unlock_my(mut self) -> Self {
        self.buf.push_str(format!("km.lock_my(0){CRLF}").as_str());
        self
    }

    pub fn click_left(mut self) -> Self {
        self.buf
            .push_str(format!("km.left(1){CRLF}km.left(0){CRLF}").as_str());
        self
    }

    pub fn click_right(mut self) -> Self {
        self.buf
            .push_str(format!("km.right(1){CRLF}km.right(0){CRLF}").as_str());
        self
    }

    pub fn run(&self) -> Result<()> {
        self.mouse.cmd(self.buf.as_str())
    }
}

/// The standard normal's 99th percentile, used to pin the lognormal tail to
/// the end of the movement.
const Z99: f64 = 2.3263478740408408;

/// One mouse report: the integer counts that go on the wire at `at`, measured
/// from the first report of the flick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoveStep {
    pub dx: i32,
    pub dy: i32,
    pub at: Duration,
}

/// A planned flick, ready for [`MouseVirtual::play_flick`].
#[derive(Debug, Clone)]
pub struct Flick {
    pub steps: Vec<MoveStep>,
    /// What this flick actually commits to, already cut by the ballistic gain.
    /// This is deliberately *not* the delta that was asked for.
    pub total: (i32, i32),
    pub duration: Duration,
}

/// Carries the little state a sequence of flicks needs between frames: the
/// sub-count remainder, and when the hand is allowed to start moving.
pub struct SmoothAim {
    cfg: SmoothConfig,
    poll_hz: u32,
    /// Counts asked for but not yet emitted. Under half a count unless a
    /// report hit the per-report ceiling, in which case it holds the deferred
    /// excess.
    carry: (f64, f64),
    /// When the flick already planned is expected to finish.
    last_flick: Option<Instant>,
    /// No flick before this instant; a fresh acquisition sets it.
    ready_at: Option<Instant>,
    /// Reused between flicks so the steady state does not allocate.
    weights: Vec<f64>,
}

impl SmoothAim {
    pub fn new(cfg: SmoothConfig, baud: u32) -> Self {
        let poll_hz = if cfg.poll_hz > 0 {
            cfg.poll_hz
        } else {
            poll_hz_for_baud(baud)
        };
        Self {
            cfg,
            poll_hz: poll_hz.clamp(1, 8000),
            carry: (0., 0.),
            last_flick: None,
            ready_at: None,
            weights: Vec::new(),
        }
    }

    pub fn poll_hz(&self) -> u32 {
        self.poll_hz
    }

    #[cfg(test)]
    pub(crate) fn carry(&self) -> (f64, f64) {
        self.carry
    }
}

/// Movement time for a `d_px` reach at a `w_px` target, Fitts's law in Shannon
/// form. The `+ 1` keeps the index of difficulty non-negative and finite as
/// the distance goes to zero.
///
/// The point of this function is that the time grows with the *logarithm* of
/// the distance. A move ten times as far costs a small constant extra, not ten
/// times as long — which is what makes a step count proportional to distance
/// read as machine-generated at a glance.
pub fn fitts_duration_ms(d_px: f64, w_px: f64, a: f64, b: f64) -> f64 {
    a + b * (2. * d_px / w_px.max(1.) + 1.).log2()
}

/// Lognormal parameters whose speed mode sits at `peak_frac * mt_ms` and whose
/// 99th percentile lands on `mt_ms`, so cutting the tail off at the end of the
/// movement loses under a percent.
///
/// Pinning both gives `sigma^2 + Z99*sigma + ln(peak_frac) = 0`, hence the
/// closed form below. No `erf` is needed, and where the speed peaks becomes a
/// direct parameter instead of something to solve for numerically.
pub fn lognormal_shape(mt_ms: f64, peak_frac: f64) -> (f64, f64) {
    let f = peak_frac.clamp(0.20, 0.60);
    let sigma = (-Z99 + (Z99 * Z99 - 4. * f.ln()).sqrt()) / 2.;
    (mt_ms.ln() - Z99 * sigma, sigma)
}

pub fn lognormal_pdf(t: f64, mu: f64, sigma: f64) -> f64 {
    if t <= 0. {
        return 0.;
    }
    let z = (t.ln() - mu) / sigma;
    (-0.5 * z * z).exp() / (t * sigma * TAU.sqrt())
}

/// Fills `out` with each report's share of the stroke, summing to exactly 1.
///
/// The flick is played on a discrete grid of reports, so the cumulative
/// distribution is never needed: sampling the speed density at report
/// midpoints and normalising by the sum makes the running total of those
/// samples *be* the displacement profile. Midpoints also step around the
/// `ln(0)` singularity at the start.
fn render_weights(out: &mut Vec<f64>, mt_ms: f64, peak_frac: f64, n: usize) {
    out.clear();
    out.reserve(n);
    let (mu, sigma) = lognormal_shape(mt_ms, peak_frac);
    let dt = mt_ms / n as f64;
    let mut sum = 0.;
    for k in 0..n {
        let w = lognormal_pdf((k as f64 + 0.5) * dt, mu, sigma);
        out.push(w);
        sum += w;
    }
    // Degenerate parameters would divide by zero. A constant speed is a poor
    // flick, but it is a finite one.
    if !(sum > 0.) || !sum.is_finite() {
        out.clear();
        out.resize(n, 1. / n as f64);
        return;
    }
    for w in out.iter_mut() {
        *w /= sum;
    }
}

/// One standard normal from two uniforms (Box-Muller). `rand` carries no
/// normal distribution of its own and `rand_distr` is not worth a dependency
/// for six lines.
fn gauss(random: &mut impl Rng) -> f64 {
    // `random::<f64>()` is [0, 1), and `ln(0)` would poison the whole plan
    // with NaN.
    let u1 = random.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = random.random::<f64>();
    (-2. * u1.ln()).sqrt() * (TAU * u2).cos()
}

/// Uniform on `[lo, hi]`, total where `random_range` is not: that one panics
/// on an empty range, which is the bug commit 996fcf9 had to go back and fix.
fn uniform(random: &mut impl Rng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * random.random::<f64>()
}

/// Physiological tremor: two components that share no common period, because
/// real tremor is a band around 8-12 Hz rather than a pure tone. The window
/// takes it to zero at both ends of the stroke, so it cannot shift the
/// endpoint.
#[derive(Clone, Copy)]
struct Tremor {
    amp: f64,
    w1: f64,
    w2: f64,
    p1: f64,
    p2: f64,
}

impl Tremor {
    fn new(cfg: &SmoothConfig, mt_ms: f64, random: &mut impl Rng) -> Self {
        let f1 = uniform(random, 8., 12.);
        let f2 = f1 * uniform(random, 1.4, 1.9);
        // Progress runs 0..1 over the stroke, so a frequency in hertz turns
        // into this much phase across it.
        let seconds = mt_ms / 1000.;
        Self {
            amp: cfg.tremor,
            w1: TAU * f1 * seconds,
            w2: TAU * f2 * seconds,
            p1: uniform(random, 0., TAU),
            p2: uniform(random, 0., TAU),
        }
    }

    fn at(&self, u: f64) -> f64 {
        if self.amp <= 0. {
            return 0.;
        }
        self.amp
            * (PI * u).sin()
            * ((self.w1 * u + self.p1).sin() + 0.5 * (self.w2 * u + self.p2).sin())
    }
}

/// Plan a ballistic flick toward `delta`, in mouse counts.
///
/// `d_px` and `w_px` are the reach and the target width in **screen pixels**,
/// not counts: how long a hand takes depends on the *visual* difficulty of the
/// shot, not on the mouse's DPI. Feeding counts to Fitts would make `MOUSE_DPI`
/// change the duration of the gesture, which is wrong physically and is
/// exactly the sort of artefact worth not leaving behind.
///
/// The flick deliberately stops short of the target. The next frame sees what
/// is left and plans the corrective submovement, which is what makes the
/// correction visually guided rather than dead reckoning against a target that
/// has since moved.
pub fn plan_flick(
    state: &mut SmoothAim,
    delta: (f64, f64),
    d_px: f32,
    w_px: f32,
    random: &mut impl Rng,
) -> Option<Flick> {
    plan_flick_at(state, delta, d_px, w_px, random, Instant::now())
}

/// [`plan_flick`] with the clock injected, so the reaction window can be
/// tested without sleeping.
pub fn plan_flick_at(
    state: &mut SmoothAim,
    delta: (f64, f64),
    d_px: f32,
    w_px: f32,
    random: &mut impl Rng,
    now: Instant,
) -> Option<Flick> {
    // `mouse_counts` divides by GAME_SENS and MOUSE_DPI. `NaN as i32`
    // saturates to zero rather than panicking, so an infinite delta would
    // surface as an aim that silently stops instead of as a crash.
    if !delta.0.is_finite() || !delta.1.is_finite() || !d_px.is_finite() || !w_px.is_finite() {
        return None;
    }
    let cfg = state.cfg;

    // Nothing in the pipeline carries a target id, so a fresh acquisition is
    // inferred from silence: if nothing was aimed at for a while, the leftover
    // fraction describes a move toward a target that no longer exists.
    let gap = Duration::from_millis(cfg.gap_ms);
    let stale = state
        .last_flick
        .is_none_or(|end| now.saturating_duration_since(end) > gap);
    if stale && state.ready_at.is_none() {
        state.carry = (0., 0.);
        let (lo, hi) = (
            cfg.react_min_ms.min(cfg.react_max_ms),
            cfg.react_min_ms.max(cfg.react_max_ms),
        );
        if hi > 0 {
            let wait = lo + ((hi - lo) as f64 * random.random::<f64>()).round() as u64;
            state.ready_at = Some(now + Duration::from_millis(wait));
        }
    }
    if let Some(ready) = state.ready_at {
        if now < ready {
            return None; // still watching; the hand has not started yet
        }
        state.ready_at = None;
    }

    // Woodworth: a ballistic reach stops short on purpose, because overshooting
    // forces a reversal and costs more than one more nudge forward. The
    // minority that do overshoot are what put a visible direction reversal in
    // the next correction.
    let gain = if random.random::<f64>() < cfg.overshoot_p {
        (1.06 + 0.04 * gauss(random).clamp(-3., 3.)).clamp(1., 1.25)
    } else {
        (cfg.gain + cfg.gain_sd * gauss(random).clamp(-3., 3.)).clamp(0.05, 1.05)
    };

    let target = (
        delta.0 * gain + state.carry.0,
        delta.1 * gain + state.carry.1,
    );
    let amp = target.0.hypot(target.1);
    if !(amp > 0.) {
        return None;
    }

    let mt = fitts_duration_ms(d_px as f64, w_px as f64, cfg.fitts_a_ms, cfg.fitts_b_ms);
    // Multiplicative jitter: it cannot produce a negative time, and it keeps
    // the spread proportional the way human variability is.
    let mt = mt * (cfg.mt_jitter * gauss(random).clamp(-3., 3.)).exp();
    let mt = mt.clamp(
        cfg.min_ms.min(cfg.max_ms).max(1.),
        cfg.min_ms.max(cfg.max_ms).max(1.),
    );

    let poll_hz = state.poll_hz;
    let n = ((mt * poll_hz as f64 / 1000.).round() as i64).clamp(2, 512) as usize;

    let peak = uniform(
        random,
        cfg.peak_min.min(cfg.peak_max),
        cfg.peak_min.max(cfg.peak_max),
    );
    let mut weights = std::mem::take(&mut state.weights);
    render_weights(&mut weights, mt, peak, n);

    // A single progress parameter drives both axes, so x and y stay on one
    // path. Smoothing them independently gives an L-shaped trajectory that no
    // hand produces.
    let (ux, uy) = (target.0 / amp, target.1 / amp);
    let (px, py) = (-uy, ux);

    // The hand pivots at the wrist, so the path is an arc rather than a
    // segment, and which way it bows follows the direction of travel instead
    // of being a coin flip.
    let bias = cfg.curve_bias.clamp(-0.5, 0.5) * if target.0 >= 0. { 1. } else { -1. };
    let sign = if random.random::<f64>() < 0.5 + bias {
        1.
    } else {
        -1.
    };
    let bow = sign * uniform(random, cfg.bow * 0.3, cfg.bow) * amp;
    let kappa = uniform(
        random,
        cfg.kappa_min.min(cfg.kappa_max),
        cfg.kappa_min.max(cfg.kappa_max),
    )
    .max(0.1);
    let tremor = Tremor::new(&cfg, mt, random);

    let max_counts = cfg.max_counts.clamp(1, i32::MAX as i64);
    let mut steps = Vec::with_capacity(n);
    let (mut ex, mut ey) = (0i64, 0i64);
    let mut u = 0.;
    for k in 0..n {
        u += weights[k];
        let last = k + 1 == n;
        // Everything that perturbs the path is switched off for the last
        // report, and because reports are computed from position rather than
        // accumulated, that one report absorbs whatever the noise did earlier.
        // The endpoint is therefore exact no matter how loud the noise is.
        let (off, along) = if last {
            (0., 0.)
        } else {
            // Harris & Wolpert: the noise standard deviation rides the
            // instantaneous speed, so it is loudest mid-flick and vanishes
            // where the hand is slow. It is anisotropic, because people vary
            // more along the direction of travel than across it.
            let m = weights[k] * amp;
            let across = 0.6 * cfg.noise * m * gauss(random);
            let forward = cfg.noise * m * gauss(random);
            (
                bow * (PI * u.powf(kappa)).sin() + tremor.at(u) + across,
                forward,
            )
        };
        let wx = u * target.0 + off * px + along * ux;
        let wy = u * target.1 + off * py + along * uy;
        // Position-based rather than incremental. This is what makes the
        // endpoint exact, makes the per-report ceiling defer its excess
        // instead of dropping it, and lets a missed deadline coalesce into the
        // next report without losing counts.
        let sx = ((wx - ex as f64).round() as i64).clamp(-max_counts, max_counts);
        let sy = ((wy - ey as f64).round() as i64).clamp(-max_counts, max_counts);
        ex += sx;
        ey += sy;
        steps.push(MoveStep {
            dx: sx as i32,
            dy: sy as i32,
            at: Duration::from_nanos(((k as u64 + 1) * 1_000_000_000) / poll_hz as u64),
        });
    }

    state.carry = (target.0 - ex as f64, target.1 - ey as f64);
    state.weights = weights;
    let duration = steps.last().map(|s| s.at).unwrap_or_default();
    // The staleness test has to measure from where the flick *ends*. From the
    // start, a long flick would look stale on the very next frame and re-arm
    // the reaction window on every shot.
    state.last_flick = Some(now + duration);
    let clamp32 = |v: i64| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    Some(Flick {
        steps,
        total: (clamp32(ex), clamp32(ey)),
        duration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn cfg() -> SmoothConfig {
        SmoothConfig::default()
    }

    /// One flick from a fresh hand, so the sub-count carry starts at zero.
    fn plan(cfg: SmoothConfig, delta: (f64, f64), d_px: f32, w_px: f32, seed: u64) -> Flick {
        let mut state = SmoothAim::new(cfg, 2_000_000);
        let mut random = StdRng::seed_from_u64(seed);
        plan_flick(&mut state, delta, d_px, w_px, &mut random).expect("a non-zero delta flicks")
    }

    /// Reports are derived from the planned position rather than accumulated,
    /// so the counts on the wire have to add up to the plan exactly however
    /// loud the noise was along the way.
    #[test]
    fn a_flick_emits_exactly_the_counts_it_planned() {
        for seed in 0..200 {
            let f = plan(cfg(), (180., -76.), 195., 24., seed);
            let sx: i32 = f.steps.iter().map(|s| s.dx).sum();
            let sy: i32 = f.steps.iter().map(|s| s.dy).sum();
            assert_eq!((sx, sy), f.total, "seed {seed}");
        }
    }

    /// Truncating would mean the mouse never moves at all below one count, and
    /// would bias every move short, always toward the crosshair.
    #[test]
    fn fractional_counts_are_carried_between_calls_instead_of_truncated() {
        let mut c = cfg();
        c.gain = 1.;
        c.gain_sd = 0.;
        c.overshoot_p = 0.;
        c.gap_ms = u64::MAX; // one continuous engagement, never stale
        let mut state = SmoothAim::new(c, 2_000_000);
        let mut random = StdRng::seed_from_u64(7);
        let mut total = 0i64;
        for _ in 0..50 {
            let f = plan_flick(&mut state, (0.4, 0.), 4., 24., &mut random).expect("flicks");
            total += f.total.0 as i64;
        }
        assert_eq!(total, 20, "fifty requests of 0.4 counts are twenty counts");
    }

    #[test]
    fn the_leftover_fraction_never_reaches_a_whole_count() {
        let mut c = cfg();
        c.gap_ms = u64::MAX;
        let mut state = SmoothAim::new(c, 2_000_000);
        let mut random = StdRng::seed_from_u64(11);
        for step in 0..200 {
            let d = (step % 17) as f64 * 3.5 - 20.;
            if plan_flick(
                &mut state,
                (d, -d / 2.),
                d.abs() as f32 * 10.,
                24.,
                &mut random,
            )
            .is_some()
            {
                let (cx, cy) = state.carry();
                assert!(
                    cx.abs() < 0.5 + 1e-9 && cy.abs() < 0.5 + 1e-9,
                    "step {step}: {cx}, {cy}"
                );
            }
        }
    }

    /// The `steps` heuristic this replaces is linear in distance, which draws a
    /// straight band on a duration-versus-distance plot. Human movement time
    /// grows with the logarithm of the distance instead.
    #[test]
    fn duration_grows_with_the_log_of_the_distance_not_with_the_distance() {
        let mut c = cfg();
        c.mt_jitter = 0.;
        let short = plan(c, (10., 0.), 10., 24., 3).duration;
        let long = plan(c, (1000., 0.), 1000., 24., 3).duration;
        assert!(long > short, "{long:?} vs {short:?}");
        assert!(
            long.as_secs_f64() < 5. * short.as_secs_f64(),
            "a hundred times the distance must not be anything like a hundred \
             times the time: {long:?} vs {short:?}"
        );
        // Equal *ratios* of distance cost equal increments of time.
        let a = fitts_duration_ms(100., 24., c.fitts_a_ms, c.fitts_b_ms);
        let b = fitts_duration_ms(200., 24., c.fitts_a_ms, c.fitts_b_ms);
        let d = fitts_duration_ms(400., 24., c.fitts_a_ms, c.fitts_b_ms);
        assert!(((b - a) - (d - b)).abs() < 3., "{a} {b} {d}");
    }

    /// A symmetric bell peaking at the midpoint is the minimum-jerk model, which
    /// is a theoretical ideal rather than anything a hand measures like.
    #[test]
    fn the_speed_profile_peaks_between_thirty_and_forty_five_percent_of_the_duration() {
        let mut w = Vec::new();
        for f in [0.30, 0.35, 0.40, 0.45] {
            render_weights(&mut w, 200., f, 400);
            let peak = (0..w.len())
                .max_by(|&a, &b| w[a].partial_cmp(&w[b]).expect("finite weights"))
                .expect("a non-empty profile");
            let frac = (peak as f64 + 0.5) / w.len() as f64;
            assert!(
                (frac - f).abs() < 0.02,
                "asked for {f}, peaked at {frac:.3}"
            );
        }
    }

    #[test]
    fn the_deceleration_phase_is_longer_than_the_acceleration_phase() {
        let mut w = Vec::new();
        render_weights(&mut w, 200., 0.35, 400);
        let mut acc = 0.;
        let half = w
            .iter()
            .position(|x| {
                acc += x;
                acc >= 0.5
            })
            .expect("the profile sums to one");
        let frac = half as f64 / w.len() as f64;
        assert!(frac < 0.45, "half the distance was covered at {frac:.3}");
    }

    #[test]
    fn the_lognormal_sigma_solves_the_peak_fraction_it_was_asked_for() {
        for f in [0.25, 0.30, 0.35, 0.45, 0.55] {
            for mt in [50., 120., 320.] {
                let (mu, sigma) = lognormal_shape(mt, f);
                let mode = (mu - sigma * sigma).exp();
                assert!((mode - f * mt).abs() < 1e-6 * mt, "f={f} mt={mt}: {mode}");
                let p99 = (mu + Z99 * sigma).exp();
                assert!((p99 - mt).abs() < 1e-6 * mt, "f={f} mt={mt}: {p99}");
            }
        }
    }

    /// The wrist pivots, so the path is an arc. It still has to start and stop
    /// on the target: a bow that survives to the endpoint is an aiming error.
    #[test]
    fn the_path_bows_off_the_straight_line_but_starts_and_ends_on_it() {
        let mut c = cfg();
        c.noise = 0.;
        c.tremor = 0.;
        for seed in 0..50 {
            let f = plan(c, (600., 0.), 600., 24., seed);
            let (tx, ty) = (f.total.0 as f64, f.total.1 as f64);
            let amp = tx.hypot(ty);
            let (nx, ny) = (-ty / amp, tx / amp);
            let (mut x, mut y) = (0., 0.);
            let mut max = 0f64;
            let last = f.steps.len() - 1;
            for (i, s) in f.steps.iter().enumerate() {
                x += s.dx as f64;
                y += s.dy as f64;
                let perp = (x * nx + y * ny).abs();
                if i == last {
                    assert!(perp < 1., "seed {seed}: ended {perp} off the line");
                } else {
                    max = max.max(perp);
                }
            }
            assert!(
                max > 0.004 * amp && max < 0.06 * amp,
                "seed {seed}: bowed {max} on an amplitude of {amp}"
            );
        }
    }

    /// Smoothing the axes independently gives an L-shaped or
    /// diagonal-then-straight path; a hand follows one trajectory.
    #[test]
    fn x_and_y_stay_on_one_path_instead_of_being_smoothed_apart() {
        let mut c = cfg();
        c.noise = 0.;
        c.tremor = 0.;
        c.bow = 0.;
        let f = plan(c, (300., 400.), 500., 24., 5);
        let (mut x, mut y) = (0f64, 0f64);
        for (i, s) in f.steps.iter().enumerate() {
            x += s.dx as f64;
            y += s.dy as f64;
            // Skip the first counts, where rounding dominates the ratio.
            if x.hypot(y) < 40. {
                continue;
            }
            assert!((y / x - 4. / 3.).abs() < 0.1, "step {i}: {x},{y}");
        }
    }

    /// Overshooting forces a reversal, which costs more than one more nudge
    /// forward, so a ballistic reach is biased short. The remainder is what the
    /// next frame turns into a visually guided correction.
    #[test]
    fn the_ballistic_phase_deliberately_falls_short_of_the_target() {
        let mut c = cfg();
        c.overshoot_p = 0.;
        let n = 400;
        let mut sum = 0.;
        for seed in 0..n {
            let g = plan(c, (500., 0.), 500., 24., seed).total.0 as f64 / 500.;
            assert!(g > 0.70 && g < 1.06, "seed {seed}: gain {g:.3}");
            sum += g;
        }
        let mean = sum / n as f64;
        assert!((mean - c.gain).abs() < 0.02, "mean gain {mean:.4}");
    }

    /// Noise perturbs the path, never where the hand lands: a hand that scatters
    /// its endpoint by the noise amplitude would miss.
    #[test]
    fn tremor_and_noise_never_move_the_endpoint() {
        let clean = SmoothConfig {
            noise: 0.,
            tremor: 0.,
            ..cfg()
        };
        let noisy = SmoothConfig {
            noise: 0.3,
            tremor: 2.,
            ..cfg()
        };
        for seed in 0..50 {
            let a = plan(clean, (250., -90.), 265., 24., seed);
            let b = plan(noisy, (250., -90.), 265., 24., seed);
            assert_eq!(a.total, b.total, "seed {seed}");
        }
    }

    /// `panic = "abort"` in release means a panic here takes the whole process
    /// with it, and a NaN delta is reachable from a misconfigured GAME_SENS.
    #[test]
    fn a_zero_delta_asks_for_no_movement_at_all() {
        let mut state = SmoothAim::new(cfg(), 2_000_000);
        let mut random = StdRng::seed_from_u64(1);
        for delta in [
            (0., 0.),
            (f64::NAN, 1.),
            (f64::INFINITY, 0.),
            (1., f64::NEG_INFINITY),
        ] {
            assert!(
                plan_flick(&mut state, delta, 100., 24., &mut random).is_none(),
                "{delta:?}"
            );
        }
        assert!(plan_flick(&mut state, (10., 0.), f32::NAN, 24., &mut random).is_none());
    }

    /// Commit 996fcf9 had to go back and fix a panic from an empty
    /// `random_range`, so every configurable pair is tried collapsed and
    /// inverted here.
    #[test]
    fn a_degenerate_config_still_plans_without_panicking() {
        let mut random = StdRng::seed_from_u64(2);
        let collapsed = SmoothConfig {
            bow: 0.,
            mt_jitter: 0.,
            gain_sd: 0.,
            min_ms: 120.,
            max_ms: 120.,
            peak_min: 0.4,
            peak_max: 0.4,
            kappa_min: 1.,
            kappa_max: 1.,
            react_min_ms: 150,
            react_max_ms: 150,
            noise: 0.,
            tremor: 0.,
            ..cfg()
        };
        for amp in [1e-9, 0.4, 1., 1e6] {
            let mut state = SmoothAim::new(collapsed, 115_200);
            let t = Instant::now();
            let delta = (amp, amp);
            assert!(
                plan_flick_at(&mut state, delta, amp as f32, 24., &mut random, t).is_none(),
                "amp {amp} moved before the reaction window elapsed"
            );
            let late = t + Duration::from_millis(200);
            assert!(
                plan_flick_at(&mut state, delta, amp as f32, 24., &mut random, late).is_some(),
                "amp {amp}"
            );
        }

        let inverted = SmoothConfig {
            min_ms: 300.,
            max_ms: 40.,
            peak_min: 0.5,
            peak_max: 0.2,
            kappa_min: 1.3,
            kappa_max: 0.8,
            react_min_ms: 200,
            react_max_ms: 50,
            curve_bias: 9.,
            max_counts: 0,
            ..cfg()
        };
        let mut state = SmoothAim::new(inverted, 115_200);
        let t = Instant::now();
        let _ = plan_flick_at(&mut state, (50., 50.), 70., 24., &mut random, t);
        let late = t + Duration::from_millis(500);
        assert!(plan_flick_at(&mut state, (50., 50.), 70., 24., &mut random, late).is_some());
    }

    #[test]
    fn no_single_report_exceeds_the_count_cap() {
        let mut c = cfg();
        c.max_counts = 5;
        let f = plan(c, (400., -300.), 500., 24., 9);
        for s in &f.steps {
            assert!(s.dx.abs() <= 5 && s.dy.abs() <= 5, "{s:?}");
        }
    }

    /// Clamping has to defer the excess rather than throw it away, or a fast
    /// flick would silently fall short by whatever the ceiling swallowed.
    #[test]
    fn a_capped_report_defers_the_excess_instead_of_dropping_it() {
        let c = SmoothConfig {
            max_counts: 5,
            overshoot_p: 0.,
            gain_sd: 0.,
            ..cfg()
        };
        let mut state = SmoothAim::new(c, 2_000_000);
        let mut random = StdRng::seed_from_u64(4);
        let f = plan_flick(&mut state, (400., -300.), 500., 24., &mut random).expect("flicks");
        let (cx, cy) = state.carry();
        assert!(
            (f.total.0 as f64 + cx - 400. * c.gain).abs() < 1e-9,
            "{} + {cx}",
            f.total.0
        );
        assert!(
            (f.total.1 as f64 + cy + 300. * c.gain).abs() < 1e-9,
            "{} + {cy}",
            f.total.1
        );
    }

    /// A real mouse reports on a fixed cadence; what varies is the size of each
    /// report, not the spacing between them.
    #[test]
    fn report_deadlines_match_the_poll_rate_and_only_move_forward() {
        for baud in [115_200u32, 2_000_000] {
            let mut state = SmoothAim::new(cfg(), baud);
            let hz = state.poll_hz();
            let mut random = StdRng::seed_from_u64(6);
            let f = plan_flick(&mut state, (240., 80.), 253., 24., &mut random).expect("flicks");
            let period = Duration::from_nanos(1_000_000_000 / hz as u64);
            let mut prev = Duration::ZERO;
            for s in &f.steps {
                assert_eq!(s.at - prev, period, "{baud}: at {:?}", s.at);
                prev = s.at;
            }
            assert_eq!(f.duration, prev, "{baud}");
        }
    }

    #[test]
    fn a_fresh_target_waits_out_a_reaction_delay_before_the_first_flick() {
        let c = SmoothConfig {
            react_min_ms: 150,
            react_max_ms: 260,
            ..cfg()
        };
        let mut state = SmoothAim::new(c, 2_000_000);
        let mut random = StdRng::seed_from_u64(8);
        let t = Instant::now();
        let d = (200., 0.);
        assert!(plan_flick_at(&mut state, d, 200., 24., &mut random, t).is_none());
        let early = t + Duration::from_millis(149);
        assert!(plan_flick_at(&mut state, d, 200., 24., &mut random, early).is_none());
        let after = t + Duration::from_millis(300);
        assert!(plan_flick_at(&mut state, d, 200., 24., &mut random, after).is_some());
    }

    /// The leftover fraction describes a move toward a target that is gone, so
    /// spending it on the next one would pull the first flick off course.
    #[test]
    fn a_long_gap_throws_away_the_leftover_from_the_old_target() {
        let mut state = SmoothAim::new(cfg(), 2_000_000);
        let mut random = StdRng::seed_from_u64(10);
        let t = Instant::now();
        plan_flick_at(&mut state, (0.4, 0.4), 4., 24., &mut random, t).expect("flicks");
        assert_ne!(state.carry(), (0., 0.));
        // A zero delta still runs the staleness check before it bails, so the
        // carry is observable without planning anything.
        let late = t + Duration::from_millis(cfg().gap_ms + 500);
        assert!(plan_flick_at(&mut state, (0., 0.), 0., 24., &mut random, late).is_none());
        assert_eq!(state.carry(), (0., 0.));
    }

    /// The command only leaves this process while the ESP trigger is held, so
    /// the operator's own finger already supplies the reaction time. Simulating
    /// it again would charge for the same latency twice.
    #[test]
    fn the_reaction_window_is_a_no_op_when_it_is_switched_off() {
        let c = cfg();
        assert_eq!((c.react_min_ms, c.react_max_ms), (0, 0));
        let mut state = SmoothAim::new(c, 2_000_000);
        let mut random = StdRng::seed_from_u64(12);
        let t = Instant::now();
        assert!(plan_flick_at(&mut state, (200., 0.), 200., 24., &mut random, t).is_some());
    }

    /// Rounding each report toward zero would bias every flick short. Rounding
    /// to nearest and carrying the remainder has to leave no drift at all.
    #[test]
    fn over_many_flicks_the_rounding_residual_has_no_systematic_bias() {
        let mut c = cfg();
        c.gap_ms = u64::MAX;
        let mut state = SmoothAim::new(c, 2_000_000);
        let mut random = StdRng::seed_from_u64(21);
        let (mut sx, mut sy) = (0f64, 0f64);
        let n = 2000;
        for k in 0..n {
            let delta = ((k % 37) as f64 * 0.27 - 5., (k % 23) as f64 * 0.41 - 4.);
            if plan_flick(&mut state, delta, 40., 24., &mut random).is_some() {
                let (cx, cy) = state.carry();
                sx += cx;
                sy += cy;
            }
        }
        let (mx, my) = (sx / n as f64, sy / n as f64);
        assert!(
            mx.abs() < 0.05 && my.abs() < 0.05,
            "the residual drifts: {mx:.4}, {my:.4}"
        );
    }

    #[test]
    fn over_many_flicks_the_time_to_peak_speed_is_never_at_the_midpoint() {
        let c = cfg();
        let mut w = Vec::new();
        let mut random = StdRng::seed_from_u64(22);
        for _ in 0..2000 {
            let peak = uniform(&mut random, c.peak_min, c.peak_max);
            let mt = uniform(&mut random, c.min_ms, c.max_ms);
            render_weights(&mut w, mt, peak, 256);
            let arg = (0..w.len())
                .max_by(|&a, &b| w[a].partial_cmp(&w[b]).expect("finite weights"))
                .expect("a non-empty profile");
            let frac = (arg as f64 + 0.5) / w.len() as f64;
            assert!(!(0.48..=0.52).contains(&frac), "peaked at {frac:.3}");
            assert!(frac > 0.25 && frac < 0.50, "peaked at {frac:.3}");
        }
    }

    #[test]
    fn over_many_flicks_about_one_in_seven_overshoots_the_target() {
        let n = 2000;
        let mut over = 0;
        for seed in 0..n {
            if plan(cfg(), (500., 0.), 500., 24., seed).total.0 > 500 {
                over += 1;
            }
        }
        let rate = over as f64 / n as f64;
        assert!((rate - 0.15).abs() < 0.04, "overshoot rate {rate:.3}");
    }

    /// Repetition is the machine tell the randomisation exists to remove.
    #[test]
    fn over_many_flicks_no_two_are_byte_identical() {
        use std::collections::HashSet;
        use std::hash::{DefaultHasher, Hash, Hasher};
        let mut seen = HashSet::new();
        for seed in 0..2000u64 {
            let f = plan(cfg(), (350., 140.), 377., 24., seed);
            let mut h = DefaultHasher::new();
            f.steps.len().hash(&mut h);
            for s in &f.steps {
                s.dx.hash(&mut h);
                s.dy.hash(&mut h);
            }
            assert!(
                seen.insert(h.finish()),
                "seed {seed} repeated an earlier flick"
            );
        }
    }
}
