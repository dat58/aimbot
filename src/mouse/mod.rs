//! The mouse: a device that can be told to move and click, and a worker that
//! does the telling.
//!
//! Two independent axes, deliberately kept apart. [`MouseDevice`] is *how the
//! command leaves this process* — a link, a protocol, a piece of hardware.
//! [`mover::Mover`] is *what shape the movement has*. Adding a device means a
//! file here and an arm in [`connect`]; adding a movement means a file in
//! [`mover`] and an arm in [`mover::build`]. Neither touches the other.

pub mod makcu;
#[cfg(feature = "medius")]
pub mod medius;
pub mod mover;

use crate::config::{DeviceKind, MouseConfig};
use anyhow::Result;
use crossbeam::queue::ArrayQueue;
use makcu::Makcu;
#[cfg(feature = "medius")]
use medius::Medius;
use mover::constant::{CLICK_HOLD_MS, IDLE_POLL};
use mover::{ClickPlan, MoveRequest, Mover};
use rand::prelude::*;
use std::{
    sync::Arc,
    thread::sleep,
    time::{Duration, Instant},
};

/// The five buttons a mouse reports, in the bit order MAKCU's status byte uses
/// — which is also the order its `pressed` array is indexed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Button {
    Left = 0,
    Right = 1,
    Middle = 2,
    Side4 = 3,
    Side5 = 4,
}

/// What a device can do beyond moving and clicking.
///
/// A mover asks this rather than trying a command and handling the refusal,
/// because an unknown command is answered with silence rather than an error:
/// a refusal that cannot be observed is not one you can plan around.
#[derive(Debug, Clone, Copy, Default)]
pub struct Caps {
    /// Puts the counts in the device's next report untouched, with no
    /// interpolation of its own on top.
    pub move_immediate: bool,
    /// Walks a whole curve itself, from one command.
    pub curve: bool,
    /// Reports the physical mouse's own buttons back.
    pub buttons: bool,
    /// Can freeze the physical mouse on one axis.
    pub lock_axis: bool,
}

/// Something that holds the mouse.
pub trait MouseDevice: Send + Sync {
    /// The `device = "..."` name this backend is loaded under.
    fn name(&self) -> &'static str;

    fn caps(&self) -> Caps;

    /// One relative move, in mouse counts.
    fn move_to(&self, dx: i32, dy: i32) -> Result<()>;

    /// Press or release without timing it; what [`Self::click`] is built from.
    fn press(&self, button: Button, down: bool) -> Result<()>;

    /// Hold `button` down for `hold`, then release it.
    fn click(&self, button: Button, hold: Duration) -> Result<()>;

    /// Freeze the physical mouse on the X axis.
    fn lock_mx(&self, locked: bool) -> Result<()>;

    /// Freeze the physical mouse on the Y axis.
    fn lock_my(&self, locked: bool) -> Result<()>;

    /// Is the operator physically holding `button`?
    ///
    /// Always `false` unless the device has a return channel
    /// ([`Caps::buttons`]) and [`Self::listen`] is pumping it.
    fn is_pressing(&self, _button: Button) -> bool {
        false
    }

    /// Pump the device's return channel, forever. A no-op where there is none.
    fn listen(self: Arc<Self>) {}

    /// Like [`Self::move_to`], but skipping any interpolation the device would
    /// otherwise do. Only worth asking for when [`Caps::move_immediate`].
    fn move_immediate(&self, dx: i32, dy: i32) -> Result<()> {
        self.move_to(dx, dy)
    }

    /// Hand a whole curve to the device. Only called when [`Caps::curve`]; the
    /// fallback is the straight move, which at least arrives.
    fn move_curve(&self, dx: i32, dy: i32, _steps: i32, _ctrl: (i32, i32)) -> Result<()> {
        self.move_to(dx, dy)
    }
}

/// Load the device the config names.
pub fn connect(cfg: &MouseConfig) -> Result<Arc<dyn MouseDevice>> {
    let device: Arc<dyn MouseDevice> = match cfg.device {
        DeviceKind::Makcu => Arc::new(Makcu::new(&cfg.port, cfg.baud)?),
        #[cfg(feature = "medius")]
        DeviceKind::Medius => Arc::new(Medius::new(cfg)?),
    };
    tracing::info!("[Mouse] {} on {}", device.name(), cfg.port);
    Ok(device)
}

/// Run `on_hold` every time `button` is held for `hold`, forever.
///
/// Reads the button rather than being pushed at, so it costs nothing on a
/// device that cannot see the physical mouse: [`MouseDevice::is_pressing`]
/// answers `false` and this parks.
pub fn on_button_hold(
    mouse: Arc<dyn MouseDevice>,
    button: Button,
    hold: Duration,
    interval: Duration,
    on_hold: impl Fn(),
) {
    loop {
        if mouse.is_pressing(button) {
            let time = Instant::now();
            loop {
                sleep(interval);
                if mouse.is_pressing(button) {
                    if time.elapsed() >= hold {
                        on_hold();
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

/// Drain `queue` and drive the mouse, forever.
///
/// The aim loop used to play the movement itself, which blocks for its length
/// and, with the auto click on, for the click behind it. A frame arriving
/// mid-flick therefore went unlooked-at until the mouse had finished — up to a
/// couple of hundred milliseconds of not detecting anything.
///
/// Moving off-thread costs no ordering. Every command still leaves from this
/// one thread in the order it pops them, and a request's click still follows
/// its own move because one iteration does both, in sequence. What it does
/// change is that the queue holds a single request which the aim loop
/// overwrites, so the worker always picks up the newest aim rather than
/// working through a backlog of places the target used to be.
pub fn handle_mouse(
    mouse: Arc<dyn MouseDevice>,
    queue: Arc<ArrayQueue<MoveRequest>>,
    mut movers: Vec<Box<dyn Mover>>,
    keep_going: impl Fn() -> bool,
) {
    let mut random = rand::rng();
    loop {
        let Some(request) = queue.pop() else {
            sleep(IDLE_POLL);
            continue;
        };
        let Some(profile) = play_request(&*mouse, &mut movers, &mut random, &keep_going, request)
        else {
            continue;
        };

        // Only a mover that models one asks for a click; the rest answer
        // `None` and the worker goes straight back to tracking.
        let until = match movers[profile].after_move(&mut random) {
            ClickPlan::None => continue,
            // Hold still: whatever the movement got wrong is what the shot
            // carries.
            ClickPlan::Miss(wait) => {
                tracing::debug!(
                    "[{}] auto click after {wait:?}, holding still (miss)",
                    movers[profile].name()
                );
                sleep(wait);
                None
            }
            // Keep taking aim: the corrections still arriving get played inside
            // the wait, so the crosshair is on target by the time it fires.
            ClickPlan::Settle(wait) => {
                tracing::debug!(
                    "[{}] auto click after {wait:?}, still tracking (hit)",
                    movers[profile].name()
                );
                Some(Instant::now() + wait)
            }
        };
        if let Some(deadline) = until {
            while Instant::now() < deadline {
                match queue.pop() {
                    Some(request) => {
                        play_request(&*mouse, &mut movers, &mut random, &keep_going, request);
                    }
                    None => sleep(IDLE_POLL),
                }
            }
        }

        // Re-read the trigger: the wait is long enough to be let go inside, and
        // firing after that is a shot nobody asked for.
        if !keep_going() {
            continue;
        }
        // Stamped at the press, not after the release: the rate limit is a gap
        // between clicks and the click holds the button for tens of
        // milliseconds, which would otherwise be added to every gap.
        let pressed_at = Instant::now();
        let hold = Duration::from_millis(random.random_range(CLICK_HOLD_MS.0..=CLICK_HOLD_MS.1));
        match mouse.click(Button::Left, hold) {
            Ok(()) => movers[profile].clicked(pressed_at),
            Err(e) => tracing::error!("[Mouse] {}", e),
        }
    }
}

/// Play one request. A failed command is one lost aim, not a reason to stop
/// aiming: the next frame plans afresh from wherever the crosshair actually is.
fn play_request<F: Fn() -> bool>(
    mouse: &dyn MouseDevice,
    movers: &mut [Box<dyn Mover>],
    random: &mut rand::rngs::ThreadRng,
    keep_going: &F,
    request: MoveRequest,
) -> Option<usize> {
    let profile = request.profile;
    let Some(mover) = movers.get_mut(profile) else {
        tracing::error!("[Mouse] request names profile {profile}, which is not configured");
        return None;
    };
    if let Err(e) = mover.apply(mouse, request.aim, random, keep_going) {
        tracing::error!("[Mouse] {}", e);
    }
    Some(profile)
}
