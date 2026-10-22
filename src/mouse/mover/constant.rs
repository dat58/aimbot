//! Numbers the movement model is built out of, as opposed to the numbers a
//! profile tunes.
//!
//! What lives here is what a *new* `move_*` would want too: report pacing, the
//! click hold, the physiological bands. Keeping them in one place means a
//! change applies to every mover instead of having to be hunted down in each.
//! Device-specific numbers — baud rates, the `km.*` wire format — belong to
//! the device, not here; see [`crate::mouse::makcu`].

use std::time::Duration;

/// `thread::sleep` on Linux overshoots by tens of microseconds, so below this
/// it is cheaper to fire a report a touch early than to oversleep it.
pub const SLEEP_FLOOR: Duration = Duration::from_micros(150);

/// How long [`crate::mouse::handle_mouse`] parks when the queue is empty. A
/// single command already costs ~0.6 ms over usbip, so a wake-up granularity
/// well inside that is not what limits this.
pub const IDLE_POLL: Duration = Duration::from_micros(200);

/// How long a click holds the button down, in milliseconds. The range MAKCU's
/// own `km.click` draws from by default, which is as good a reference as any
/// for what a click should look like on the wire.
pub const CLICK_HOLD_MS: (u64, u64) = (35, 75);

/// The standard normal's 99th percentile, used to pin the lognormal tail to
/// the end of the movement.
pub const Z99: f64 = 2.3263478740408408;

/// Reports a single stroke may be planned with.
///
/// Two is the floor because one report cannot express a velocity profile at
/// all; the ceiling is there so a pathological duration cannot queue thousands
/// of commands behind each other.
pub const REPORTS: (i64, i64) = (2, 512);

/// Physiological tremor band, in hertz. Real tremor is a band around 8-12 Hz
/// rather than a pure tone, which is why the frequency is drawn per stroke.
pub const TREMOR_HZ: (f64, f64) = (8., 12.);

/// The second tremor component, as a multiple of the first. Deliberately not a
/// whole number: two components that shared a period would beat visibly.
pub const TREMOR_HARMONIC: (f64, f64) = (1.4, 1.9);

/// How loud the second tremor component is next to the first.
pub const TREMOR_HARMONIC_GAIN: f64 = 0.5;

/// Harris & Wolpert noise is anisotropic: people vary less *across* the
/// direction of travel than along it, by about this factor.
pub const NOISE_ACROSS: f64 = 0.6;

/// How far a gaussian draw may stray, in standard deviations. A tail event
/// several sigma out is not a hand, it is an artefact.
pub const GAUSS_CLAMP: f64 = 3.;

/// Gain of an overshooting reach: mean, then standard deviation.
pub const OVERSHOOT_GAIN: (f64, f64) = (1.06, 0.04);

/// Bounds on an overshooting reach's gain.
pub const OVERSHOOT_CLAMP: (f64, f64) = (1., 1.25);

/// Bounds on an ordinary reach's gain. The ceiling sits just above one so a
/// generous draw can still land on the target rather than always short of it.
pub const GAIN_CLAMP: (f64, f64) = (0.05, 1.05);

/// A non-positive DPI would make every implied speed infinite; this is what
/// `mouse.dpi` itself defaults to, so falling back to it poisons nothing.
pub const FALLBACK_DPI: f64 = 1000.;

/// Bounds on the report cadence, in hertz. No real mouse reports faster than
/// the top of this, so a higher cadence would itself be a signature.
pub const POLL_HZ_CLAMP: (u32, u32) = (1, 8000);

/// Steps a firmware-side bezier is asked for, as a fraction of the amplitude
/// in pixels.
pub const BEZIER_STEPS_FRAC: (f64, f64) = (0.2, 0.55);

/// Range the bezier's control-point offsets are drawn from, inclusive.
pub const BEZIER_CTRL: (i32, i32) = (1, 5);
