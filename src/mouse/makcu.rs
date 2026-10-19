use super::{Button, Caps, MouseDevice};
use anyhow::{Result, bail};
use serialport::{self, SerialPort, available_ports};
use std::io::Write;
use std::{
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

pub struct Makcu {
    serial: Mutex<Box<dyn SerialPort>>,
    pressed: [AtomicBool; 5],
    /// Whether this firmware answers `km.move_now`. Probed once at connect,
    /// because an unknown command is answered with silence rather than an
    /// error: asking for `move_now` on firmware without it would swallow every
    /// report and stop the mouse dead with nothing in the log to say why.
    move_now: AtomicBool,
}

impl Makcu {
    /// The `device = \"...\"` name this backend is loaded under.
    pub const NAME: &'static str = "makcu";

    pub fn new(port: &str, baud: u32) -> Result<Self> {
        tracing::debug!("All available serial port: {:?}", available_ports());
        if !ALLOWED_BAUD_RATE.contains(&baud) {
            bail!("Baud rate out of range, allowed: {:?}", ALLOWED_BAUD_RATE);
        }
        let mut serial = serialport::new(port, baud)
            .timeout(Duration::from_millis(300))
            .open()?;
        let mut serial = match Self::check_km_version_ok(&mut serial) {
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
        let move_now = Self::probe_move_now(&mut serial);
        if move_now {
            tracing::info!("Firmware supports km.move_now");
        } else {
            tracing::info!(
                "Firmware has no km.move_now; MOVE_SMOOTH_MOVE_NOW falls back to km.move"
            );
        }
        Ok(Self {
            serial: Mutex::new(serial),
            pressed: Default::default(),
            move_now: AtomicBool::new(move_now),
        })
    }

    /// Does this firmware know `km.move_now`?
    ///
    /// A zero-count move is a no-op whichever way it goes, so the probe cannot
    /// disturb the pointer. A known command is echoed back; an unknown one gets
    /// no reply at all, which is what makes the distinction observable.
    fn probe_move_now(serial: &mut Box<dyn SerialPort>) -> bool {
        let _ = serial.clear(serialport::ClearBuffer::Input);
        if serial.write_all(b"km.move_now(0,0)\r\n").is_err() {
            return false;
        }
        // Button reports may already be streaming, so scan for the echo rather
        // than expecting it to be the whole of the reply.
        let deadline = Instant::now() + Duration::from_millis(250);
        let mut seen = String::new();
        let mut buffer = [0u8; 128];
        while Instant::now() < deadline {
            match serial.read(&mut buffer) {
                Ok(0) => {}
                Ok(n) => {
                    seen.push_str(&String::from_utf8_lossy(&buffer[..n]));
                    if seen.contains("move_now") {
                        return true;
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => {}
                Err(_) => return false,
            }
        }
        false
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
}

/// The `km.*` verb that names `button`.
///
/// Only `left` and `right` have ever been sent by this driver; the other three
/// follow MAKCU's documented naming and are unverified against firmware. An
/// unknown command is answered with silence rather than an error, so a wrong
/// name here would be a button that quietly never presses — check the three on
/// hardware before relying on them.
fn verb(button: Button) -> &'static str {
    match button {
        Button::Left => "left",
        Button::Right => "right",
        Button::Middle => "middle",
        Button::Side4 => "ms1",
        Button::Side5 => "ms2",
    }
}

impl MouseDevice for Makcu {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn caps(&self) -> Caps {
        Caps {
            // Probed once at connect: an unknown command is answered with
            // silence rather than an error, so asking without probing would
            // swallow every report and stop the mouse dead.
            move_immediate: self.move_now.load(Ordering::Acquire),
            curve: true,
            buttons: true,
            lock_axis: true,
        }
    }

    fn move_to(&self, dx: i32, dy: i32) -> Result<()> {
        self.cmd(format!("km.move({dx},{dy})").as_str())
    }

    /// `km.move` is interpolated by V4.028+ firmware over 4-42 ms, which smears
    /// a profile whose every report is already on its own deadline.
    /// `km.move_now` puts the counts in the next report untouched.
    fn move_immediate(&self, dx: i32, dy: i32) -> Result<()> {
        self.cmd(format!("km.move_now({dx},{dy})").as_str())
    }

    fn move_curve(&self, dx: i32, dy: i32, steps: i32, ctrl: (i32, i32)) -> Result<()> {
        let (ref_x, ref_y) = ctrl;
        self.cmd(format!("km.move({dx},{dy},{steps},{ref_x},{ref_y})").as_str())
    }

    fn press(&self, button: Button, down: bool) -> Result<()> {
        let state = u8::from(down);
        self.cmd(format!("km.{}({state})", verb(button)).as_str())
    }

    /// Press a button, hold it long enough to be seen, release it.
    ///
    /// `km.left(1)` immediately followed by `km.left(0)` — which is what this
    /// used to send, both in one write — does not click. The firmware emits HID
    /// reports on a fixed cadence, so a press and release that both land inside
    /// one report interval never appear on the wire at all; at 4 Mbaud the two
    /// lines are tens of microseconds apart.
    ///
    /// MAKCU's own `km.click(button)` would time the hold in firmware, but this
    /// firmware does not have it: it answers `km.click` with the same silence
    /// it gives any unknown command, so the hold is timed by the caller.
    fn click(&self, button: Button, hold: Duration) -> Result<()> {
        self.press(button, true)?;
        sleep(hold);
        self.press(button, false)
    }

    /// Lock the physical mouse on the X axis.
    fn lock_mx(&self, locked: bool) -> Result<()> {
        self.cmd(format!("km.lock_mx({})", u8::from(locked)).as_str())
    }

    /// Lock the physical mouse on the Y axis.
    fn lock_my(&self, locked: bool) -> Result<()> {
        self.cmd(format!("km.lock_my({})", u8::from(locked)).as_str())
    }

    fn is_pressing(&self, button: Button) -> bool {
        self.pressed[button as usize].load(Ordering::Acquire)
    }

    fn listen(self: Arc<Self>) {
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
}
