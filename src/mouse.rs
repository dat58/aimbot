use anyhow::{Result, bail};
use rand::prelude::*;
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

    #[inline(always)]
    pub(crate) fn find_bezier(&self, dx: f64, dy: f64, random: &mut ThreadRng) -> (i32, i32, i32) {
        let pixel = (dx * dx + dy * dy).sqrt();
        let lower = (pixel * 0.2) as i32;
        let upper = (pixel * 0.55) as i32 + 1;
        let steps = random.random_range(lower..=upper);
        let ref_x = random.random_range(4..17);
        let ref_y = random.random_range(4..17);
        (steps, ref_x, ref_y)
    }

    pub fn listen_button_presses(self: Arc<Self>) {
        let mut last_value = 0;
        let mut buf = [0; 8];
        loop {
            let bytes_read = {
                let mut serial = self.serial.lock().expect("Could not acquire serial lock");
                serial.read(&mut buf)
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
            sleep(Duration::from_millis(2));
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

    pub fn batch(&self) -> BatchCommands {
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
