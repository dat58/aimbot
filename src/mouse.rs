use anyhow::{Result, bail};
use rand::prelude::*;
use serialport::{self, SerialPort};
use std::{thread::sleep, time::Duration};

const BAUD_CHANGE_COMMAND: [u8; 9] = [0xDE, 0xAD, 0x05, 0x00, 0xA5, 0x00, 0x09, 0x3D, 0x00];
const VERIFY_COMMAND: &[u8] = b"km.version()\r\n";
const DEFAULT_BAUD_RATE: u32 = 115_200;
const ALLOWED_BAUD_RATE: [u32; 3] = [115_200, 2_000_000, 4_000_000];
const CRLF: &str = "\r\n";

pub struct MouseVirtual {
    serial: Box<dyn SerialPort>,
    pub random: ThreadRng,
}

impl MouseVirtual {
    pub fn new(port: &str, baud: u32) -> Result<Self> {
        if !ALLOWED_BAUD_RATE.contains(&baud) {
            bail!("Baud rate out of range, allowed: {:?}", ALLOWED_BAUD_RATE);
        }
        let mut serial = serialport::new(port, DEFAULT_BAUD_RATE)
            .timeout(Duration::from_millis(100))
            .open()?;
        if baud != DEFAULT_BAUD_RATE {
            serial.write_all(&BAUD_CHANGE_COMMAND)?;
            serial.flush()?;
            serial.set_baud_rate(baud)?;
            serial.clear(serialport::ClearBuffer::Input)?;
            serial.clear(serialport::ClearBuffer::Output)?;
            sleep(Duration::from_millis(100));
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
        }
        Ok(Self {
            serial,
            random: rand::rng(),
        })
    }

    #[inline(always)]
    fn cmd(&mut self, command: &str) -> Result<()> {
        Ok(self
            .serial
            .write_all(format!("{command}{CRLF}").as_bytes())?)
    }

    pub fn move_shift(&mut self, dx: i32, dy: i32) -> Result<()> {
        self.cmd(format!("km.move({dx},{dy})").as_str())
    }

    pub fn move_bezier(&mut self, dx: i32, dy: i32) -> Result<()> {
        let (steps, ref_x, ref_y) = self.find_bezier(dx, dy);
        self.cmd(format!("km.move({dx},{dy},{steps},{ref_x},{ref_y})").as_str())
    }

    #[inline(always)]
    pub(crate) fn find_bezier(&mut self, dx: i32, dy: i32) -> (i32, i32, i32) {
        let pixel = (dx * dx + dy * dy).isqrt();
        let steps = if pixel < 50 {
            self.random.random_range(1..=10)
        } else if pixel < 200 {
            self.random.random_range(5..=15)
        } else if pixel < 500 {
            self.random.random_range(10..=20)
        } else if pixel < 1200 {
            self.random.random_range(16..=28)
        } else {
            self.random.random_range(26..=40)
        };
        let ref_x = self.random.random_range(4..17);
        let ref_y = self.random.random_range(4..17);
        (steps, ref_x, ref_y)
    }

    /// Lock physical mouse on X-axis direction
    pub fn lock_mx(&mut self) -> Result<()> {
        self.cmd("km.lock_mx(1)")
    }

    /// Unlock physical mouse on X-axis direction
    pub fn unlock_mx(&mut self) -> Result<()> {
        self.cmd("km.lock_mx(0)")
    }

    /// Lock physical mouse on Y-axis direction
    pub fn lock_my(&mut self) -> Result<()> {
        self.cmd("km.lock_my(1)")
    }

    /// Unlock physical mouse on Y-axis direction
    pub fn unlock_my(&mut self) -> Result<()> {
        self.cmd("km.lock_my(0)")
    }

    pub fn batch(&mut self) -> BatchCommands {
        BatchCommands::new(self)
    }
}

pub struct BatchCommands<'a> {
    mouse: &'a mut MouseVirtual,
    buf: String,
}

impl<'a> BatchCommands<'a> {
    pub fn new(mouse: &'a mut MouseVirtual) -> Self {
        Self {
            mouse,
            buf: String::new(),
        }
    }

    pub fn move_shift(mut self, dx: i32, dy: i32) -> Self {
        self.buf
            .push_str(format!("km.move({dx},{dy}){CRLF}").as_str());
        self
    }

    pub fn move_bezier(mut self, dx: i32, dy: i32) -> Self {
        let (steps, ref_x, ref_y) = self.mouse.find_bezier(dx, dy);
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

    pub fn run(&mut self) -> Result<()> {
        self.mouse.cmd(self.buf.as_str())
    }
}
