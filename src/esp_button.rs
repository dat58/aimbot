use anyhow::Result;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

pub struct EspButton {
    serial: Box<dyn serialport::SerialPort>,
    is_pressed: Arc<AtomicBool>,
}

impl EspButton {
    pub fn new(port: &str, is_pressed: Arc<AtomicBool>) -> Result<Self> {
        let serial = match serialport::new(port, 115200)
            .timeout(Duration::from_millis(300))
            .open()
        {
            Ok(mut serial) => {
                let mut buf = [0u8; 1];
                loop {
                    match serial.read(&mut buf) {
                        Ok(_) => {
                            drop(serial);
                            thread::sleep(Duration::from_millis(300));
                            break;
                        }
                        _ => {}
                    }
                }
                serialport::new(port, 115200)
                    .timeout(Duration::from_millis(100))
                    .open()?
            }
            Err(e) => Err(e)?,
        };
        Ok(Self { serial, is_pressed })
    }

    pub fn is_pressed(&self) -> bool {
        self.is_pressed.load(Ordering::Acquire)
    }

    pub fn listen(&mut self) {
        let mut buf = [0u8; 8];
        let mut last_state = self.is_pressed();
        loop {
            match self.serial.read(&mut buf) {
                Ok(count) if count > 0 => {
                    for i in 0..count {
                        if buf[i] == 48 || buf[i] == 49 {
                            if buf[i] == 49 {
                                if last_state != true {
                                    self.is_pressed.store(true, Ordering::Release);
                                    last_state = true;
                                }
                            } else {
                                if last_state != false {
                                    self.is_pressed.store(false, Ordering::Release);
                                    last_state = false;
                                }
                            }
                        }
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => {
                    tracing::debug!("Read esp button timeout");
                }
                Err(e) => {
                    tracing::error!("Error while reading esp button state: {e}");
                    break;
                }
                _ => {}
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}
