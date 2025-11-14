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
    is_pressed_1: Arc<AtomicBool>,
    is_pressed_2: Arc<AtomicBool>,
}

impl EspButton {
    pub fn new(
        port: &str,
        is_pressed_1: Arc<AtomicBool>,
        is_pressed_2: Arc<AtomicBool>,
    ) -> Result<Self> {
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
        Ok(Self {
            serial,
            is_pressed_1,
            is_pressed_2,
        })
    }

    pub fn listen(&mut self) {
        let mut buf = [0u8; 8];
        let mut last_state_1 = self.is_pressed_1.load(Ordering::Acquire);
        let mut last_state_2 = self.is_pressed_2.load(Ordering::Acquire);
        loop {
            match self.serial.read(&mut buf) {
                Ok(count) if count > 0 => {
                    for i in 0..count {
                        if buf[i] >= 48 && buf[i] <= 51 {
                            if buf[i] == 48 {
                                if last_state_1 != false {
                                    self.is_pressed_1.store(false, Ordering::Release);
                                    last_state_1 = false;
                                }
                            } else if buf[i] == 49 {
                                if last_state_1 != true {
                                    self.is_pressed_1.store(true, Ordering::Release);
                                    last_state_1 = true;
                                }
                            } else if buf[i] == 50 {
                                if last_state_2 != false {
                                    self.is_pressed_2.store(false, Ordering::Release);
                                    last_state_2 = false;
                                }
                            } else {
                                if last_state_2 != true {
                                    self.is_pressed_2.store(true, Ordering::Release);
                                    last_state_2 = true;
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
            thread::sleep(Duration::from_millis(2));
        }
    }
}
