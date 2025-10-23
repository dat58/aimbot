use std::time::Duration;

fn main() {
    println!(
        "{:?}",
        serialport::available_ports().expect("No serial ports found!")
    );
    let port = std::env::var("port").expect("Please input an serial port");
    let mut serial = serialport::new(&port, 115200)
        .timeout(Duration::from_millis(150))
        .open()
        .unwrap();
    let mut buf = [0u8; 8];
    let mut last_state = false;
    loop {
        match serial.read(&mut buf) {
            Ok(count) if count > 0 => {
                for i in 0..count {
                    if buf[i] == 48 || buf[i] == 49 {
                        if buf[i] == 49 {
                            if last_state != true {
                                println!("PRESSED");
                                last_state = true;
                            }
                        } else {
                            if last_state != false {
                                println!("RELEASED");
                                last_state = false;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}
