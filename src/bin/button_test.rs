use aimbot::{config::Config, esp_button::EspButton};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tracing_subscriber::{EnvFilter, Layer, fmt, layer::SubscriberExt, util::SubscriberInitExt};

fn main() {
    dotenv::dotenv().ok();
    tracing_subscriber::registry()
        .with(
            fmt::Layer::new().with_writer(std::io::stdout).with_filter(
                EnvFilter::try_from_default_env()
                    .or_else(|_| EnvFilter::try_new("info"))
                    .unwrap(),
            ),
        )
        .init();
    tracing::info!(
        "{:?}",
        serialport::available_ports().expect("No serial ports found!")
    );
    let config = Config::new();
    let state1 = Arc::new(AtomicBool::new(false));
    let state2 = Arc::new(AtomicBool::new(false));
    let mut esp_button = EspButton::new(&config.esp_port.unwrap(), state1.clone(), state2.clone())
        .expect("Failed to connect to ESP button");
    std::thread::spawn(move || {
        esp_button.listen();
    });
    let mut last_state1 = state1.load(Ordering::Acquire);
    let mut last_state2 = state1.load(Ordering::Acquire);
    loop {
        let current_state1 = state1.load(Ordering::Acquire);
        if last_state1 != current_state1 {
            if current_state1 {
                println!("[BUTTON 1] PRESSED");
            } else {
                println!("[BUTTON 1] RELEASED");
            }
            last_state1 = current_state1;
        }

        let current_state2 = state2.load(Ordering::Acquire);
        if last_state2 != current_state2 {
            if current_state2 {
                println!("[BUTTON 2] PRESSED");
            } else {
                println!("[BUTTON 2] RELEASED");
            }
            last_state2 = current_state2;
        }

        std::thread::sleep(Duration::from_millis(2));
    }
}
