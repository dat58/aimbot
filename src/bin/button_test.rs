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
    let state = Arc::new(AtomicBool::new(false));
    let mut esp_button = EspButton::new(&config.esp_port.unwrap(), state.clone())
        .expect("Failed to connect to ESP button");
    std::thread::spawn(move || {
        esp_button.listen();
    });
    let mut last_state = state.load(Ordering::Acquire);
    loop {
        let current_state = state.load(Ordering::Acquire);
        if last_state != current_state {
            if current_state {
                println!("PRESSED");
            } else {
                println!("RELEASED");
            }
            last_state = current_state;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}
