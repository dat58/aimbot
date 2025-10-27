use aimbot::{config::Config, mouse::MouseVirtual};
use std::{error::Error, sync::Arc, thread, time::Duration};
use tracing_subscriber::{EnvFilter, Layer, fmt, layer::SubscriberExt, util::SubscriberInitExt};

fn main() -> Result<(), Box<dyn Error>> {
    dotenv::dotenv().ok();
    tracing_subscriber::registry()
        .with(fmt::Layer::new().with_writer(std::io::stdout).with_filter(
            EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new("info"))?,
        ))
        .init();
    let config = Config::new();
    let mouse = Arc::new(MouseVirtual::new(&config.makcu_port, config.makcu_baud)?);
    let mouse_clone = mouse.clone();
    thread::spawn(move || {
        mouse_clone.listen_button_presses();
    });
    tracing::info!("[1] Testing for left mouse button presses");
    tracing::info!("[1] Please hold your left mouse button");
    while !mouse.is_left_pressing() {
        thread::sleep(Duration::from_millis(100));
    }
    tracing::info!("[1] Testing left mouse button presses successfully");
    tracing::info!("---------------------------------------------------");
    tracing::info!("[2] Testing for right mouse button presses");
    tracing::info!("[2] Please hold your right mouse button");
    while !mouse.is_right_pressing() {
        thread::sleep(Duration::from_millis(100));
    }
    tracing::info!("[2] Testing right mouse button presses successfully");
    tracing::info!("---------------------------------------------------");
    tracing::info!("[3] Testing for side4 mouse button presses");
    tracing::info!("[3] Please hold your side4 mouse button");
    while !mouse.is_side4_pressing() {
        thread::sleep(Duration::from_millis(100));
    }
    tracing::info!("[3] Testing side4 mouse button presses successfully");
    tracing::info!("---------------------------------------------------");
    tracing::info!("[4] Testing for side5 mouse button presses");
    tracing::info!("[4] Please hold your side5 mouse button");
    while !mouse.is_side5_pressing() {
        thread::sleep(Duration::from_millis(100));
    }
    tracing::info!("[4] Testing side5 mouse button presses successfully");
    tracing::info!("---------------------------------------------------");
    tracing::info!("[5] Testing for mouse move");
    tracing::info!("[5] Input separate for dy, dy; type q to quit");
    let mut random = rand::rng();
    loop {
        let mut value = String::new();
        tracing::info!("dx: ");
        std::io::stdin().read_line(&mut value)?;
        let v = value.trim();
        if v.trim() == "q" {
            break;
        }
        let dx = v.parse::<i64>()?;

        let mut value = String::new();
        tracing::info!("dy: ");
        std::io::stdin().read_line(&mut value)?;
        let v = value.trim();
        if v.trim() == "q" {
            break;
        }
        let dy = v.parse::<i64>()?;
        mouse.move_bezier(dx as f64, dy as f64, &mut random)?;
        tracing::info!("-------------------------");
    }
    Ok(())
}
