use crate::aim::{AimMode, Mode};
use actix_web::{App, HttpResponse, HttpServer, Result, put, web};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub enum Event {
    AimOff,
    AimOn,
    AimModeHead,
    AimModeNeck,
    AimModeChest,
    AimModeAbdomen,
}

#[put("/stream/event/{id}")]
async fn event(
    id: web::Path<String>,
    signal: web::Data<Arc<AtomicBool>>,
    aim_mode: web::Data<AimMode>,
) -> Result<HttpResponse> {
    let id = id.into_inner();
    match Event::try_from(id.as_str()) {
        Ok(event) => {
            match event {
                Event::AimOff => {
                    signal.store(false, Ordering::SeqCst);
                    tracing::info!("[Event] turn off aim bot.")
                }
                Event::AimOn => {
                    signal.store(true, Ordering::SeqCst);
                    tracing::info!("[Event] turn on aim bot.")
                }
                Event::AimModeHead => {
                    aim_mode.set_mode(Mode::Head);
                    tracing::info!("[Event] change to aim mode Head.")
                }
                Event::AimModeNeck => {
                    aim_mode.set_mode(Mode::Neck);
                    tracing::info!("[Event] change to aim mode Neck.")
                }
                Event::AimModeChest => {
                    aim_mode.set_mode(Mode::Chest);
                    tracing::info!("[Event] change to aim mode Chest.")
                }
                Event::AimModeAbdomen => {
                    aim_mode.set_mode(Mode::Abdomen);
                    tracing::info!("[Event] change to aim mode Abdomen.")
                }
            }
            Ok(HttpResponse::Ok().finish())
        }
        Err(_) => Ok(HttpResponse::BadRequest().body("Invalid event id")),
    }
}

pub fn start_event_listener(signal: Arc<AtomicBool>, aim_mode: AimMode, serving_port: u16) -> anyhow::Result<()> {
    actix_web::rt::System::new().block_on(async {
        let signal = web::Data::new(signal);
        let aim_mode = web::Data::new(aim_mode);
        HttpServer::new(move || {
            App::new()
                .app_data(signal.clone())
                .app_data(aim_mode.clone())
                .app_data(web::PayloadConfig::default().limit(1024 * 1024))
                .route("/health", web::get().to(HttpResponse::Ok))
                .service(event)
        })
        .workers(1)
        .bind(format!("0.0.0.0:{serving_port}"))?
        .run()
        .await?;
        Ok(())
    })
}

impl TryFrom<&str> for Event {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "aim_off" | "aimOff" | "AimOff" | "on" | "off" | "Off" | "OFF" | "0" => {
                Ok(Event::AimOff)
            }
            "aim_on" | "aimOn" | "AimOn" | "On" | "ON" | "1" => Ok(Event::AimOn),
            "aim_mode_head" | "aimModeHead" | "AimModeHead" | "head" | "Head" | "2" => {
                Ok(Event::AimModeHead)
            }
            "aim_mode_neck" | "aimModeNeck" | "AimModeNeck" | "neck" | "Neck" | "3" => {
                Ok(Event::AimModeNeck)
            }
            "aim_mode_chest" | "aimModeChest" | "AimModeChest" | "chest" | "Chest" | "4" => {
                Ok(Event::AimModeChest)
            }
            "aim_mode_abdomen" | "aimModeAbdomen" | "AimModeAbdomen" | "abdomen" | "Abdomen"
            | "5" => Ok(Event::AimModeAbdomen),
            _ => Err(value.to_string()),
        }
    }
}
