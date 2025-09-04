use crate::aim::{AimMode, Mode};
use actix_cors::Cors;
use actix_web::{App, HttpResponse, HttpServer, Responder, Result, get, http::header, put, web};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio_util::sync::CancellationToken;

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

#[get("/stream/board")]
async fn board() -> impl Responder {
    web::Html::new(String::from(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stream Event Control</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        body {
            font-family: 'Inter', sans-serif;
        }
    </style>
</head>
<body class="bg-gray-900 text-gray-100 p-6 flex items-center justify-center min-h-screen">

    <div class="w-full max-w-2xl bg-gray-800 p-8 rounded-2xl shadow-2xl">
        <!-- Main title -->
        <h1 class="text-3xl font-bold text-center mb-8 text-blue-400">Mouse Control Panel</h1>

        <!-- Status message display -->
        <div id="status-message" class="text-center text-sm font-medium h-6 mb-4"></div>

        <!-- Button grid container -->
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-6">

            <!-- AimOff Button -->
            <button
                class="button-style bg-red-600 hover:bg-red-700 active:bg-red-800"
                onclick="sendEvent('0')">
                <span class="text-lg">🔴</span> Aim Off
            </button>

            <!-- AimOn Button -->
            <button
                class="button-style bg-green-600 hover:bg-green-700 active:bg-green-800"
                onclick="sendEvent('1')">
                <span class="text-lg">🟢</span> Aim On
            </button>

            <!-- AimModeHead Button -->
            <button
                class="button-style bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800"
                onclick="sendEvent('2')">
                <span class="text-lg">👤</span> Aim Head
            </button>

            <!-- AimModeNeck Button -->
            <button
                class="button-style bg-purple-600 hover:bg-purple-700 active:bg-purple-800"
                onclick="sendEvent('3')">
                <span class="text-lg">👔</span> Aim Neck
            </button>
            
            <!-- AimModeChest Button -->
            <button
                class="button-style bg-pink-600 hover:bg-pink-700 active:bg-pink-800"
                onclick="sendEvent('4')">
                <span class="text-lg">🎽</span> Aim Chest
            </button>

            <!-- AimModeAbdomen Button -->
            <button
                class="button-style bg-yellow-600 hover:bg-yellow-700 active:bg-yellow-800"
                onclick="sendEvent('5')">
                <span class="text-lg">🎯</span> Aim Abdomen
            </button>

        </div>
    </div>

    <script>
        // Dynamically get the base URL from the browser's current location.
        // This makes the app portable and environment-agnostic.
        const BASE_URL = `${window.location.protocol}//${window.location.host}/stream/event`;
        
        /**
         * Sends a PUT request to the specified event URL without a body.
         * @param {string} eventType The type of event to send (e.g., 'AimOn', 'AimModeHead').
         */
        async function sendEvent(eventType) {
            // Get the status message element to provide feedback.
            const statusElement = document.getElementById('status-message');
            statusElement.className = 'text-center text-sm font-medium h-6 mb-4 text-yellow-500';
            statusElement.textContent = `Sending event: ${eventType}...`;

            try {
                const url = `${BASE_URL}/${eventType}`;
                const response = await fetch(url, {
                    method: 'PUT'
                });

                if (response.ok) {
                    // Success message.
                    statusElement.className = 'text-center text-sm font-medium h-6 mb-4 text-green-500';
                    statusElement.textContent = `Event '${eventType}' sent successfully!`;
                } else {
                    // Error message for non-200 responses.
                    statusElement.className = 'text-center text-sm font-medium h-6 mb-4 text-red-500';
                    statusElement.textContent = `Failed to send event. Status: ${response.status}`;
                }
            } catch (error) {
                // Catch network or CORS errors.
                statusElement.className = 'text-center text-sm font-medium h-6 mb-4 text-red-500';
                statusElement.textContent = `Error: Could not connect to server. Please ensure the server is running at ${BASE_URL}.`;
                console.error('Fetch error:', error);
            }
        }
    </script>
    <style>
        .button-style {
            @apply flex items-center justify-center p-4 rounded-xl font-bold text-white transition-all duration-200 ease-in-out transform hover:scale-105 shadow-lg;
        }
    </style>
</body>
</html>
"#,
    ))
}

#[get("/stream/status")]
async fn stream_status(
    signal: web::Data<Arc<AtomicBool>>,
    aim_mode: web::Data<AimMode>,
) -> Result<HttpResponse> {
    let signal = if signal.load(Ordering::Relaxed) {
        "ON"
    } else {
        "OFF"
    };
    let aim_mode = aim_mode.to_string();
    Ok(HttpResponse::Ok().body(format!("{signal},{aim_mode}")))
}

pub async fn start_event_listener(
    signal: Arc<AtomicBool>,
    aim_mode: AimMode,
    serving_port: u16,
    cancel_token: CancellationToken,
) -> anyhow::Result<()> {
    let signal = web::Data::new(signal);
    let aim_mode = web::Data::new(aim_mode);
    HttpServer::new(move || {
        App::new()
            .wrap(
                Cors::default()
                    .allow_any_origin()
                    .allowed_headers(vec![
                        header::AUTHORIZATION,
                        header::ACCEPT,
                        header::CONTENT_TYPE,
                    ])
                    .allowed_methods(vec!["GET", "PUT"])
                    .max_age(3600),
            )
            .app_data(signal.clone())
            .app_data(aim_mode.clone())
            .app_data(web::PayloadConfig::default().limit(1024 * 1024))
            .route("/health", web::get().to(HttpResponse::Ok))
            .service(event)
            .service(board)
            .service(stream_status)
    })
    .workers(2)
    .bind(format!("0.0.0.0:{serving_port}"))?
    .shutdown_signal(cancel_token.cancelled_owned())
    .run()
    .await?;
    tracing::warn!("Server stopped.");
    Ok(())
}

impl TryFrom<&str> for Event {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "aim_off" | "aimOff" | "AimOff" | "off" | "Off" | "OFF" | "0" => Ok(Event::AimOff),
            "aim_on" | "aimOn" | "AimOn" | "on" | "On" | "ON" | "1" => Ok(Event::AimOn),
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
