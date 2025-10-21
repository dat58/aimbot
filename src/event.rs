use crate::aim::{AimMode, Mode};
use actix_cors::Cors;
use actix_web::{App, HttpResponse, HttpServer, Responder, Result, get, http::header, put, web};
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
    UseTrigger,
    UseAutoAim,
}

#[put("/stream/event/{id}")]
async fn event(
    id: web::Path<String>,
    signal: web::Data<(Arc<AtomicBool>, Arc<AtomicBool>)>,
    aim_mode: web::Data<AimMode>,
) -> Result<HttpResponse> {
    let id = id.into_inner();
    let signal = signal.into_inner();
    let (use_trigger, use_auto_aim) = (&signal.0, &signal.1);
    match Event::try_from(id.as_str()) {
        Ok(event) => {
            match event {
                Event::AimOff => {
                    use_auto_aim.store(false, Ordering::SeqCst);
                    tracing::info!("[Event] turn off aim bot.")
                }
                Event::AimOn => {
                    use_auto_aim.store(true, Ordering::SeqCst);
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
                Event::UseTrigger => {
                    use_trigger.store(true, Ordering::SeqCst);
                    use_auto_aim.store(true, Ordering::SeqCst);
                    tracing::info!("[Event] change to Trigger.")
                }
                Event::UseAutoAim => {
                    use_trigger.store(false, Ordering::SeqCst);
                    use_auto_aim.store(true, Ordering::SeqCst);
                    tracing::info!("[Event] change to Auto Aim.")
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
        .button-style {
            /* Applied Tailwind classes for consistent styling */
            @apply flex items-center justify-center p-4 rounded-xl font-bold text-white transition-all duration-200 ease-in-out transform hover:scale-105 shadow-lg;
        }
    </style>
</head>
<body class="bg-gray-900 text-gray-100 p-6 flex items-center justify-center min-h-screen">

    <div class="w-full max-w-2xl bg-gray-800 p-8 rounded-2xl shadow-2xl border border-gray-700">
        <!-- Main title -->
        <h1 class="text-3xl font-bold text-center mb-8 text-blue-400">Mouse Control Panel</h1>

        <!-- Status message display -->
        <div id="status-message" class="text-center text-sm font-medium h-6 mb-4"></div>

        <!-- Button grid container -->
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-6">

            <!-- 0: AimOff Button -->
            <button
                class="button-style bg-red-600 hover:bg-red-700 active:bg-red-800"
                onclick="sendEvent('0')">
                <span class="text-lg mr-2">🔴</span> Aim Off
            </button>

            <!-- 1: AimOn Button -->
            <button
                class="button-style bg-green-600 hover:bg-green-700 active:bg-green-800"
                onclick="sendEvent('1')">
                <span class="text-lg mr-2">🟢</span> Aim On
            </button>

            <!-- 2: AimModeHead Button -->
            <button
                class="button-style bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800"
                onclick="sendEvent('2')">
                <span class="text-lg mr-2">👤</span> Aim Head
            </button>

            <!-- 3: AimModeNeck Button -->
            <button
                class="button-style bg-purple-600 hover:bg-purple-700 active:bg-purple-800"
                onclick="sendEvent('3')">
                <span class="text-lg mr-2">👔</span> Aim Neck
            </button>
            
            <!-- 4: AimModeChest Button -->
            <button
                class="button-style bg-pink-600 hover:bg-pink-700 active:bg-pink-800"
                onclick="sendEvent('4')">
                <span class="text-lg mr-2">🎽</span> Aim Chest
            </button>

            <!-- 5: AimModeAbdomen Button -->
            <button
                class="button-style bg-yellow-600 hover:bg-yellow-700 active:bg-yellow-800"
                onclick="sendEvent('5')">
                <span class="text-lg mr-2">🎯</span> Aim Abdomen
            </button>

            <!-- NEW 6: UseTrigger Button -->
            <button
                class="button-style bg-teal-600 hover:bg-teal-700 active:bg-teal-800"
                onclick="sendEvent('6')">
                <span class="text-lg mr-2">⚙️</span> Use Trigger
            </button>

            <!-- NEW 7: UseAutoAim Button -->
            <button
                class="button-style bg-orange-600 hover:bg-orange-700 active:bg-orange-800"
                onclick="sendEvent('7')">
                <span class="text-lg mr-2">🤖</span> Use Auto Aim
            </button>

        </div>
    </div>

    <script>
        // Dynamically get the base URL from the browser's current location.
        const BASE_URL = `${window.location.protocol}//${window.location.host}/stream/event`;
        
        /**
         * Sends a PUT request to the specified event URL without a body.
         * @param {string} eventType The type of event to send (e.g., '0' to '7').
         */
        async function sendEvent(eventType) {
            // Get the status message element to provide feedback.
            const statusElement = document.getElementById('status-message');
            statusElement.className = 'text-center text-sm font-medium h-6 mb-4 text-yellow-500';
            statusElement.textContent = `Sending event: ${eventType}...`;

            // Function for exponential backoff retry logic
            const maxRetries = 3;
            const initialDelay = 1000; // 1 second

            for (let attempt = 0; attempt < maxRetries; attempt++) {
                try {
                    const url = `${BASE_URL}/${eventType}`;
                    const response = await fetch(url, {
                        method: 'PUT'
                    });

                    if (response.ok) {
                        // Success message.
                        statusElement.className = 'text-center text-sm font-medium h-6 mb-4 text-green-500';
                        statusElement.textContent = `Event '${eventType}' sent successfully!`;
                        return; // Exit on success
                    } else {
                        // Non-200 responses are treated as temporary failures if not the last attempt
                        if (attempt < maxRetries - 1) {
                            const delay = initialDelay * Math.pow(2, attempt);
                            statusElement.textContent = `Server responded with ${response.status}. Retrying in ${delay / 1000}s... (Attempt ${attempt + 1}/${maxRetries})`;
                            await new Promise(resolve => setTimeout(resolve, delay));
                        } else {
                            statusElement.className = 'text-center text-sm font-medium h-6 mb-4 text-red-500';
                            statusElement.textContent = `Failed to send event. Status: ${response.status} after ${maxRetries} attempts.`;
                            return; // Exit after max retries
                        }
                    }
                } catch (error) {
                    // Catch network errors
                    if (attempt < maxRetries - 1) {
                        const delay = initialDelay * Math.pow(2, attempt);
                        statusElement.textContent = `Network error. Retrying in ${delay / 1000}s... (Attempt ${attempt + 1}/${maxRetries})`;
                        console.error('Fetch error:', error);
                        await new Promise(resolve => setTimeout(resolve, delay));
                    } else {
                        statusElement.className = 'text-center text-sm font-medium h-6 mb-4 text-red-500';
                        statusElement.textContent = `Fatal Error: Could not connect to server at ${BASE_URL}.`;
                        console.error('Final Fetch error:', error);
                        return; // Exit after max retries
                    }
                }
            }
        }
    </script>
</body>
</html>
"#,
    ))
}

#[get("/stream/status")]
async fn stream_status(
    signal: web::Data<(Arc<AtomicBool>, Arc<AtomicBool>)>,
    aim_mode: web::Data<AimMode>,
) -> Result<HttpResponse> {
    let signal = signal.into_inner();
    let (use_trigger, use_auto_aim) = (&signal.0, &signal.1);
    let signal = if use_trigger.load(Ordering::Relaxed) {
        "Trigger"
    } else {
        if use_auto_aim.load(Ordering::Relaxed) {
            "Auto [ON]"
        } else {
            "Auto [OFF]"
        }
    };
    let aim_mode = aim_mode.to_string();
    Ok(HttpResponse::Ok().body(format!("{signal},{aim_mode}")))
}

pub fn start_event_listener(
    use_trigger: Arc<AtomicBool>,
    use_auto_aim: Arc<AtomicBool>,
    aim_mode: AimMode,
    serving_port: u16,
) -> anyhow::Result<()> {
    actix_web::rt::System::new().block_on(async {
        let signal = web::Data::new((use_trigger, use_auto_aim));
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
        .run()
        .await?;
        Ok(())
    })
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
            "trigger" | "Trigger" | "6" => Ok(Event::UseTrigger),
            "auto_aim" | "autoAim" | "AutoAim" | "7" => Ok(Event::UseAutoAim),
            _ => Err(value.to_string()),
        }
    }
}
