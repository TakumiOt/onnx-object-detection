mod adapter;
mod domain;
mod infrastructure;
mod use_case;

use axum::Router;
use axum::routing::get;
use tokio::sync::watch;
use tracing::info;

use adapter::mjpeg_handler::{MjpegState, index_html, mjpeg_stream};
use infrastructure::camera::v4l2::V4l2Camera;
use infrastructure::config::AppConfig;
use infrastructure::image_renderer::ImageRenderer;
use infrastructure::label_loader::load_labels;
use infrastructure::yolox_detector::YoloxDetector;
use use_case::detection_pipeline::DetectionPipeline;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let config = AppConfig::from_env();

    let labels =
        load_labels(std::path::Path::new(&config.labels_path)).expect("failed to load labels");
    let camera = V4l2Camera::open(&config.device_path).expect("failed to open camera");
    let detector = YoloxDetector::new(&config.model_path, labels, config.detector)
        .expect("failed to load YOLOX model");
    let renderer = ImageRenderer::new(config.jpeg_quality);

    let (tx, rx) = watch::channel(Vec::<u8>::new());

    let pipeline = DetectionPipeline::new(camera, detector, renderer);
    std::thread::spawn(move || pipeline.run(tx));

    let state = MjpegState { rx };
    let app = Router::new()
        .route("/", get(index_html))
        .route("/stream", get(mjpeg_stream))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&config.bind_addr)
        .await
        .unwrap();
    info!("listening on http://{}", config.bind_addr);
    axum::serve(listener, app).await.unwrap();
}
