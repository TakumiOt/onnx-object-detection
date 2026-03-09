mod adapter;
mod domain;
mod infrastructure;
mod use_case;

use axum::Router;
use axum::routing::get;
use tokio::sync::watch;
use tracing::info;

use adapter::mjpeg_handler::{MjpegState, index_html, mjpeg_stream};
use infrastructure::image_renderer::ImageRenderer;
use infrastructure::label_loader::load_labels;
use infrastructure::camera::v4l2::V4l2Camera;
use infrastructure::yolox_detector::YoloxDetector;
use use_case::detection_pipeline::DetectionPipeline;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let labels = load_labels(std::path::Path::new("labels.csv")).expect("failed to load labels");
    let camera = V4l2Camera::open("/dev/video0").expect("failed to open camera");
    let detector =
        YoloxDetector::new("models/yolox_s.onnx", labels).expect("failed to load YOLOX model");
    let renderer = ImageRenderer::new();

    let (tx, rx) = watch::channel(Vec::<u8>::new());

    let pipeline = DetectionPipeline::new(camera, detector, renderer);
    std::thread::spawn(move || pipeline.run(tx));

    let state = MjpegState { rx };
    let app = Router::new()
        .route("/", get(index_html))
        .route("/stream", get(mjpeg_stream))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    info!("listening on http://0.0.0.0:8080");
    axum::serve(listener, app).await.unwrap();
}
