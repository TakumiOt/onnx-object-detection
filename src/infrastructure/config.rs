use std::env;

use crate::infrastructure::yolox_detector::DetectorConfig;

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub struct AppConfig {
    pub device_path: String,
    pub model_path: String,
    pub labels_path: String,
    pub bind_addr: String,
    pub detector: DetectorConfig,
    pub jpeg_quality: u8,
}

impl AppConfig {
    pub fn from_env() -> Self {
        Self {
            device_path: env::var("DEVICE_PATH").unwrap_or_else(|_| "/dev/video0".into()),
            model_path: env::var("MODEL_PATH").unwrap_or_else(|_| "models/yolox_s.onnx".into()),
            labels_path: env::var("LABELS_PATH").unwrap_or_else(|_| "labels.csv".into()),
            bind_addr: env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".into()),
            detector: DetectorConfig {
                input_size: env_or("INPUT_SIZE", 640),
                score_threshold: env_or("SCORE_THRESHOLD", 0.5),
                nms_iou_threshold: env_or("NMS_IOU_THRESHOLD", 0.45),
                num_classes: env_or("NUM_CLASSES", 80),
            },
            jpeg_quality: env_or("JPEG_QUALITY", 75),
        }
    }
}
