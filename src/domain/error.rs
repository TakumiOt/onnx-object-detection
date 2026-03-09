use thiserror::Error;

#[derive(Debug, Error)]
pub enum DomainError {
    #[error("capture error: {0}")]
    CaptureError(String),

    #[error("inference error: {0}")]
    InferenceError(String),

    #[error("render error: {0}")]
    RenderError(String),

    #[error("config error: {0}")]
    ConfigError(String),
}
