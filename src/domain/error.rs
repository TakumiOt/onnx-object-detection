use thiserror::Error;

#[derive(Debug, Error)]
pub enum DomainError {
    #[error("capture error: {0}")]
    Capture(String),

    #[error("inference error: {0}")]
    Inference(String),

    #[error("render error: {0}")]
    Render(String),

    #[error("config error: {0}")]
    Config(String),
}
