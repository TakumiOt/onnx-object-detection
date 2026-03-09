use crate::domain::detection::Detection;
use crate::domain::error::DomainError;
use crate::domain::frame::Frame;

pub trait FrameSource: Send {
    fn capture(&mut self) -> Result<Frame, DomainError>;
}

pub trait ObjectDetector: Send {
    fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>, DomainError>;
}

pub trait FrameRenderer: Send {
    fn render(&self, frame: &Frame, detections: &[Detection]) -> Result<Vec<u8>, DomainError>;
}
