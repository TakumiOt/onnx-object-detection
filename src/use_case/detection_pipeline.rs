use tokio::sync::watch;
use tracing::warn;

use crate::use_case::port::{FrameRenderer, FrameSource, ObjectDetector};

pub struct DetectionPipeline<S, D, R> {
    source: S,
    detector: D,
    renderer: R,
}

impl<S, D, R> DetectionPipeline<S, D, R>
where
    S: FrameSource,
    D: ObjectDetector,
    R: FrameRenderer,
{
    pub fn new(source: S, detector: D, renderer: R) -> Self {
        Self {
            source,
            detector,
            renderer,
        }
    }

    pub fn run(mut self, tx: watch::Sender<Vec<u8>>)
    where
        S: FrameSource,
        D: ObjectDetector,
        R: FrameRenderer,
    {
        loop {
            let frame = match self.source.capture() {
                Ok(f) => f,
                Err(e) => {
                    warn!("capture failed: {e}");
                    continue;
                }
            };

            let detections = match self.detector.detect(&frame) {
                Ok(d) => d,
                Err(e) => {
                    warn!("detection failed: {e}");
                    continue;
                }
            };

            let jpeg = match self.renderer.render(&frame, &detections) {
                Ok(j) => j,
                Err(e) => {
                    warn!("render failed: {e}");
                    continue;
                }
            };

            if tx.send(jpeg).is_err() {
                break;
            }
        }
    }
}
