use std::collections::HashMap;

use ndarray::Array4;
use ort::session::Session;

use crate::domain::detection::{BoundingBox, Detection};
use crate::domain::error::DomainError;
use crate::domain::frame::Frame;
use crate::use_case::port::ObjectDetector;

pub struct DetectorConfig {
    pub input_size: u32,
    pub score_threshold: f32,
    pub nms_iou_threshold: f32,
    pub num_classes: usize,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            input_size: 640,
            score_threshold: 0.5,
            nms_iou_threshold: 0.45,
            num_classes: 80,
        }
    }
}

pub struct YoloxDetector {
    session: Session,
    labels: HashMap<usize, String>,
    config: DetectorConfig,
}

impl YoloxDetector {
    pub fn new(
        model_path: &str,
        labels: HashMap<usize, String>,
        config: DetectorConfig,
    ) -> Result<Self, DomainError> {
        let session = Session::builder()
            .and_then(|mut b| b.commit_from_file(model_path))
            .map_err(|e| DomainError::Config(format!("load model: {e}")))?;
        Ok(Self {
            session,
            labels,
            config,
        })
    }
}

struct LetterboxResult {
    input: Array4<f32>,
    scale: f32,
    dw: f32,
    dh: f32,
}

fn letterbox(frame: &Frame, input_size: u32) -> Result<LetterboxResult, DomainError> {
    let img = image::RgbImage::from_raw(frame.width, frame.height, frame.data.clone())
        .ok_or_else(|| DomainError::Inference("invalid frame dimensions".into()))?;

    let scale =
        (input_size as f32 / frame.width as f32).min(input_size as f32 / frame.height as f32);
    let new_w = (frame.width as f32 * scale) as u32;
    let new_h = (frame.height as f32 * scale) as u32;
    let dw = (input_size - new_w) as f32 / 2.0;
    let dh = (input_size - new_h) as f32 / 2.0;

    let resized =
        image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Triangle);

    let sz = input_size as usize;
    let mut input = Array4::<f32>::from_elem((1, 3, sz, sz), 114.0);
    let dw_u = dw as u32;
    let dh_u = dh as u32;

    for y in 0..new_h {
        for x in 0..new_w {
            let pixel = resized.get_pixel(x, y);
            let ty = (y + dh_u) as usize;
            let tx = (x + dw_u) as usize;
            input[[0, 0, ty, tx]] = pixel[0] as f32;
            input[[0, 1, ty, tx]] = pixel[1] as f32;
            input[[0, 2, ty, tx]] = pixel[2] as f32;
        }
    }

    Ok(LetterboxResult {
        input,
        scale,
        dw,
        dh,
    })
}

struct DecodeContext<'a> {
    scale: f32,
    dw: f32,
    dh: f32,
    labels: &'a HashMap<usize, String>,
    config: &'a DetectorConfig,
}

fn decode_outputs(shape: &[i64], data: &[f32], ctx: &DecodeContext) -> Vec<Detection> {
    let num = shape[1] as usize;
    let cols = shape[2] as usize;
    let mut detections = Vec::new();

    for i in 0..num {
        let offset = i * cols;
        let obj_conf = data[offset + 4];
        if obj_conf < ctx.config.score_threshold {
            continue;
        }

        let class_data = &data[offset + 5..offset + 5 + ctx.config.num_classes];
        let (best_class, &best_score) = class_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let label = match ctx.labels.get(&best_class) {
            Some(l) => l.clone(),
            None => continue,
        };

        let score = obj_conf * best_score;
        if score < ctx.config.score_threshold {
            continue;
        }

        let cx = (data[offset] - ctx.dw) / ctx.scale;
        let cy = (data[offset + 1] - ctx.dh) / ctx.scale;
        let w = data[offset + 2] / ctx.scale;
        let h = data[offset + 3] / ctx.scale;

        detections.push(Detection {
            bbox: BoundingBox {
                x: cx - w / 2.0,
                y: cy - h / 2.0,
                width: w,
                height: h,
            },
            class_id: best_class as u32,
            label,
            confidence: score,
        });
    }
    detections
}

fn nms(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    let mut keep = Vec::new();

    while !detections.is_empty() {
        let best = detections.remove(0);
        detections
            .retain(|d| d.class_id != best.class_id || best.bbox.iou(&d.bbox) < iou_threshold);
        keep.push(best);
    }
    keep
}

impl ObjectDetector for YoloxDetector {
    fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>, DomainError> {
        let lb = letterbox(frame, self.config.input_size)?;
        let ctx = DecodeContext {
            scale: lb.scale,
            dw: lb.dw,
            dh: lb.dh,
            labels: &self.labels,
            config: &self.config,
        };

        let input_value = ort::value::Tensor::from_array(lb.input)
            .map_err(|e| DomainError::Inference(format!("input: {e}")))?;

        let outputs = self
            .session
            .run(ort::inputs!["images" => input_value])
            .map_err(|e| DomainError::Inference(format!("run: {e}")))?;

        let (output_shape, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| DomainError::Inference(format!("extract: {e}")))?;

        let detections = decode_outputs(output_shape, output_data, &ctx);
        Ok(nms(detections, self.config.nms_iou_threshold))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_NUM_CLASSES: usize = 80;
    const TEST_SCORE_THRESHOLD: f32 = 0.5;
    const TEST_NMS_IOU_THRESHOLD: f32 = 0.45;

    fn test_labels() -> HashMap<usize, String> {
        HashMap::from([(0, "class_a".into()), (1, "class_b".into())])
    }

    fn test_config() -> DetectorConfig {
        DetectorConfig {
            input_size: 640,
            score_threshold: TEST_SCORE_THRESHOLD,
            nms_iou_threshold: TEST_NMS_IOU_THRESHOLD,
            num_classes: TEST_NUM_CLASSES,
        }
    }

    fn test_decode_ctx<'a>(
        scale: f32,
        dw: f32,
        dh: f32,
        labels: &'a HashMap<usize, String>,
        config: &'a DetectorConfig,
    ) -> DecodeContext<'a> {
        DecodeContext {
            scale,
            dw,
            dh,
            labels,
            config,
        }
    }

    fn make_output_row(
        cx: f32,
        cy: f32,
        w: f32,
        h: f32,
        obj_conf: f32,
        class_scores: &[f32],
    ) -> Vec<f32> {
        let mut row = vec![cx, cy, w, h, obj_conf];
        row.extend_from_slice(class_scores);
        row
    }

    #[test]
    fn decode_outputs_should_extract_high_confidence_detection() {
        let mut class_scores = vec![0.0f32; TEST_NUM_CLASSES];
        class_scores[0] = 0.9;
        let data = make_output_row(320.0, 240.0, 100.0, 200.0, 0.9, &class_scores);
        let cols = (5 + TEST_NUM_CLASSES) as i64;
        let shape = vec![1i64, 1, cols];

        let labels = test_labels();
        let config = test_config();
        let ctx = test_decode_ctx(1.0, 0.0, 0.0, &labels, &config);
        let dets = decode_outputs(&shape, &data, &ctx);

        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].label, "class_a");
        assert!((dets[0].confidence - 0.81).abs() < 1e-4);
        assert!((dets[0].bbox.x - 270.0).abs() < 1e-4);
        assert!((dets[0].bbox.y - 140.0).abs() < 1e-4);
    }

    #[test]
    fn decode_outputs_should_filter_low_objectness() {
        let mut class_scores = vec![0.0f32; TEST_NUM_CLASSES];
        class_scores[0] = 0.9;
        let data = make_output_row(100.0, 100.0, 50.0, 50.0, 0.1, &class_scores);
        let cols = (5 + TEST_NUM_CLASSES) as i64;
        let shape = vec![1i64, 1, cols];

        let labels = test_labels();
        let config = test_config();
        let ctx = test_decode_ctx(1.0, 0.0, 0.0, &labels, &config);
        let dets = decode_outputs(&shape, &data, &ctx);
        assert!(dets.is_empty());
    }

    #[test]
    fn decode_outputs_should_filter_low_combined_score() {
        let mut class_scores = vec![0.0f32; TEST_NUM_CLASSES];
        class_scores[0] = 0.4;
        let data = make_output_row(100.0, 100.0, 50.0, 50.0, 0.6, &class_scores);
        let cols = (5 + TEST_NUM_CLASSES) as i64;
        let shape = vec![1i64, 1, cols];

        let labels = test_labels();
        let config = test_config();
        let ctx = test_decode_ctx(1.0, 0.0, 0.0, &labels, &config);
        let dets = decode_outputs(&shape, &data, &ctx);
        assert!(dets.is_empty());
    }

    #[test]
    fn decode_outputs_should_apply_scale_and_offset() {
        let mut class_scores = vec![0.0f32; TEST_NUM_CLASSES];
        class_scores[1] = 0.95;
        let data = make_output_row(330.0, 250.0, 100.0, 80.0, 0.95, &class_scores);
        let cols = (5 + TEST_NUM_CLASSES) as i64;
        let shape = vec![1i64, 1, cols];

        let labels = test_labels();
        let config = test_config();
        let ctx = test_decode_ctx(0.5, 10.0, 20.0, &labels, &config);
        let dets = decode_outputs(&shape, &data, &ctx);

        assert_eq!(dets.len(), 1);
        assert!((dets[0].bbox.x - (640.0 - 100.0)).abs() < 1e-4);
        assert!((dets[0].bbox.y - (460.0 - 80.0)).abs() < 1e-4);
    }

    #[test]
    fn decode_outputs_should_skip_class_not_in_labels() {
        let mut class_scores = vec![0.0f32; TEST_NUM_CLASSES];
        class_scores[5] = 0.95; // index 5 is not in test_labels()
        let data = make_output_row(100.0, 100.0, 50.0, 50.0, 0.95, &class_scores);
        let cols = (5 + TEST_NUM_CLASSES) as i64;
        let shape = vec![1i64, 1, cols];

        let labels = test_labels();
        let config = test_config();
        let ctx = test_decode_ctx(1.0, 0.0, 0.0, &labels, &config);
        let dets = decode_outputs(&shape, &data, &ctx);
        assert!(dets.is_empty());
    }

    #[test]
    fn nms_should_suppress_overlapping_same_class_detections() {
        let d1 = Detection {
            bbox: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 100.0,
            },
            class_id: 0,
            label: "person".into(),
            confidence: 0.9,
        };
        let d2 = Detection {
            bbox: BoundingBox {
                x: 10.0,
                y: 10.0,
                width: 100.0,
                height: 100.0,
            },
            class_id: 0,
            label: "person".into(),
            confidence: 0.7,
        };

        let result = nms(vec![d1, d2], TEST_NMS_IOU_THRESHOLD);
        assert_eq!(result.len(), 1);
        assert!((result[0].confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn nms_should_keep_different_class_detections() {
        let d1 = Detection {
            bbox: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 100.0,
            },
            class_id: 0,
            label: "person".into(),
            confidence: 0.9,
        };
        let d2 = Detection {
            bbox: BoundingBox {
                x: 10.0,
                y: 10.0,
                width: 100.0,
                height: 100.0,
            },
            class_id: 1,
            label: "bicycle".into(),
            confidence: 0.8,
        };

        let result = nms(vec![d1, d2], TEST_NMS_IOU_THRESHOLD);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn nms_should_keep_non_overlapping_same_class_detections() {
        let d1 = Detection {
            bbox: BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 50.0,
                height: 50.0,
            },
            class_id: 0,
            label: "person".into(),
            confidence: 0.9,
        };
        let d2 = Detection {
            bbox: BoundingBox {
                x: 200.0,
                y: 200.0,
                width: 50.0,
                height: 50.0,
            },
            class_id: 0,
            label: "person".into(),
            confidence: 0.8,
        };

        let result = nms(vec![d1, d2], TEST_NMS_IOU_THRESHOLD);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn letterbox_should_return_error_for_invalid_frame() {
        let frame = Frame {
            data: vec![0u8; 10], // too short for 640x480
            width: 640,
            height: 480,
        };
        assert!(letterbox(&frame, 640).is_err());
    }

    #[test]
    fn letterbox_should_succeed_for_valid_frame() {
        let width = 4u32;
        let height = 2u32;
        let frame = Frame {
            data: vec![0u8; (width * height * 3) as usize],
            width,
            height,
        };
        let result = letterbox(&frame, 640);
        assert!(result.is_ok());
        let lb = result.unwrap();
        assert_eq!(lb.input.shape(), &[1, 3, 640, 640]);
    }
}
