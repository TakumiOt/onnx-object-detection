use image::{Rgb, RgbImage};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;

use crate::domain::detection::Detection;
use crate::domain::error::DomainError;
use crate::domain::frame::Frame;
use crate::use_case::port::FrameRenderer;

const JPEG_QUALITY: u8 = 75;

const COLOR_PALETTE: [Rgb<u8>; 10] = [
    Rgb([255, 0, 0]),
    Rgb([0, 255, 0]),
    Rgb([0, 0, 255]),
    Rgb([255, 255, 0]),
    Rgb([255, 0, 255]),
    Rgb([0, 255, 255]),
    Rgb([255, 128, 0]),
    Rgb([128, 0, 255]),
    Rgb([0, 255, 128]),
    Rgb([255, 128, 128]),
];

pub struct ImageRenderer;

impl ImageRenderer {
    pub fn new() -> Self {
        Self
    }
}

impl FrameRenderer for ImageRenderer {
    fn render(&self, frame: &Frame, detections: &[Detection]) -> Result<Vec<u8>, DomainError> {
        let mut img = RgbImage::from_raw(frame.width, frame.height, frame.data.clone())
            .ok_or_else(|| DomainError::Render("invalid frame data".into()))?;

        for det in detections {
            let color = COLOR_PALETTE[det.class_id as usize % COLOR_PALETTE.len()];
            let x = det.bbox.x.max(0.0) as i32;
            let y = det.bbox.y.max(0.0) as i32;
            let w = det.bbox.width.min(frame.width as f32 - x as f32) as u32;
            let h = det.bbox.height.min(frame.height as f32 - y as f32) as u32;

            if w > 0 && h > 0 {
                draw_hollow_rect_mut(&mut img, Rect::at(x, y).of_size(w, h), color);
            }
        }

        encode_jpeg(&img)
    }
}

fn encode_jpeg(img: &RgbImage) -> Result<Vec<u8>, DomainError> {
    let mut buf = Vec::new();
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, JPEG_QUALITY);
    img.write_with_encoder(encoder)
        .map_err(|e| DomainError::Render(format!("jpeg encode: {e}")))?;
    Ok(buf)
}
