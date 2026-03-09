use v4l::buffer::Type;
use v4l::io::mmap::Stream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;
use v4l::{Device, FourCC};

use crate::domain::error::DomainError;
use crate::domain::frame::Frame;
use crate::use_case::port::FrameSource;

pub struct V4l2Camera {
    stream: Stream<'static>,
    width: u32,
    height: u32,
    format: CapturedFormat,
}

enum CapturedFormat {
    Yuyv,
    Mjpeg,
}

impl V4l2Camera {
    pub fn open(device_path: &str) -> Result<Self, DomainError> {
        let dev =
            Box::leak(Box::new(Device::with_path(device_path).map_err(|e| {
                DomainError::Capture(format!("open {device_path}: {e}"))
            })?));

        let (fmt, captured_format) = negotiate_format(dev)?;
        let width = fmt.width;
        let height = fmt.height;

        let stream = Stream::with_buffers(dev, Type::VideoCapture, 4)
            .map_err(|e| DomainError::Capture(format!("mmap stream: {e}")))?;

        Ok(Self {
            stream,
            width,
            height,
            format: captured_format,
        })
    }
}

fn negotiate_format(dev: &Device) -> Result<(v4l::Format, CapturedFormat), DomainError> {
    let mut fmt = dev
        .format()
        .map_err(|e| DomainError::Capture(format!("get format: {e}")))?;

    fmt.fourcc = FourCC::new(b"YUYV");
    fmt.width = 640;
    fmt.height = 480;
    if let Ok(f) = dev.set_format(&fmt)
        && f.fourcc == FourCC::new(b"YUYV")
    {
        return Ok((f, CapturedFormat::Yuyv));
    }

    fmt.fourcc = FourCC::new(b"MJPG");
    let f = dev
        .set_format(&fmt)
        .map_err(|e| DomainError::Capture(format!("set format: {e}")))?;
    Ok((f, CapturedFormat::Mjpeg))
}

fn yuyv_to_rgb(yuyv: &[u8], width: u32, height: u32) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    let mut rgb = Vec::with_capacity(pixel_count * 3);

    for chunk in yuyv.chunks_exact(4) {
        let (y0, u, y1, v) = (
            chunk[0] as f32,
            chunk[1] as f32,
            chunk[2] as f32,
            chunk[3] as f32,
        );
        for y in [y0, y1] {
            let r = (y + 1.402 * (v - 128.0)).clamp(0.0, 255.0) as u8;
            let g = (y - 0.344136 * (u - 128.0) - 0.714136 * (v - 128.0)).clamp(0.0, 255.0) as u8;
            let b = (y + 1.772 * (u - 128.0)).clamp(0.0, 255.0) as u8;
            rgb.extend_from_slice(&[r, g, b]);
        }
    }
    rgb
}

fn mjpeg_to_rgb(data: &[u8]) -> Result<(Vec<u8>, u32, u32), DomainError> {
    let img = image::load_from_memory(data)
        .map_err(|e| DomainError::Capture(format!("decode mjpeg: {e}")))?;
    let rgb = img.to_rgb8();
    let w = rgb.width();
    let h = rgb.height();
    Ok((rgb.into_raw(), w, h))
}

impl FrameSource for V4l2Camera {
    fn capture(&mut self) -> Result<Frame, DomainError> {
        let (buf, _meta) = self
            .stream
            .next()
            .map_err(|e| DomainError::Capture(format!("capture: {e}")))?;

        match self.format {
            CapturedFormat::Yuyv => {
                let data = yuyv_to_rgb(buf, self.width, self.height);
                Ok(Frame {
                    data,
                    width: self.width,
                    height: self.height,
                })
            }
            CapturedFormat::Mjpeg => {
                let (data, w, h) = mjpeg_to_rgb(buf)?;
                Ok(Frame {
                    data,
                    width: w,
                    height: h,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yuyv_to_rgb_should_convert_white_pixel_pair() {
        // Y=235, U=128, V=128 is white in YUYV
        let yuyv = vec![235, 128, 235, 128];
        let rgb = yuyv_to_rgb(&yuyv, 2, 1);
        assert_eq!(rgb.len(), 6);
        // Both pixels should be near-white
        for channel in &rgb {
            assert!(*channel >= 230, "expected near-white, got {channel}");
        }
    }

    #[test]
    fn yuyv_to_rgb_should_convert_black_pixel_pair() {
        // Y=16, U=128, V=128 is black in YUYV
        let yuyv = vec![16, 128, 16, 128];
        let rgb = yuyv_to_rgb(&yuyv, 2, 1);
        assert_eq!(rgb.len(), 6);
        for channel in &rgb {
            assert!(*channel <= 25, "expected near-black, got {channel}");
        }
    }

    #[test]
    fn yuyv_to_rgb_should_produce_correct_output_length() {
        let width = 4u32;
        let height = 2u32;
        let yuyv = vec![128u8; (width * height * 2) as usize];
        let rgb = yuyv_to_rgb(&yuyv, width, height);
        assert_eq!(rgb.len(), (width * height * 3) as usize);
    }

    #[test]
    fn yuyv_to_rgb_should_not_panic_on_extreme_values() {
        // Extreme values that could produce out-of-range intermediate results
        let yuyv = vec![255, 0, 255, 255];
        let rgb = yuyv_to_rgb(&yuyv, 2, 1);
        assert_eq!(rgb.len(), 6);
    }
}
