pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl BoundingBox {
    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);

        let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let area_self = self.width * self.height;
        let area_other = other.width * other.height;
        let union = area_self + area_other - intersection;

        if union <= 0.0 {
            0.0
        } else {
            intersection / union
        }
    }
}

pub struct Detection {
    pub bbox: BoundingBox,
    pub class_id: u32,
    pub label: String,
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bbox(x: f32, y: f32, w: f32, h: f32) -> BoundingBox {
        BoundingBox {
            x,
            y,
            width: w,
            height: h,
        }
    }

    #[test]
    fn identical_boxes_should_have_iou_of_one() {
        let a = bbox(0.0, 0.0, 10.0, 10.0);
        let b = bbox(0.0, 0.0, 10.0, 10.0);
        assert!((a.iou(&b) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn non_overlapping_boxes_should_have_iou_of_zero() {
        let a = bbox(0.0, 0.0, 10.0, 10.0);
        let b = bbox(20.0, 20.0, 10.0, 10.0);
        assert!((a.iou(&b)).abs() < f32::EPSILON);
    }

    #[test]
    fn partially_overlapping_boxes_should_return_correct_iou() {
        let a = bbox(0.0, 0.0, 10.0, 10.0);
        let b = bbox(5.0, 0.0, 10.0, 10.0);
        // intersection: 5*10=50, union: 100+100-50=150
        let expected = 50.0 / 150.0;
        assert!((a.iou(&b) - expected).abs() < 1e-6);
    }

    #[test]
    fn iou_should_be_symmetric() {
        let a = bbox(0.0, 0.0, 10.0, 10.0);
        let b = bbox(3.0, 4.0, 8.0, 6.0);
        assert!((a.iou(&b) - b.iou(&a)).abs() < f32::EPSILON);
    }

    #[test]
    fn zero_area_box_should_return_iou_of_zero() {
        let a = bbox(0.0, 0.0, 0.0, 0.0);
        let b = bbox(0.0, 0.0, 10.0, 10.0);
        assert!((a.iou(&b)).abs() < f32::EPSILON);
    }
}
