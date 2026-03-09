use axum::extract::State;
use axum::response::{Html, IntoResponse};
use bytes::Bytes;
use tokio::sync::watch;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::WatchStream;

#[derive(Clone)]
pub struct MjpegState {
    pub rx: watch::Receiver<Vec<u8>>,
}

pub async fn mjpeg_stream(State(state): State<MjpegState>) -> impl IntoResponse {
    let stream = WatchStream::new(state.rx)
        .filter(|data| !data.is_empty())
        .map(|jpeg| {
            let header = format!(
                "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                jpeg.len()
            );
            let mut payload = Vec::with_capacity(header.len() + jpeg.len() + 2);
            payload.extend_from_slice(header.as_bytes());
            payload.extend_from_slice(&jpeg);
            payload.extend_from_slice(b"\r\n");
            Ok::<_, std::convert::Infallible>(Bytes::from(payload))
        });

    let body = axum::body::Body::from_stream(stream);

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "multipart/x-mixed-replace; boundary=frame",
        )],
        body,
    )
}

pub async fn index_html() -> Html<&'static str> {
    Html(
        r#"<!DOCTYPE html>
<html>
<head><title>Object Detection</title></head>
<body style="margin:0;background:#000;display:flex;justify-content:center;align-items:center;height:100vh">
<img src="/stream" style="max-width:100%;max-height:100vh">
</body>
</html>"#,
    )
}
