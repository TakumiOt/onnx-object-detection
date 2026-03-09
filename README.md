# onnx-object-detection

V4L2 カメラから映像を取得し、YOLOX (ONNX Runtime) でリアルタイム物体検出を行い、結果を MJPEG ストリームとしてブラウザに配信する Rust アプリケーションです。

## Requirements

- Linux (V4L2 対応カメラ)
- Rust 1.85+
- YOLOX の ONNX モデルファイル

## Quick Start

```bash
# Build
cargo build --release

# モデルを配置
mkdir -p models
# yolox_s.onnx を models/ に配置

# 実行
./target/release/onnx-object-detection
```

ブラウザで `http://localhost:8080` を開くとストリームが表示されます。

## Configuration

すべて環境変数で設定できます。

| Variable | Default | Description |
|---|---|---|
| `DEVICE_PATH` | `/dev/video0` | V4L2 カメラデバイスパス |
| `MODEL_PATH` | `models/yolox_s.onnx` | ONNX モデルファイルパス |
| `LABELS_PATH` | `labels.csv` | ラベル CSV ファイルパス |
| `BIND_ADDR` | `0.0.0.0:8080` | HTTP サーバーのバインドアドレス |
| `SCORE_THRESHOLD` | `0.5` | 検出スコア閾値 |
| `NMS_IOU_THRESHOLD` | `0.45` | NMS の IoU 閾値 |
| `INPUT_SIZE` | `640` | モデル入力サイズ (px) |
| `NUM_CLASSES` | `80` | モデルのクラス数 |
| `JPEG_QUALITY` | `75` | MJPEG ストリームの JPEG 品質 (0-100) |

```bash
SCORE_THRESHOLD=0.3 BIND_ADDR=0.0.0.0:3000 cargo run --release
```

## Labels CSV Format

```csv
0,person
1,bicycle
2,car
```

インデックスは YOLOX モデルのクラス ID に対応します。ラベルに含まれないクラスの検出はスキップされます。

## Architecture

Clean Architecture に基づく 4 層構造です。

```
src/
  domain/          # Entities: Detection, Frame, DomainError
  use_case/        # Use Cases: DetectionPipeline, Ports (traits)
  adapter/         # Adapters: MJPEG HTTP handler
  infrastructure/  # Infrastructure: V4L2, YOLOX/ONNX, ImageRenderer, Config
```

## Development

```bash
# Lint
cargo clippy

# Format
cargo fmt

# Test
cargo test
```
