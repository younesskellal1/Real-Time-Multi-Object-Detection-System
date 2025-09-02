# YOLO CNN Project

A computer vision project that combines YOLO (You Only Look Once) object detection with CNN-based age and gender classification.

## Features

- **YOLO Object Detection**: Real-time object detection using YOLOv8
- **Face Detection**: Haar Cascade-based face detection
- **Age Classification**: CNN-based age prediction
- **Gender Classification**: CNN-based gender prediction

## Project Structure

```
yolo_cnn_pj/
├── yolo.py                 # Main application file
├── requirements.txt        # Python dependencies
├── haarcascade_frontalface_default.xml  # Face detection model
├── age_deploy.prototxt    # Age classification model architecture
├── age_net.caffemodel     # Age classification model weights
├── gender_deploy.prototxt # Gender classification model architecture
├── gender_net.caffemodel  # Gender classification model weights
├── coco.txt              # COCO dataset class names
└── weights/              # YOLO model weights directory
    └── yolov8n.pt       # YOLOv8 nano model
```

## Requirements

- Python 3.7+
- OpenCV
- PyTorch
- NumPy
- Caffe (for age/gender models)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd yolo_cnn_pj
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required model files:
   - The large model files (`.caffemodel`, `.pt`) are not included in this repository due to size constraints
   - Download them separately and place them in the appropriate directories

## Usage

Run the main application:
```bash
python yolo.py
```

## Model Files

**Note**: The following large model files are not included in this repository:
- `age_net.caffemodel` (~44MB)
- `gender_net.caffemodel` (~44MB)
- `yolov8n.pt` (in weights/ directory)

These files need to be downloaded separately and placed in the project directory.

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here] 