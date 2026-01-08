# Vision Tracking Project 🎯

Real-time object tracking project using OpenCV and Python. Implements multiple tracking algorithms to compare performance and accuracy across different scenarios.

## 📋 Features

- **Multiple tracking algorithms**: CSRT, KCF, MOSSE, MedianFlow
- **Single and multi-object tracking**: Track one or multiple objects simultaneously
- **Real-time processing**: Compatible with webcam and video files
- **Comparative analysis**: Performance and accuracy metrics
- **Intuitive interface**: Visual object selection for tracking

## 🚀 Installation

### Requirements

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. Clone the repository:
```bash
git clone https://github.com/your-username/vision-tracking-project.git
cd vision-tracking-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Single object tracking

```bash
python src/main.py --video videos/sample.mp4 --tracker CSRT
```

### Multi-object tracking

```bash
python src/main.py --video videos/sample.mp4 --multi --num-objects 3 --tracker KCF
```

### Real-time webcam tracking

```bash
python src/main.py --camera 0 --tracker MOSSE
```

### Algorithm comparison

```bash
python src/main.py --video videos/sample.mp4 --tracker CSRT --kalman
```

## 📊 Available Algorithms

| Algorithm | Speed | Accuracy | Recommended Use |
|-----------|-------|----------|----------------|
| **CSRT** | Slow | High | Objects with scale changes |
| **KCF** | Medium | Medium-High | General use, good balance |
| **MOSSE** | Very Fast | Medium | Real-time, limited resources |
| **MedianFlow** | Fast | Medium | Predictable movements |

## 📁 Project Structure

```
vision-tracking-project/
├── src/                    # Source code
│   ├── models/             # Tracking implementations
│   │   ├── detector.py     # Object detection (YOLO, HOG)
│   │   ├── tracker.py      # Tracking algorithms
│   │   ├── kalman_filter.py # Kalman filtering
│   │   ├── speed_calculator.py # Speed estimation
│   │   └── video_processor.py # Video I/O
│   ├── controllers/        # Application controllers
│   ├── utils/              # Utilities
│   └── main.py             # Main entry point
├── examples/               # Usage examples
├── notebooks/              # Jupyter notebooks
├── videos/                 # Test videos
├── results/                # Generated results
└── docs/                   # Documentation
```

## 📖 Examples

Check the `examples/` folder for detailed examples:

- `single_object.py`: Basic single object tracking
- `multi_object.py`: Multi-object tracking
- `webcam_tracking.py`: Real-time webcam tracking

## 🧪 Testing

Run tests:
```bash
python -m pytest tests/
```

## 🤝 Contributing

Contributions are welcome. Please:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is under the MIT License. See `LICENSE` file for more details.

## 🔗 References

- [OpenCV Tracking API](https://docs.opencv.org/4.x/d9/df8/group__tracking.html)
- [CSRT Paper](https://arxiv.org/abs/1611.08461)
- [KCF Paper](https://arxiv.org/abs/1404.7584)

## 👨‍💻 Author

Developed as part of Computer Vision project

---

⭐ If you found this project useful, consider giving it a star on GitHub!
