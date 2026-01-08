# 👁️ Vision Tracking Project

A powerful computer vision application for real-time object tracking using OpenCV and Python. It supports multiple tracking algorithms and now features a modern **Web Interface** for easy usage.

## ✨ Features

- **🆕 Web Interface**: Modern, easy-to-use dashboard built with Streamlit.
- **Multiple Tracking Algorithms**: CSRT, KCF, MOSSE, MedianFlow, MIL, Boosting.
- **Auto-Detection**: Integrated YOLO object detection to automatically find objects to track.
- **Multi-Object Tracking**: Track multiple objects simultaneously with independent speed calculations.
- **Kalman Filtering**: Advanced motion smoothing and prediction for robust tracking.
- **Real-time Metrics**: Speed estimation (px/s) and trajectory monitoring.

## 🚀 Quick Start (Web Interface)

The easiest way to use the project is via the new Web Dashboard.

### 1. Installation
First, ensure you have Python 3.8+ installed. We recommend using `uv` for fast dependency management, but `pip` works too.

```bash
# Clone the repository
git clone https://github.com/gabzregis030/tracking-vision-artificial.git
cd tracking-vision-artificial

# Install dependencies
pip install -r requirements.txt
# OR if using uv:
uv pip install -r requirements.txt
```

### 2. Launch the Web App
Run the following command to start the interface:

```bash
# Using standard pip/python
streamlit run src/app.py

# Using uv (recommended)
uv run streamlit run src/app.py
```

This will automatically open your browser at `http://localhost:8501`.

### 3. How to Use
1.  **Input Source**: Choose between "Video File" (upload your own) or "Webcam".
2.  **Configuration**:
    *   **Tracker**: `KCF` is recommended for speed/accuracy balance. `CSRT` for high accuracy (slower).
    *   **Auto-Detection**: Enabled by default (uses YOLO to find people/cars/etc automatically).
3.  **Start**: Click **Start Tracking**.

---

## 💻 CLI Usage (Advanced)

If you prefer the command line or need maximum performance without the web overhead, use the Python script directly.

### Basic Commands

**Track objects in a video file:**
```bash
uv run src/main.py --video videos/demo_two_objects.mp4 --tracker KCF
```

**Track using Webcam:**
```bash
uv run src/main.py --camera 0 --tracker KCF
```

**Multi-object Tracking (2 objects):**
```bash
uv run src/main.py --video videos/demo_two_objects.mp4 --multi --num-objects 2
```
*Note: You will be asked to draw a rectangle around each object and press ENTER.*

**Enable Kalman Filter (Smoother movement):**
```bash
uv run src/main.py --video videos/demo_two_objects.mp4 --kalman
```

## 📊 Available Algorithms

| Algorithm | Speed | Accuracy | Recommended Use |
|-----------|-------|----------|----------------|
| **KCF** | ⭐⭐⭐⭐ | ⭐⭐⭐ | **Best General Choice**. Good balance. |
| **CSRT** | ⭐⭐ | ⭐⭐⭐⭐⭐ | High precision, handles occlusion well. Slow. |
| **MOSSE** | ⭐⭐⭐⭐⭐ | ⭐⭐ | Extremely fast. Good for old hardware. |
| **MedianFlow** | ⭐⭐⭐⭐ | ⭐⭐ | Good if motion is predictable and small. |

## 📁 Project Structure

```
vision-tracking-project/
├── src/
│   ├── app.py              # Web Interface Entry Point
│   ├── main.py             # CLI Entry Point
│   ├── controllers/        # Logic controllers
│   └── models/             # Core algorithms (Tracker, Kalman, etc)
├── videos/                 # Place your test videos here
├── requirements.txt        # Project dependencies
└── README.md              # Documentation
```

## 🛠️ Testing

Run the test suite to ensure everything is working:

```bash
uv run pytest tests/
```

## 📝 License

MIT License. See `LICENSE` for details.
