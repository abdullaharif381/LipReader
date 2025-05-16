# LipNet Transcription Web Application

A web application that transcribes mute videos by reading the lips of people speaking in English using a pre-trained LipNet model.

## 🧠 Overview

This Flask-based web application performs lip-reading transcription on silent videos using a pre-trained LipNet model. The app allows users to upload `.mpg` format videos, processes them frame-by-frame, and displays the predicted transcription.

## ✨ Features

- 🎥 Upload silent `.mpg` videos for transcription
- 🤖 Process videos using a pre-trained LipNet model
- 📋 Display transcription results in a clean, modern interface
- ⚠️ Error handling for invalid files or missing model

## 🛠️ Requirements

- Python 3.8+
- Flask
- TensorFlow 2.x
- OpenCV
- NumPy

You can install all dependencies via:

```bash
pip install -r requirements.txt
```

## 🚀 Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/your-username/lipnet-transcription-app.git
   cd lipnet-transcription-app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask app:**
   ```bash
   flask run
   ```

4. **Access the application:**
   Open your browser and navigate to http://127.0.0.1:5000

## 📁 Project Structure

```
lipnet-transcription-app/
│
├── app.py                # Main Flask application
├── templates/
│   └── index.html        # HTML front-end
├── static/
│   ├── css/
│   │   └── style.css     # Custom CSS styles
│   └── js/
│       └── script.js     # JavaScript functionality
├── model/
│   └── lipnet_model.h5   # Pre-trained LipNet weights
├── utils/
│   └── preprocess.py     # Video preprocessing utilities
└── requirements.txt      # Python dependencies
```

## 🧪 Usage Notes

- Ensure your `.mpg` videos are clear with well-centered faces for optimal results
- The pre-trained model expects specific video dimensions and frame rates
- For best results, videos should have good lighting and minimal background noise

## 📄 License

MIT License © 2025

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
