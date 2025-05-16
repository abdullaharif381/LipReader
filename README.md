## 🧪 Usage Notes & Tips

- **Video Quality Requirements**:
  - Clear frontal face with visible mouth region
  - Consistent lighting with minimal shadows
  - Limited head movement for best accuracy
  - Default expecting video dimensions cropped to mouth region (46x140 pixels)
  - Optimal frame rate: 25fps (model was trained at this rate)

- **Performance Considerations**:
  - GPU acceleration significantly improves processing speed
  - For CPU-only environments, expect slower inference times
  - Batch processing is more efficient for multiple videos

- **Limitations**:
  - Limited vocabulary to training dataset words
  - English language only in current implementation
  - Reduced accuracy with extreme facial angles or poor lighting

## 📚 Resources

This project builds upon research and implementations from:
- [LipNet: End-to-End Sentence-level Lipreading](https://arxiv.org/abs/1611.01599) - Original research paper

## 📄 License

MIT License © 2025

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for bugs, questions, or feature requests.# LipNet Transcription Web Application
A deep learning application that transcribes mute videos by reading the lips of people speaking in English using a custom LipNet architecture.

## 🧠 Overview

This application performs automatic lip-reading transcription on silent videos using a 3D CNN + Bidirectional LSTM architecture. The system processes video frames to recognize spoken words without audio, making it useful for accessibility, silent video understanding, and situations where audio is unavailable.

## ✨ Features

- 🎥 Upload and process silent `.mpg` video files
- 🤖 Advanced lip-reading with 3D CNN + Bidirectional LSTM architecture
- 📋 Character-level transcription with CTC loss optimization
- 🔄 Frame-by-frame video processing pipeline
- ⚠️ Robust error handling for various input scenarios

## 🛠️ Technologies & Requirements

- **Python 3.8+**
- **Deep Learning Framework**: TensorFlow 2.x
- **Computer Vision**: OpenCV for video processing
- **Web Framework**: Flask for deployment interface
- **Data Processing**: NumPy for numerical operations
- **GPU Support**: CUDA-compatible for faster inference

Additional dependencies:
- matplotlib (visualization)
- imageio (additional image processing)
- gdown (for model downloading)

```bash
pip install -r requirements.txt
```

## 🔄 Data Pipeline & Preprocessing

The LipNet data pipeline includes several key preprocessing steps:

1. **Video Frame Extraction**: 
   - Extract frames at 25fps from `.mpg` videos
   - Convert frames to grayscale for simplified processing
   - Crop to mouth region (rows 190-236, columns 80-220)
   - Apply mean and standard deviation normalization

2. **Text Processing**:
   - Character-level tokenization with a vocabulary of lowercase letters, numbers, and special characters
   - Convert between text and numerical representations using StringLookup layers
   - Handle alignment files that map frames to phonetic sequences

3. **Data Augmentation**:
   - Batch processing with padded sequences
   - Dataset shuffling and prefetching for training efficiency

The pipeline uses TensorFlow's `tf.data` API for efficient data loading and preprocessing, with `tf.py_function` wrappers to integrate custom Python functions into the TensorFlow graph.

## 🧠 Model Architecture

The LipNet architecture combines spatial and temporal processing:

```
Input Video Frames → 3D CNNs → Bidirectional LSTMs → Dense Output → CTC Decoder
```

### Detailed Architecture:

1. **3D Convolutional Layers**:
   - Input shape: (75, 46, 140, 1) - 75 frames of 46x140 grayscale images
   - Three convolutional blocks with increasing filter depth (128 → 256 → 75)
   - Each block: Conv3D + ReLU + MaxPool3D
   - Spatial downsampling through MaxPool3D operations

2. **Recurrent Layers**:
   - Reshape output to sequence format (75 timesteps)
   - Two Bidirectional LSTM layers (128 units each)
   - Dropout (0.5) between LSTM layers for regularization

3. **Output Layer**:
   - Dense layer with softmax activation
   - Output size matches vocabulary plus blank character (for CTC)

4. **Loss Function**:
   - Connectionist Temporal Classification (CTC) loss
   - Handles variable-length sequence alignment without exact frame-level labels

## 🚀 Training Process

The model is trained with:
- Adam optimizer with 0.0001 learning rate
- Learning rate scheduling (exponential decay after 30 epochs)
- Checkpoint saving after each epoch
- Custom callback to monitor transcription quality during training

## 🚀 Installation & Usage

1. **Clone this repository:**
   ```bash
   git clone https://github.com/abdullaharif381/LipReader.git
   cd LipReader
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your LipNet model in models folder**
   ```bash
   save the lipnet.keras model in /models
   ```

4. **Run the Flask app:**
   ```bash
   flask run
   ```

5. **Access the application:**
   Open your browser and navigate to http://127.0.0.1:5000
   
6. **Using the application:**
   - Upload a silent `.mpg` video file that clearly shows a person's lips
   - The model processes the video frames and predicts the spoken text
   - Results are displayed in the web interface

## 📁 Project Structure

```
lipnet-transcription-app/
│
├── app.py                        # Main Flask application
├── model/
│   ├── lipnet.keras        # your model goes here
├── templates/
│   └── index.html                # HTML front-end
├── static/
│   ├── css/
│   │   └── style.css             # Custom CSS styles
│   └── js/
│       └── script.js             # JavaScript functionality
├── notebooks/
│   └── syncvsr-preprocessing.ipynb
|   └── syncvsr-training.ipynb
|   └── lipnet-training.ipynb
└── requirements.txt              # Python dependencies
```


## Team members:
[Tahmooras Khan - Linkedin](https://www.linkedin.com/in/tahmooras-khan-8341452a1).
[Ibtehaj Ali - Github](https://github.com/Ibtehaj778)
