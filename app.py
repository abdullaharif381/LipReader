import os
import uuid
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, jsonify
import cv2
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['MODEL_PATH'] = 'models/lipnet.keras'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Vocabulary and decoding setup
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# Custom CTC loss for loading the model
def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def load_video(path: str):
    import cv2
    import tensorflow as tf
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        frame = tf.image.rgb_to_grayscale(tf.convert_to_tensor(frame))
        frames.append(frame[190:236, 80:220, :])  # Crop ROI
    cap.release()
    frames = tf.stack(frames)
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    normalized_frames = tf.cast((frames - mean), tf.float32) / std
    return normalized_frames

def load_model():
    """Load the LipNet model if it exists"""
    if not os.path.exists(app.config['MODEL_PATH']):
        return None
    
    try:
        model = tf.keras.models.load_model(
            app.config['MODEL_PATH'], 
            custom_objects={'CTCLoss': CTCLoss}
        )
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_video(video_path: str, model):
    """Run inference on a video file"""
    frames = load_video(video_path)
    input_tensor = tf.expand_dims(frames, axis=0)  # Add batch dimension
    input_tensor = tf.expand_dims(input_tensor, axis=-1)  # Add channel dimension
    
    # Ensure the input tensor has the correct shape
    if input_tensor.shape[1:] != (75, 46, 140, 1):
        # Pad or truncate to 75 frames
        if input_tensor.shape[1] < 75:
            # Pad with zeros
            padding = tf.zeros((1, 75 - input_tensor.shape[1], 46, 140, 1), dtype=tf.float32)
            input_tensor = tf.concat([input_tensor, padding], axis=1)
        else:
            # Truncate to 75 frames
            input_tensor = input_tensor[:, :75, :, :, :]
    
    y_pred = model.predict(input_tensor)
    decoded = tf.keras.backend.ctc_decode(y_pred, input_length=[75], greedy=False)[0][0].numpy()
    prediction = tf.strings.reduce_join(num_to_char(decoded)).numpy().decode('utf-8')
    return prediction.strip()

@app.route('/')
def index():
    """Render the homepage"""
    model_exists = os.path.exists(app.config['MODEL_PATH'])
    return render_template('index.html', model_exists=model_exists)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and transcription"""
    # Check if model exists
    if not os.path.exists(app.config['MODEL_PATH']):
        return jsonify({
            'success': False,
            'error': 'Model not found. Please place lipnet.keras in the models directory.'
        })
    
    # Check if the post request has the file part
    if 'video' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file part in the request'
        })
    
    file = request.files['video']
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        })
    
    # Check file extension
    if not file.filename.lower().endswith('.mpg'):
        return jsonify({
            'success': False,
            'error': 'Only .mpg files are supported'
        })
    
    try:
        # Generate a unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(filepath)
        
        # Load the model
        model = load_model()
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Failed to load the model'
            })
        
        # Process the video and get transcription
        transcription = predict_video(filepath, model)
        
        # Clean up the uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'transcription': transcription
        })
    
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        })

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    app.run(debug=True)