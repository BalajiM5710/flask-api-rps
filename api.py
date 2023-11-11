from flask import Flask, request, jsonify
import librosa
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
import pickle

with open('model5.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


disease_names = ["COPD", "Bronchiolitis", "Pneumonia", "URTI", "Healthy"]

def stretch(data, rate):
    data_stretched = librosa.effects.time_stretch(data, rate=rate)
    return data_stretched

def extract_features(audio_path):
    data_x, sampling_rate = librosa.load(audio_path)
    data_x = stretch(data_x, 1.2)
    features = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=52).T, axis=0)
    features = features.reshape(1, 1, 52)  # Expand dimensions to match the model input shape
    return features

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Assuming the audio file is sent as 'file' in the POST request
        audio_file = request.files['file']
        audio_file.save('/tmp/uploaded_audio.wav')  # Save the uploaded file
        features = extract_features('/tmp/uploaded_audio.wav')
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction)
        
        if 0 <= predicted_class < len(disease_names):
            predicted_disease = disease_names[predicted_class]
        else:
            predicted_disease = "Unknown Disease"

        return jsonify({'predicted_class': int(predicted_class), 'predicted_disease': predicted_disease, 'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
from flask import Flask, request, jsonify
import librosa
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('model5.pkl')  # Load your model here

disease_names = ["COPD", "Bronchiolitis", "Pneumonia", "URTI", "Healthy"]

def stretch(data, rate):
    data_stretched = librosa.effects.time_stretch(data, rate=rate)
    return data_stretched

def extract_features(audio_path):
    data_x, sampling_rate = librosa.load(audio_path)
    data_x = stretch(data_x, 1.2)
    features = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=52).T, axis=0)
    features = features.reshape(1, 1, 52)  # Expand dimensions to match the model input shape
    return features

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Assuming the audio file is sent as 'file' in the POST request
        audio_file = request.files['file']
        audio_file.save('/tmp/uploaded_audio.wav')  # Save the uploaded file
        features = extract_features('/tmp/uploaded_audio.wav')
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction)
        
        if 0 <= predicted_class < len(disease_names):
            predicted_disease = disease_names[predicted_class]
        else:
            predicted_disease = "Unknown Disease"

        return jsonify({'predicted_class': int(predicted_class), 'predicted_disease': predicted_disease, 'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
