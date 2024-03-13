# Real-time Gender and Age Recognition from Audio

This project aims to recognize the gender and age of speakers from speech audio samples. It utilizes machine learning techniques and signal processing to achieve this task. Then enables real-time gender and age recognition from audio input using the trained model. It utilizes PyAudio for audio input processing and TensorFlow for model inference.

### Authors
- [@Pooya Nasiri](https://github.com/PooyaNasiri)
- [@Bahador Mirzazadeh](https://github.com/Baha2rM98)



## Requirements
Make sure you have the following additional libraries installed:
- pyaudio
- matplotlib
- plotly
- tensorflow
- tqdm
- pandas
- numpy
- librosa

## Setup
- Define constants such as output sample rate, voice duration, and chunk size.
- Initialize PyAudio for audio input streaming.
- Load the pre-trained CustomCRNN model.

## Dataset
The dataset used in this project is derived from Mozilla Common Voice, which provides a large collection of speech data contributed by volunteers. The dataset has been preprocessed to ensure balanced representation of male and female genders for improved model performance.
[Mozilla Common Voice](https://commonvoice.mozilla.org/en/)

## Data Preprocessing
- Implement functions for random cropping and normalization.
- Preprocess the incoming audio data using librosa to match the input requirements of the model.

## Model Architecture
- Implement a CustomCRNN (Convolutional Recurrent Neural Network) for feature extraction.
- Define output layers for gender and age prediction.

## Model Training
- Compile the model with appropriate loss functions, metrics, and optimizer.
- Train the model using the prepared datasets.

## Real-time Inference
- Continuously read audio input from the microphone.
- Preprocess the audio data.
- Perform inference using the loaded model.
- Display the predicted gender and age in real-time.

## Conclusion
This project provides a comprehensive solution for gender and age recognition from speech audio samples. By combining data preprocessing, model training, and real-time inference, it offers a complete pipeline for analyzing audio data. Further enhancements can be made to improve model accuracy and real-time performance.
