import numpy as np
import tensorflow as tf
import pyaudio
import librosa
import os

out_rate = 22050
voice_duration = 5
p = pyaudio.PyAudio()
in_rate = int(p.get_device_info_by_index(p.get_default_input_device_info()['index'])['defaultSampleRate'])
target_size = voice_duration * out_rate
CHUNK = in_rate * voice_duration
stream = p.open(format=pyaudio.paFloat32, channels=1,
                rate=in_rate, input=True, frames_per_buffer=CHUNK)

model = tf.keras.models.load_model('trained model/customcrnn.keras')


def random_crop(data, center_crop=False):
    N = data.shape[0]
    if N == target_size:
        return data
    if N < target_size:
        tot_pads = target_size - N
        left_pads = int(np.ceil(tot_pads / 2))
        right_pads = int(np.floor(tot_pads / 2))
        return np.pad(data, [left_pads, right_pads], mode='constant')
    if center_crop:
        from_ = int((N / 2) - (target_size / 2))
    else:
        from_ = np.random.randint(0, np.floor(N - target_size))
    to_ = from_ + target_size
    return data[from_:to_]


def normalize(features):
    return (features - np.mean(features, axis=0)) / np.std(features, axis=0)


def preprocess_audio(data):
    data = librosa.resample(data.astype(float), orig_sr=in_rate, target_sr=out_rate)
    data = random_crop(data)
    data = librosa.feature.mfcc(y=data, sr=out_rate)
    data = normalize(data)
    data = np.expand_dims(data, axis=0)
    return data


if __name__ == '__main__':
    while True:
        voice = stream.read(CHUNK)
        np_buff = np.frombuffer(voice, dtype=np.float32)
        processed_data = preprocess_audio(np_buff)
        pred = model.predict(processed_data)
        os.system('clear')
        percent = round(pred[0][0][0] * 100, 2)
        if percent < 45.:
            print("Gender : Male %", percent)
        elif percent > 55.:
            print("Gender : Female %", percent)
        else:
            print("Gender : Unknown")
        print("Age: ", pred[1][0][0])
