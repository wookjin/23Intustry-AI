import pyaudio
import numpy as np
import wave
import librosa
import os
import keras2onnx
from keras.models import load_model
import onnx
from onnx2pytorch import ConvertModel


def some_preprocessing_function(signal, sr, threshold=0.025):
    """
    Perform preprocessing on audio signal.

    Parameters:
    - signal: Audio time-series
    - sr: Sampling rate
    - threshold: Silence threshold for trimming

    Returns:
    - processed_signal: Preprocessed audio signal
    """

    # 1. Normalize audio data to [-1, 1]
    signal = signal.astype(np.float32) / 32768.0  # assuming 16-bit PCM

    # 2. Remove silent parts (simple approach using threshold)
    mask = np.abs(signal) > threshold
    processed_signal = signal[mask]

    return processed_signal


# 1. 마이크 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5  # 5초 동안 녹음
OUTPUT_FILENAME = "output.wav"

# Keras 모델 로드
keras_model = load_model('./mfcc_y_48x173_0.h5')

# Keras 모델을 ONNX로 변환
onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)

# ONNX 모델 저장
onnx.save_model(onnx_model, "model.onnx")

# ONNX 모델을 PyTorch로 변환
pytorch_model = ConvertModel(onnx_model)

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

frames = []

for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(np.frombuffer(data, dtype=np.int16))

print("Finished recording")

stream.stop_stream()
stream.close()
audio.terminate()

# 2. 녹음된 데이터를 WAV 파일로 저장
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

# 3. 저장된 파일 재생 (Windows의 경우 'start'를 사용, macOS의 경우 'afplay', Linux의 경우 'aplay'를 사용)
os.system("start " + OUTPUT_FILENAME)

# 2. 음성 데이터를 numpy 배열로 변환
signal = np.concatenate(frames, axis=0)
signal = signal.astype(np.float32) / 32768.0  # 16-bit PCM format

# Mel spectrogram 추출 (다른 특징을 사용할 수 있습니다)
mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=RATE)

processed_data = some_preprocessing_function(mel_spectrogram)
#predictions = model.predict(processed_data)