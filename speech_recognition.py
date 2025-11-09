'''
Author

Selim Ben Haj Braiek
🎓 Master’s student in Data Science and Artificial Intelligence
📍 Budapest University of Technology and Economics (BME)

Description: This script demonstrates various speech recognition techniques using Python libraries.
'''


import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
import soundfile as sf
import speech_recognition as sr
from jiwer import wer, cer
from IPython.display import Audio, display
import whisper
import csv, os
from gtts import gTTS
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 1 Load audio
audio_signal, sample_rate = librosa.load('speech_01.wav', sr=None)
plt.figure(figsize=(12, 4))
librosa.display.waveshow(audio_signal, sr=sample_rate)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# 2. Play audio
display(Audio('speech_01.wav'))

# 3. Transcribe using Google Speech Recognition
recognizer = sr.Recognizer()

def transcribe_audio(file_path):
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        print("Google Transcription:", text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
        return ""
    except sr.RequestError as e:
        print(f"Request error: {e}")
        return ""

file_path = 'speech_01.wav'
transcribed_text = transcribe_audio(file_path)

# 4. Ground truth
ground_truth = """My name is Ivan and I am excited to have you as part of our learning community! 
Before we get started, I’d like to tell you a little bit about myself. I’m a sound engineer turned data scientist,
curious about machine learning and Artificial Intelligence. My professional background is primarily in media production,
with a focus on audio, IT, and communications."""

# 5. Calculate WER / CER
calculated_wer = wer(ground_truth.lower(), transcribed_text.lower())
calculated_cer = cer(ground_truth.lower(), transcribed_text.lower())
print(f"WER: {calculated_wer:.4f} | CER: {calculated_cer:.4f}")

# 6. Apply pre-emphasis
signal_filtered = librosa.effects.preemphasis(audio_signal, coef=0.97)
sf.write('filtered_speech_01.wav', signal_filtered, sample_rate)

# 7. Compare spectrograms
S = librosa.stft(signal_filtered)
S_dB = librosa.amplitude_to_db(abs(S), ref=np.max)
plt.figure(figsize=(12, 4))
librosa.display.specshow(S_dB, sr=sample_rate, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram after pre-emphasis')
plt.show()

# 8. Transcribe the filtered file again
transcribed_text_preemphasis = transcribe_audio('filtered_speech_01.wav')
calculated_wer = wer(ground_truth.lower(), transcribed_text_preemphasis.lower())
calculated_cer = cer(ground_truth.lower(), transcribed_text_preemphasis.lower())
print(f"Filtered WER: {calculated_wer:.4f} | CER: {calculated_cer:.4f}")

# 9. Whisper transcription
model = whisper.load_model("base")
result = model.transcribe(file_path)
transcribed_text_whisper = result["text"]
print("\nWhisper Transcription:", transcribed_text_whisper)

# 10. WER / CER for Whisper
calculated_wer = wer(ground_truth.lower(), transcribed_text_whisper.lower())
calculated_cer = cer(ground_truth.lower(), transcribed_text_whisper.lower())
print(f"Whisper WER: {calculated_wer:.4f} | CER: {calculated_cer:.4f}")

# 11. Transcribe all .wav files in a directory using Whisper
def transcribe_directory_whisper(directory_path):
    transcriptions = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(directory_path, file_name)
            result = model.transcribe(file_path)
            transcriptions.append({
                "file_name": file_name,
                "transcription": result["text"]
            })
    return transcriptions

directory_path = "C:/Users/PC/Downloads/Speech Recognition/Recordings"
transcriptions = transcribe_directory_whisper(directory_path)

# 12. Save all transcriptions to CSV
with open("transcriptions.csv", mode="w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Track Number", "File Name", "Transcription"])
    for number, t in enumerate(transcriptions, start=1):
        writer.writerow([number, t["file_name"], t["transcription"]])

# 13. Text-to-Speech
text = """Thank you for taking the time to watch our course on speech recognition!
This concludes the final lesson of this section. See you soon!"""

tts = gTTS(text=text, lang='en')
tts.save("output.mp3")
os.system("start output.mp3")  # For Windows; use "afplay output.mp3" on macOS or "xdg-open output.mp3" on Linux
