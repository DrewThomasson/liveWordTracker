import sqlite3
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from flask import Flask, render_template, make_response
from datetime import datetime
from queue import Queue
import threading
import argparse
import matplotlib
matplotlib.use('Agg')  # Needed for headless environments
import matplotlib.pyplot as plt
import io
import base64

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8080, help='Web server port')
parser.add_argument('--model', default='small', help='Whisper model size')
parser.add_argument('--buffer', type=int, default=5, help='Audio buffer size in seconds')
parser.add_argument('--sr', type=int, default=16000, help='Sample rate')
args = parser.parse_args()

# Database setup
def init_db():
    conn = sqlite3.connect('audio_transcriptions.db', isolation_level=None)  # Auto-commit enabled
    conn.execute('''CREATE TABLE IF NOT EXISTS transcriptions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME,
                  text TEXT,
                  volume REAL,
                  hour INTEGER,
                  day_of_week INTEGER,
                  date DATE)''')
    conn.commit()
    conn.close()

init_db()

# Audio processing
class AudioProcessor:
    def __init__(self):
        self.sample_rate = args.sr
        self.buffer_duration = args.buffer
        self.buffer_size = self.sample_rate * self.buffer_duration
        self.audio_queue = Queue()
        self.volume_window = int(1.0 * self.sample_rate)  # 1 second volume window

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio error: {status}")
        self.audio_queue.put(indata.copy())

    def get_volume(self, audio_chunk):
        rms = np.sqrt(np.mean(np.square(audio_chunk)))
        return 20 * np.log10(rms) if rms > 0 else -100

    def start_recording(self):
        print("Starting audio recording...")
        with sd.InputStream(callback=self.audio_callback,
                          channels=1,
                          samplerate=self.sample_rate,
                          blocksize=self.volume_window):
            buffer = np.array([])
            while True:
                chunk = self.audio_queue.get()
                buffer = np.concatenate((buffer, chunk.flatten()))
                
                if len(buffer) >= self.buffer_size:
                    yield buffer[:self.buffer_size]
                    buffer = buffer[self.buffer_size:]

# Transcription model
class Transcriber:
    def __init__(self):
        self.model = WhisperModel(args.model, device="cpu", compute_type="int8")
        print(f"Loaded Whisper model '{args.model}' with VAD filtering")

    def transcribe(self, audio_np):
        try:
            audio_np = audio_np.astype(np.float32)
            audio_np = audio_np / np.max(np.abs(audio_np)) if np.max(np.abs(audio_np)) > 0 else audio_np

            segments, info = self.model.transcribe(
                audio_np,
                language='en',
                vad_parameters={
                    'threshold': 0.95,
                    'min_silence_duration_ms': 1500
                }
            )
            
            texts = [seg.text.strip() for seg in segments if seg.text.strip()]
            if texts:
                print(f"Detected speech segments: {texts}")
                return ' '.join(texts)
            
            return ""
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

# Web server
app = Flask(__name__)

@app.route('/')
@app.route('/')
def dashboard():
    conn = sqlite3.connect('audio_transcriptions.db', isolation_level=None)
    cursor = conn.cursor()

    print("\n---- DEBUG: Fetching volume data ----")
    cursor.execute('''
        SELECT strftime('%H:%M', timestamp) as time, AVG(volume) 
        FROM transcriptions 
        WHERE timestamp >= datetime('now', '-1 day')  -- Fetch last 24 hours
        GROUP BY strftime('%H:%M', timestamp)
    ''')
    volume_data = cursor.fetchall()
    print("Fetched Volume Data:", volume_data)

    print("\n---- DEBUG: Fetching word frequency data ----")
    cursor.execute('''
        SELECT text FROM transcriptions 
        WHERE date = (SELECT MAX(date) FROM transcriptions) AND text != ''
    ''')
    words = cursor.fetchall()
    print("Raw Words from DB:", words)

    conn.close()

    if not words:
        print("WARNING: No words fetched from the database!")

    word_freq = {}
    for (text,) in words:
        for word in text.split():
            cleaned_word = word.lower().strip(",.!?")
            if cleaned_word:
                word_freq[cleaned_word] = word_freq.get(cleaned_word, 0) + 1

    print("Final Word Frequency Data:", word_freq)

    # Generate graph
    top_words = sorted(word_freq.items(), key=lambda x: -x[1])[:15]

    plt.figure(figsize=(10, 5))
    plt.barh([w[0] for w in top_words], [w[1] for w in top_words])
    plt.title('Most Frequent Words Today')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    response = make_response(render_template('dashboard.html', volume_data=volume_data, plot_url=plot_url))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    
    return response


# Main application
def main():
    processor = AudioProcessor()
    transcriber = Transcriber()
    
    server_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': args.port, 'use_reloader': False})
    server_thread.daemon = True
    server_thread.start()
    
    MIN_VOLUME_THRESHOLD = -40  # dB, adjust based on noise environment
    
    for audio_buffer in processor.start_recording():
        try:
            volume = processor.get_volume(audio_buffer)

            # Ignore low-volume recordings (assumed silence/noise)
            if volume < MIN_VOLUME_THRESHOLD:
                print(f"[{datetime.now()}] Skipping transcription (Low volume: {volume:.1f}dB)")
                continue  

            text = transcriber.transcribe(audio_buffer)
            
            now = datetime.now()
            status = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Volume: {volume:.1f}dB"

            if text:
                status += f" | Text: {text}"
                
                conn = sqlite3.connect('audio_transcriptions.db', isolation_level=None)
                conn.execute('''
                    INSERT INTO transcriptions (timestamp, text, volume, hour, day_of_week, date)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (now, text, volume, now.hour, now.weekday(), now.date()))
                conn.commit()
                conn.close()

            print(status)
            
        except Exception as e:
            print(f"Error processing audio: {e}")

if __name__ == '__main__':
    main()

