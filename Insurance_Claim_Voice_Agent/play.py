import soundfile as sf
import sounddevice as sd
import threading
import queue
import time

audio_queue = queue.Queue()

def audio_worker():
    while True:
        file = audio_queue.get()
        if file is None:
            break
        data, rate = sf.read(file)
        sd.play(data, rate)
        sd.wait()
        audio_queue.task_done()

def start_audio_thread():
    threading.Thread(target = audio_worker, daemon = True).start()

def enqueue_audio(file):
    audio_queue.put(file)

def stop_audio_thread():
    audio_queue.put(None)
