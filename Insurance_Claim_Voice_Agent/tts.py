import os
def call_tts(tts_model, text, count):
    os.makedirs("./audio", exist_ok=True)
    filename = f"./audio/chunk_{count}.wav"
    tts_model.tts_to_file(text=text, speaker="p228", file_path=filename)
    return filename
