def call_asr(recorder):
    print("Start speaking...")
    text = recorder.text()
    print("Transcribed:", text)
    return text