import whisper

model = whisper.load_model("small")  # choose model based on resources

def transcribe_file(filename: str):
    # run whisper with timestamps
    result = model.transcribe(filename, word_timestamps=False)  # returns 'segments' with start/end & text
    # result['text'] - full transcript
    # result['segments'] - list of {'start','end','text'}
    transcript = result.get("text", "")
    segments = result.get("segments", [])
    return transcript, segments
