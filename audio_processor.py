def trim_audio(pydub_audio, from_sec=0, to_sec=10):
    from_msec = from_sec * 1000 #Works in milliseconds
    to_msec = to_sec * 1000
    trimmed_audio = pydub_audio[from_msec:to_msec]
    return trimmed_audio