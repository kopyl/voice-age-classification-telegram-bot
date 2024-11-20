import io
import pydub


async def get_voice_pydub(update):
    voice_file_telegram_reference = await update.message.voice.get_file()
    voice_bytearray = await voice_file_telegram_reference.download_as_bytearray()
    file_bytes = io.BytesIO(voice_bytearray)
    pydub_audio = pydub.AudioSegment.from_file(file_bytes)
    return pydub_audio
