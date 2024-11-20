import os
from telegram import ForceReply, Update, File
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram_file_handler import get_voice_pydub
from audio_processor import trim_audio
from model import predict_age_from_pydub_audio


BOT_TOKEN = os.getenv('BOT_TOKEN')


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_html(
        "Send a voice message up to 10s (or else i crop it) "
        "and i'll try to predict your age"
    )


async def process_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pydub_audio = await get_voice_pydub(update)
    pydub_audio_trimmed = trim_audio(pydub_audio, from_sec=0, to_sec=10)
    await update.message.reply_text("Let me think...")
    age = predict_age_from_pydub_audio(pydub_audio_trimmed)
    await update.message.reply_text(f"You sound like {age}")


def main() -> None:
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VOICE, process_voice))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()