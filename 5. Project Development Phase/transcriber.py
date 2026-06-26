import whisper
import os
import nltk
nltk.download('punkt', quiet=True)

# Load Whisper model
model = whisper.load_model("base")

def transcribe_audio(audio_path: str, language: str = "en") -> dict:
    """
    Transcribe audio file using OpenAI Whisper
    Args:
        audio_path: Path to audio file
        language: 'en' for English, 'te' for Telugu (beta)
    """
    try:
        if language == "te":
            result = model.transcribe(
                audio_path,
                language="te",
                task="transcribe"
            )
        else:
            result = model.transcribe(
                audio_path,
                language="en",
                task="transcribe"
            )

        return {
            "success": True,
            "text": result["text"].strip(),
            "language": language,
            "segments": result.get("segments", [])
        }

    except Exception as e:
        return {
            "success": False,
            "text": "",
            "error": str(e)
        }

def get_word_count(text: str) -> int:
    return len(text.split())

def get_speaking_rate(text: str, duration_seconds: float) -> float:
    words = get_word_count(text)
    minutes = duration_seconds / 60
    return round(words / minutes, 2) if minutes > 0 else 0

def detect_filler_words(text: str) -> dict:
    fillers = ["um", "uh", "like", "you know", "basically",
               "actually", "literally", "so", "right", "okay"]
    text_lower = text.lower()
    filler_count = {}
    total = 0
    for filler in fillers:
        count = text_lower.split().count(filler)
        if count > 0:
            filler_count[filler] = count
            total += count
    return {
        "filler_words": filler_count,
        "total_fillers": total
    }
