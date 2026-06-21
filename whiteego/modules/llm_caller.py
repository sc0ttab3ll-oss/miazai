import requests


OLLAMA_URL = "http://localhost:11434/api/generate"

# Runtime role split:
# - qwen2.5:7b is the operational default for reflection and dry-run stability
# - gemma3:4b is reserved for optional lyric-specialist generation
LYRIC_MODEL = "gemma3:4b"
REFLECTION_MODEL = "qwen2.5:7b"


def build_prompt(psycho_context: dict) -> str:
    dominant_emotion = psycho_context.get("dominant_emotion", "ismeretlen")
    intensity = psycho_context.get("intensity", "közepes")
    polarity = psycho_context.get("polarity", "semleges")
    narrative_frame = psycho_context.get("narrative_frame", "belső átalakulás")

    return f"""
Írj pontosan 4 rövid magyar dalsort.

Szabályok:
- csak a 4 sort add vissza
- ne számozz
- ne magyarázz
- természetes magyar nyelven írj
- a hangnem legyen melankolikus, de reményt hordozó
- minden sor legyen rövid és tiszta
- a 4 sor együtt egy kis belső elmozdulást mutasson
- az 1. sor nyisson képpel vagy érzéssel
- a 2. sor mélyítse a feszültséget
- a 3. sor hozzon enyhe fordulatot
- a 4. sor adjon halk lezárást vagy reményt
- törekedj rá, hogy a 2. és 4. sor vége hangzásban kapcsolódjon
- kerüld a modoros, túl homályos megfogalmazást

Hangulati brief:
- domináns érzelem: {dominant_emotion}
- intenzitás: {intensity}
- polaritás: {polarity}
- narratív keret: {narrative_frame}
""".strip()


def generate_lyrics(psycho_context: dict, model_name: str = LYRIC_MODEL) -> str:
    prompt = build_prompt(psycho_context)
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException:
        return "A lírai generálás most nem ért célba a megadott időn belül."


def build_psycho_reflection_prompt(psycho_context: dict) -> str:
    dominant_emotion = psycho_context.get("dominant_emotion", "unknown")
    intensity = psycho_context.get("intensity", "medium")
    polarity = psycho_context.get("polarity", "mixed")
    narrative_frame = psycho_context.get("narrative_frame", "inner transformation")

    return (
        f"Describe this inner state in 3 short English sentences. "
        f"No therapy. No diagnosis. No advice. "
        f"Emotion: {dominant_emotion}. "
        f"Intensity: {intensity}. "
        f"Polarity: {polarity}. "
        f"Frame: {narrative_frame}."
    )


def generate_psycho_reflection(
    psycho_context: dict,
    model_name: str = REFLECTION_MODEL,
) -> str:
    prompt = build_psycho_reflection_prompt(psycho_context)
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
            },
            timeout=600,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException:
        return "Qwen reflection is currently unavailable within the allowed time on this machine."
