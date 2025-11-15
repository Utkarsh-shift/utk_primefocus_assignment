import os
import re
import argparse
import requests
from dotenv import load_dotenv
from datetime import timedelta
import whisper

# ---------------------------------------
# Load .env
# ---------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ---------------------------------------
# LLM CALLERS
# ---------------------------------------
def generate_transcript_whisper(video_path, whisper_model="small"):
    """
    Generate transcript text from video/audio file using OpenAI Whisper (local).
    Saves no files, returns transcript text as a single string.
    """
    print(f"[Whisper] Loading model: {whisper_model}")
    model = whisper.load_model(whisper_model)

    print(f"[Whisper] Transcribing: {video_path}")
    result = model.transcribe(video_path)

    # join all segments into a clean paragraph
    text = "\n".join([seg["text"].strip() for seg in result.get("segments", [])])
    print("[Whisper] Transcript generated successfully.")
    return text

def call_openai(prompt, model="gpt-4o-mini"):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY missing in .env")
    import openai
    openai.api_key = OPENAI_API_KEY
    resp = openai.chat.completions.create(
        model=model,
        messages=[{"role":"user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

import json
import requests

def call_ollama(prompt, model="llama3.1"):
    url = f"{OLLAMA_HOST}/api/generate"
    data = { "model": model, "prompt": prompt }
    
    r = requests.post(url, json=data, stream=True)
    output = ""

    for line in r.iter_lines():
        if not line:
            continue
        try:
            obj = json.loads(line.decode("utf-8"))
            if "response" in obj:
                output += obj["response"]
        except json.JSONDecodeError:
            # skip malformed lines
            continue
    
    return output.strip()


# ---------------------------------------
# Subtitle processing helpers
# ---------------------------------------

def chunk_text_smart(text, max_chars=42):
    """
    Splits text into lines <= max_chars.
    Avoids breaking pronoun-verb, determiner-noun etc.
    Very simplified logic: prefer splitting at punctuation or spaces.
    """
    tokens = text.split()

    lines = []
    current = ""

    for tok in tokens:
        if len(current) + len(tok) + 1 <= max_chars:
            current += (" " + tok if current else tok)
        else:
            # push line
            lines.append(current)
            current = tok

    if current:
        lines.append(current)

    # if more than 2 lines, merge cleanly
    if len(lines) > 2:
        merged = [" ".join(lines[:len(lines)//2]), " ".join(lines[len(lines)//2:])]
        return merged[:2]

    return lines


def format_srt_block(index, start, end, original, translated):
    """
    Produce SRT block:
    1
    00:00:00,000 --> 00:00:05,000
    English line
    Translated line
    """
    return (
        f"{index}\n"
        f"{start} --> {end}\n"
        f"{original}\n"
        f"{translated}\n\n"
    )


def seconds_to_timestamp(sec):
    td = timedelta(seconds=sec)
    s = str(td)
    if "." not in s:
        s += ".000000"
    hrs, mins, rest = s.split(":")
    secs, micros = rest.split(".")
    return f"{int(hrs):02}:{int(mins):02}:{int(secs):02},{int(micros[:3]):03}"

# ---------------------------------------
# Main LLM subtitle generator
# ---------------------------------------

def generate_subtitles(text, llm_type="ollama", model="llama3.1", target_lang="Hindi"):
    prompt = f"""
You are a subtitle formatter.

Rules:
- Rewrite the transcript into clear subtitle sentences.
- DO NOT add or remove meaning.
- Break into short readable subtitle chunks.
- Avoid splitting pronoun–verb, determiner–noun, preposition–phrase, conjunction–phrase.
- Max 2 lines per subtitle, max 42 characters per line.
- Provide ONLY the cleaned English subtitle sentences, one per line.
Transcript:
{text}
"""
    if llm_type == "openai":
        cleaned = call_openai(prompt, model=model)
    else:
        cleaned = call_ollama(prompt, model=model)

    english_subs = [line.strip() for line in cleaned.split("\n") if line.strip()]
    return english_subs


def translate_lines(lines, llm_type="ollama", model="llama3.1", target_lang="Hindi"):
    final = []
    for line in lines:
        prompt = f"""
Translate the following subtitle line into {target_lang}.
Only give the translation, no explanation.

Line:
{line}
"""
        if llm_type == "openai":
            tr = call_openai(prompt, model=model)
        else:
            tr = call_ollama(prompt, model=model)

        final.append(tr.strip())
    return final

# ---------------------------------------
# Main pipeline
# ---------------------------------------

def main(args):
    # ------------------------------------------------------------------
    # If transcript file is provided → read it
    # Else if --video is provided → generate transcript using Whisper
    # ------------------------------------------------------------------
    if args.transcript:
        print(f"Reading transcript from file: {args.transcript}")
        with open(args.transcript, "r", encoding="utf-8") as f:
            original_text = f.read()
    elif args.video:
        print("No transcript provided. Generating transcript using Whisper...")
        original_text = generate_transcript_whisper(
            args.video,
            whisper_model=args.whisper_model
        )
    else:
        raise ValueError("You must provide either --transcript or --video")

    # Step 1: Get cleaned English subtitle lines
    english = generate_subtitles(
        original_text,
        llm_type=args.llm_type,
        model=args.model,
        target_lang=args.target_lang
    )

    # Step 2: Translate each line
    translated = translate_lines(
        english,
        llm_type=args.llm_type,
        model=args.model,
        target_lang=args.target_lang
    )

    # Step 3: SRT generation
    srt_out = ""
    time_cursor = 0
    block_length = 4  # assume 4 sec per subtitle

    for idx, (eng, tr) in enumerate(zip(english, translated), start=1):

        # break English line into up to 2 lines
        lines = chunk_text_smart(eng, max_chars=args.max_chars)

        original_fmt = "\n".join(lines)
        trans_fmt = tr

        start = seconds_to_timestamp(time_cursor)
        end = seconds_to_timestamp(time_cursor + block_length)
        time_cursor += block_length

        srt_out += format_srt_block(idx, start, end, original_fmt, trans_fmt)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(srt_out)

    print(f"Generated SRT saved to: {args.output}")


# ---------------------------------------
# CLI
# ---------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", help="Input transcript .txt")
    parser.add_argument("--output", default="subtitles.srt", help="Output SRT file")
    parser.add_argument("--target-lang", default="Hindi")
    parser.add_argument("--llm-type", default="ollama", choices=["ollama","openai"])
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--max-chars", type=int, default=42)
    parser.add_argument("--video", help="Video file to auto-generate transcript with Whisper")
    parser.add_argument("--whisper-model", default="small", help="Whisper model: tiny/small/medium/large")

    args = parser.parse_args()

    main(args)
