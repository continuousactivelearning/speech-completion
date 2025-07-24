import os
import subprocess
import json
import webvtt
import time

VIDEO_LINKS_FILE = "video_links.txt"
TITLES_FILE = "titles.txt"
OUTPUT_DIR = "formatted_transcripts"
START_INDEX = 163
MAX_RETRIES = 1
BASE_WAIT = 1


def format_time_vtt(vtt_time):
    h, m, s = vtt_time.split(":")
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s.replace(",", "."))
    m = int(total_seconds // 60)
    s = int(total_seconds % 60)
    return f"{m}:{s:02d}"


def convert_vtt_to_json(vtt_path, output_path, title):
    transcript = []
    for caption in webvtt.read(vtt_path):
        start = format_time_vtt(caption.start)
        end = format_time_vtt(caption.end)
        text = caption.text.replace("\n", " ").strip()
        if text:
            transcript.append({"start": start, "end": end, "text": text})

    result = {
        "title": title,
        "source": "NPTEL",
        "course": "Discrete Mathematics by Prof Sudarshan Iyengar",
        "transcript": transcript,
        "timestamped": True,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def download_subtitles(url, filename_base):
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  ⏳ Attempt {attempt} for {url}")
        try:
            subprocess.run(
                [
                    "yt-dlp",
                    "--write-auto-sub",
                    "--sub-lang",
                    "en",
                    "--skip-download",
                    "--output",
                    f"{filename_base}.%(ext)s",
                    url,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            vtt_path = f"{filename_base}.en.vtt"
            if os.path.exists(vtt_path):
                return vtt_path

            print("  ❌ Subtitle file not found after download.")
        except subprocess.CalledProcessError as e:
            print("  ⚠️ yt-dlp failed:", e)

        time.sleep(BASE_WAIT * attempt)

    return None


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(VIDEO_LINKS_FILE, "r") as f:
        video_links = [line.strip() for line in f if line.strip()]

    try:
        with open(TITLES_FILE, "r") as f:
            titles = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        titles = [""] * len(video_links)

    for i, url in enumerate(video_links):
        number = START_INDEX + i
        json_path = os.path.join(OUTPUT_DIR, f"{number}.json")

        if os.path.exists(json_path):
            print(f"✅ Skipping {number}.json — already exists")
            number += 1
            continue

        print(f"\n🎬 Processing {url} → {number}.json")
        filename_base = f"{number}"

        vtt_file = download_subtitles(url, filename_base)
        if vtt_file:
            title = titles[i] if i < len(titles) else ""
            try:
                convert_vtt_to_json(vtt_file, json_path, title)
                os.remove(vtt_file)
                print(f"✅ Saved {json_path}")
            except Exception as e:
                print(f"  ❌ Error converting {vtt_file} to JSON: {e}")
        else:
            print(f"❌ Failed to get subtitles after {MAX_RETRIES} attempts for {url}")
        number += 1


if __name__ == "__main__":
    main()
