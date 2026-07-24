"""
Uzbek Video Dubbing — всё приложение в одном файле.

Запуск (Python 3.11):
  1. Установите FFmpeg и убедитесь, что команды ffmpeg/ffprobe доступны.
  2. pip install fastapi "uvicorn[standard]" python-multipart httpx faster-whisper
  3. Создайте НОВЫЙ Vertex AI Express Mode API key (формат "AQ....") и задайте
     его только через окружение (никогда не пишите его в код):
       export GEMINI_API_KEY="ваш_новый_ключ"
  4. python app.py
  5. Откройте http://localhost:8000

Настройки окружения (необязательно):
  GEMINI_TEXT_MODEL=gemini-2.5-flash
  GEMINI_TTS_MODEL=gemini-3.1-flash-tts-preview
  WHISPER_MODEL=small
  WHISPER_DEVICE=cpu
  WHISPER_COMPUTE_TYPE=int8
  MAX_UPLOAD_MB=500
  MAX_VIDEO_MINUTES=20
  JOB_TTL_HOURS=24
  PORT=8000

API-ключ намеренно не хранится в этом файле и не отправляется в браузер.
"""

from __future__ import annotations

import atexit
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse


# -----------------------------------------------------------------------------
# Конфигурация
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "data"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_VIDEO_MINUTES = int(os.getenv("MAX_VIDEO_MINUTES", "20"))
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_HOURS", "24")) * 3600
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}
VOICES = [
    "Kore", "Zephyr", "Puck", "Charon", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]

ProgressCallback = Callable[[int, str], None]
app = FastAPI(title="Uzbek Video Dubbing", version="1.0.0")
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dubbing")
_whisper_models: dict[str, object] = {}
_whisper_lock = threading.Lock()


# -----------------------------------------------------------------------------
# Gemini API: перевод и TTS
# -----------------------------------------------------------------------------

@dataclass
class DubSegment:
    index: int
    start: float
    end: float
    source_text: str
    translated_text: str

    @property
    def duration(self) -> float:
        return max(0.25, self.end - self.start)


class GeminiClient:
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("На сервере не задан GEMINI_API_KEY")
        self.api_key = api_key
        self.text_model = self._safe_model(os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash"))
        self.tts_model = self._safe_model(
            os.getenv("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")
        )
        self.http = httpx.Client(timeout=httpx.Timeout(180.0, connect=20.0))

    @staticmethod
    def _safe_model(value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            raise RuntimeError("Некорректное имя Gemini-модели")
        return value

    def close(self) -> None:
        self.http.close()

    def _post(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        # Vertex AI Express Mode: global endpoint, key as query param, snake_case fields.
        url = (
            f"https://aiplatform.googleapis.com/v1/publishers/google/models/"
            f"{model}:generateContent?key={urllib.parse.quote(self.api_key, safe='')}"
        )
        last_error = "Неизвестная ошибка Vertex AI API"
        for attempt in range(3):
            try:
                response = self.http.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
            except httpx.HTTPError as exc:
                last_error = f"Vertex AI API недоступен: {exc}"
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(last_error) from exc

            if response.is_success:
                return response.json()

            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except ValueError:
                detail = response.text
            last_error = f"Vertex AI API: HTTP {response.status_code}: {detail}"
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(last_error)
        raise RuntimeError(last_error)

    @staticmethod
    def _response_text(data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini не вернул результат")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            reason = candidates[0].get("finishReason") or candidates[0].get(
                "finish_reason", "UNKNOWN"
            )
            raise RuntimeError(f"Gemini не вернул текст, причина: {reason}")
        return text

    @staticmethod
    def _json(text: str) -> Any:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Gemini вернул некорректный JSON перевода") from exc

    def translate(self, segments: list[dict[str, Any]]) -> list[DubSegment]:
        translated: list[DubSegment] = []
        for offset in range(0, len(segments), 20):
            batch = segments[offset : offset + 20]
            source = [
                {
                    "id": item["index"],
                    "duration_seconds": round(item["end"] - item["start"], 2),
                    "text": item["text"],
                }
                for item in batch
            ]
            prompt = (
                "Переведи реплики для видеодубляжа на естественный разговорный узбекский "
                "язык ЛАТИНИЦЕЙ. Сохрани смысл, имена и числа. Сделай каждую реплику достаточно "
                "короткой для duration_seconds. Не добавляй объяснений. Верни только JSON-массив "
                'вида [{"id":0,"translated_text":"..."}].\nВход:\n'
                + json.dumps(source, ensure_ascii=False)
            )
            data = self._post(
                self.text_model,
                {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generation_config": {
                        "temperature": 0.2,
                        "response_mime_type": "application/json",
                    },
                },
            )
            result = self._json(self._response_text(data))
            if isinstance(result, dict):
                result = result.get("segments", result.get("translations", []))
            if not isinstance(result, list):
                raise RuntimeError("Gemini вернул перевод в неожиданном формате")
            by_id = {
                int(item["id"]): str(item["translated_text"]).strip()
                for item in result
                if isinstance(item, dict) and item.get("translated_text") and "id" in item
            }
            for item in batch:
                text = by_id.get(item["index"])
                if not text:
                    raise RuntimeError(f"Gemini пропустил реплику {item['index'] + 1}")
                translated.append(
                    DubSegment(
                        index=item["index"],
                        start=item["start"],
                        end=item["end"],
                        source_text=item["text"],
                        translated_text=text,
                    )
                )
        return translated

    def shorten(self, text: str, duration: float) -> str:
        prompt = (
            f"Сократи узбекскую реплику для произнесения максимум за {duration:.1f} сек. "
            "Сохрани смысл и узбекскую латиницу. Верни только реплику без кавычек:\n" + text
        )
        data = self._post(
            self.text_model,
            {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generation_config": {"temperature": 0.2},
            },
        )
        return self._response_text(data).strip().strip('"')

    def tts(self, text: str, voice: str, output_path: Path, duration: float) -> None:
        pace = "quickly but clearly" if duration < 3.0 else "naturally and expressively"
        prompt = (
            f"Speak {pace} in natural Uzbek. Read only the text after MATN, without comments.\n"
            f"MATN:\n{text}"
        )
        data = self._post(
            self.tts_model,
            {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generation_config": {
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "voice_config": {"prebuilt_voice_config": {"voice_name": voice}}
                    },
                },
            },
        )
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini TTS не вернул результат")
        parts = candidates[0].get("content", {}).get("parts", [])
        inline = next(
            (
                part.get("inlineData") or part.get("inline_data")
                for part in parts
                if part.get("inlineData") or part.get("inline_data")
            ),
            None,
        )
        if not inline or not inline.get("data"):
            raise RuntimeError("Gemini TTS не вернул аудио")
        try:
            audio = base64.b64decode(inline["data"], validate=True)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Gemini TTS вернул повреждённое аудио") from exc

        if audio.startswith(b"RIFF"):
            output_path.write_bytes(audio)
            return
        mime = str(inline.get("mimeType") or inline.get("mime_type") or "")
        match = re.search(r"rate=(\d+)", mime)
        sample_rate = int(match.group(1)) if match else 24000
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio)


# -----------------------------------------------------------------------------
# FFmpeg, Whisper и сборка аудио
# -----------------------------------------------------------------------------

def run_command(command: list[str], timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Не найдена команда {command[0]}. Установите FFmpeg.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Команда {command[0]} выполнялась слишком долго") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()[-1600:]
        raise RuntimeError(f"Ошибка {command[0]}: {detail}") from exc


def media_duration(path: Path) -> float:
    result = run_command(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        timeout=60,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("Не удалось определить длительность файла") from exc
    if duration <= 0:
        raise RuntimeError("Файл имеет нулевую длительность")
    return duration


def has_audio(path: Path) -> bool:
    result = run_command(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
        ],
        timeout=60,
    )
    return bool(result.stdout.strip())


def extract_audio(video: Path, output: Path) -> None:
    run_command(
        [
            "ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(output),
        ]
    )


def whisper_model():
    name = os.getenv("WHISPER_MODEL", "small")
    with _whisper_lock:
        if name not in _whisper_models:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "Не установлен faster-whisper: pip install faster-whisper"
                ) from exc
            device = os.getenv("WHISPER_DEVICE", "cpu")
            compute = os.getenv(
                "WHISPER_COMPUTE_TYPE", "int8" if device == "cpu" else "float16"
            )
            _whisper_models[name] = WhisperModel(name, device=device, compute_type=compute)
        return _whisper_models[name]


def transcribe(audio: Path, language: str | None) -> list[dict[str, Any]]:
    segments_iterator, _ = whisper_model().transcribe(
        str(audio),
        language=language,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    result: list[dict[str, Any]] = []
    for segment in segments_iterator:
        text = segment.text.strip()
        if text and segment.end > segment.start:
            result.append(
                {
                    "index": len(result),
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text,
                }
            )
    if not result:
        raise RuntimeError("В видео не найдена речь")
    return result


def atempo_filter(ratio: float) -> str:
    factors: list[float] = []
    while ratio > 2.0:
        factors.append(2.0)
        ratio /= 2.0
    while ratio < 0.5:
        factors.append(0.5)
        ratio /= 0.5
    factors.append(ratio)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def normalize_and_fit(source: Path, output: Path, target_seconds: float) -> None:
    actual = media_duration(source)
    filters: list[str] = []
    if actual > target_seconds * 1.04:
        filters.extend(["-filter:a", atempo_filter(actual / max(target_seconds, 0.25))])
    run_command(
        [
            "ffmpeg", "-y", "-i", str(source), *filters, "-t", f"{target_seconds:.3f}",
            "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(output),
        ],
        timeout=180,
    )


def read_mono_pcm(path: Path) -> array:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 24000:
            raise RuntimeError("Внутренняя ошибка формата WAV")
        samples = array("h")
        samples.frombytes(wav.readframes(wav.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def render_timeline(
    duration: float,
    segments: list[DubSegment],
    client: GeminiClient,
    voice: str,
    work_dir: Path,
    progress: ProgressCallback,
) -> Path:
    sample_rate = 24000
    timeline = array("h", [0]) * (int(duration * sample_rate) + sample_rate)
    segments_dir = work_dir / "segments"
    segments_dir.mkdir(exist_ok=True)
    transcript: list[dict[str, Any]] = []

    for number, segment in enumerate(segments, start=1):
        raw = segments_dir / f"{segment.index:04d}-raw.wav"
        fitted = segments_dir / f"{segment.index:04d}.wav"
        spoken_text = segment.translated_text
        client.tts(spoken_text, voice, raw, segment.duration)
        if media_duration(raw) > segment.duration * 1.65 and len(spoken_text) > 18:
            spoken_text = client.shorten(spoken_text, segment.duration)
            client.tts(spoken_text, voice, raw, segment.duration)
        normalize_and_fit(raw, fitted, segment.duration)

        clip = read_mono_pcm(fitted)
        start_sample = max(0, int(segment.start * sample_rate))
        available = min(len(clip), len(timeline) - start_sample)
        for index in range(available):
            mixed = timeline[start_sample + index] + clip[index]
            timeline[start_sample + index] = max(-32768, min(32767, mixed))

        transcript.append(
            {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "source": segment.source_text,
                "uzbek": spoken_text,
            }
        )
        progress(
            45 + round(number / len(segments) * 40),
            f"Озвучивается реплика {number} из {len(segments)}",
        )

    output = work_dir / "dubbed.wav"
    final_samples = timeline[: int(duration * sample_rate)]
    if sys.byteorder != "little":
        final_samples.byteswap()
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(final_samples.tobytes())
    (work_dir / "segments.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def mux_video(video: Path, dubbed: Path, output: Path, mode: str) -> None:
    should_mix = mode == "mix" and has_audio(video)

    def command(video_codec: list[str]) -> list[str]:
        base = ["ffmpeg", "-y", "-i", str(video), "-i", str(dubbed)]
        if should_mix:
            audio = [
                "-filter_complex",
                "[0:a:0]volume=0.18[original];[original][1:a:0]"
                "amix=inputs=2:duration=longest:normalize=0[aout]",
                "-map", "0:v:0", "-map", "[aout]",
            ]
        else:
            audio = ["-map", "0:v:0", "-map", "1:a:0"]
        return base + audio + video_codec + [
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            "-shortest", str(output),
        ]

    try:
        run_command(command(["-c:v", "copy"]))
    except RuntimeError:
        output.unlink(missing_ok=True)
        run_command(command(["-c:v", "libx264", "-preset", "medium", "-crf", "20"]))


def dubbing_pipeline(
    input_path: Path,
    output_path: Path,
    language: str | None,
    voice: str,
    audio_mode: str,
    progress: ProgressCallback,
) -> None:
    work_dir = input_path.parent
    progress(5, "Проверяется видео")
    duration = media_duration(input_path)
    if duration > MAX_VIDEO_MINUTES * 60:
        raise RuntimeError(f"Видео длиннее лимита {MAX_VIDEO_MINUTES} минут")
    if not has_audio(input_path):
        raise RuntimeError("В видео нет аудиодорожки")

    source_audio = work_dir / "source.wav"
    progress(10, "Извлекается аудио")
    extract_audio(input_path, source_audio)
    progress(18, "Распознаётся речь")
    source_segments = transcribe(source_audio, language)

    client = GeminiClient()
    try:
        progress(35, f"Переводятся {len(source_segments)} реплик на узбекский")
        translated = client.translate(source_segments)
        progress(45, "Начинается генерация узбекского голоса")
        dubbed = render_timeline(duration, translated, client, voice, work_dir, progress)
    finally:
        client.close()

    progress(90, "Собирается готовое видео")
    mux_video(input_path, dubbed, output_path, audio_mode)
    progress(98, "Проверяется результат")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("FFmpeg не создал итоговое видео")
    media_duration(output_path)


# -----------------------------------------------------------------------------
# Очередь и HTTP API
# -----------------------------------------------------------------------------

def public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "error": job.get("error"),
        "result_url": f"/api/jobs/{job['id']}/result" if job["status"] == "completed" else None,
    }


def update_job(job_id: str, **changes: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(changes)


def cleanup_old_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    for path in JOBS_DIR.iterdir():
        try:
            if path.is_dir() and path.stat().st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def process_job(
    job_id: str,
    input_path: Path,
    language: str | None,
    voice: str,
    audio_mode: str,
) -> None:
    output_path = input_path.parent / "uzbek-dubbed.mp4"

    def progress(percent: int, message: str) -> None:
        update_job(job_id, status="processing", progress=percent, message=message)

    try:
        dubbing_pipeline(input_path, output_path, language, voice, audio_mode, progress)
        update_job(
            job_id,
            status="completed",
            progress=100,
            message="Дубляж готов",
            result_path=str(output_path),
        )
    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            message="Не удалось обработать видео",
            error=str(exc),
        )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    return HTML_PAGE.replace("__MAX_MB__", str(MAX_UPLOAD_MB)).replace(
        "__MAX_MINUTES__", str(MAX_VIDEO_MINUTES)
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/jobs", status_code=202)
async def create_job(
    video: UploadFile = File(...),
    source_language: str = Form("auto"),
    voice: str = Form("Kore"),
    audio_mode: str = Form("mix"),
) -> dict[str, Any]:
    if not os.getenv("GEMINI_API_KEY", "").strip():
        raise HTTPException(status_code=503, detail="На сервере не задан GEMINI_API_KEY")
    if source_language not in {"auto", "ru", "en", "uz"}:
        raise HTTPException(status_code=400, detail="Неподдерживаемый исходный язык")
    if voice not in VOICES:
        raise HTTPException(status_code=400, detail="Неизвестный голос")
    if audio_mode not in {"mix", "replace"}:
        raise HTTPException(status_code=400, detail="Неизвестный режим звука")

    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Поддерживаются MP4, MOV, WEBM и MKV")

    cleanup_old_jobs()
    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    input_path = job_dir / f"input{suffix}"
    written = 0
    try:
        with input_path.open("wb") as destination:
            while chunk := await video.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413, detail=f"Файл больше лимита {MAX_UPLOAD_MB} МБ"
                    )
                destination.write(chunk)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    finally:
        await video.close()
    if written == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Загружен пустой файл")

    job = {
        "id": job_id,
        "status": "queued",
        "progress": 2,
        "message": "Видео добавлено в очередь",
        "error": None,
        "result_path": None,
    }
    with _jobs_lock:
        _jobs[job_id] = job
    _executor.submit(
        process_job,
        job_id,
        input_path,
        None if source_language == "auto" else source_language,
        voice,
        audio_mode,
    )
    return public_job(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Задача не найдена")
        return public_job(job)


@app.get("/api/jobs/{job_id}/result")
def result(job_id: str) -> FileResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Задача не найдена")
        result_path = job.get("result_path")
        if job["status"] != "completed" or not result_path:
            raise HTTPException(status_code=409, detail="Видео ещё не готово")
    path = Path(result_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Результат был удалён")
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Content-Disposition": 'inline; filename="uzbek-dubbed.mp4"'},
    )


@atexit.register
def shutdown() -> None:
    _executor.shutdown(wait=False, cancel_futures=True)


# -----------------------------------------------------------------------------
# Весь frontend: HTML + CSS + JavaScript
# -----------------------------------------------------------------------------

HTML_PAGE = r'''<!doctype html>
<html lang="ru">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ovoz — дубляж на узбекский</title>
<style>
:root{--bg:#12121a;--raised:#1a1a24;--panel:#1e1e2a;--line:#30303f;--text:#ede9e3;--dim:#9292a1;--gold:#d4a574;--red:#c85049}*{box-sizing:border-box}html,body{margin:0;min-height:100%}body{background:radial-gradient(ellipse 900px 520px at 15% -10%,rgba(212,165,116,.1),transparent 60%),var(--bg);color:var(--text);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;padding:46px 20px 80px}.wrap{width:min(100%,820px);margin:auto}header{margin-bottom:36px}.eyebrow{color:var(--gold);font-size:11px;letter-spacing:.18em;text-transform:uppercase;display:flex;align-items:center;gap:10px}.eyebrow:before{content:"";width:7px;height:7px;border-radius:50%;background:var(--red);box-shadow:0 0 8px var(--red)}h1{font-family:Georgia,serif;font-size:clamp(36px,7vw,58px);line-height:1;letter-spacing:-.025em;margin:13px 0;font-weight:600}h1 em{color:var(--gold);font-weight:500}.sub{color:var(--dim);max-width:620px;font-size:14px;line-height:1.7}.panel{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:24px;margin-bottom:18px}.title{margin:0 0 17px;color:var(--dim);font-size:11px;letter-spacing:.14em;text-transform:uppercase;display:flex;justify-content:space-between;gap:16px}.drop{min-height:170px;border:1px dashed #48485a;background:var(--raised);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;cursor:pointer;text-align:center;padding:24px}.drop:hover,.drop.drag{border-color:var(--gold);background:rgba(212,165,116,.06)}.drop input{display:none}.plus{color:var(--gold);font-size:28px}.drop strong{font-size:14px}.drop small,.option small,.hint{display:block;color:var(--dim);font-size:11px;line-height:1.5}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}label,legend{color:var(--dim);font-size:10px;letter-spacing:.08em;text-transform:uppercase}select{width:100%;margin-top:8px;background:var(--raised);border:1px solid var(--line);color:var(--text);font:13px ui-monospace,monospace;padding:12px;border-radius:3px}fieldset{border:0;padding:0;margin:22px 0 0}legend{margin-bottom:9px}.option{display:flex;align-items:center;gap:10px;background:var(--raised);border:1px solid var(--line);padding:12px;margin-top:8px;cursor:pointer;text-transform:none;letter-spacing:0}.option:has(input:checked){border-color:var(--gold)}.option input{accent-color:var(--gold)}.option span{display:flex;align-items:baseline;justify-content:space-between;width:100%;gap:16px}.option strong{color:var(--text);font-size:12px}.primary,.download{display:block;width:100%;border:0;border-radius:3px;background:var(--gold);color:#12121a;padding:16px;font:700 12px ui-monospace,monospace;letter-spacing:.1em;text-transform:uppercase;text-align:center;cursor:pointer;text-decoration:none}.primary:disabled{background:var(--line);color:var(--dim);cursor:wait}.error{color:#ee736c;font-size:12px;line-height:1.6;display:none}.error.show{display:block}.progress,.result{margin-top:18px}.head{display:flex;align-items:center;justify-content:space-between;font-size:12px}.dot{display:inline-block;width:8px;height:8px;background:var(--red);border-radius:50%;margin-right:10px;box-shadow:0 0 8px var(--red);animation:pulse 1.2s infinite}.track{height:5px;background:var(--raised);margin:17px 0 10px;overflow:hidden}.bar{width:0;height:100%;background:var(--gold);transition:width .35s}video{display:block;width:100%;max-height:460px;background:#08080c;margin-bottom:16px}footer{color:var(--dim);text-align:center;font-size:10px;margin-top:30px}@keyframes pulse{50%{opacity:.35}}@media(max-width:620px){body{padding:30px 14px 60px}.panel{padding:18px}.grid{grid-template-columns:1fr}.option span{display:block}.limit{display:none}}
</style>
</head><body><main class="wrap">
<header><p class="eyebrow">Gemini TTS · Uzbek dubbing</p><h1>Видео говорит <em>по-узбекски</em>.</h1><p class="sub">Загрузите видео — сервис распознает речь, переведёт её и создаст узбекскую озвучку.</p></header>
<form id="form">
<section class="panel"><p class="title"><span>01 · Видео</span><span class="limit">до __MAX_MB__ МБ · __MAX_MINUTES__ минут</span></p><label class="drop" id="drop" for="video"><input id="video" name="video" type="file" accept="video/mp4,video/quicktime,video/webm,.mkv" required><span class="plus">＋</span><strong id="fileName">Выберите видео</strong><small id="fileMeta">MP4, MOV, WEBM или MKV</small></label></section>
<section class="panel"><p class="title">02 · Настройки</p><div class="grid"><div><label for="lang">Исходный язык</label><select id="lang" name="source_language"><option value="auto">Определить автоматически</option><option value="ru">Русский</option><option value="en">Английский</option><option value="uz">Узбекский</option></select></div><div><label for="voice">Узбекский голос</label><select id="voice" name="voice"><option>Kore</option><option>Zephyr</option><option>Puck</option><option>Charon</option><option>Fenrir</option><option>Leda</option><option>Aoede</option><option>Orus</option></select></div></div><fieldset><legend>Оригинальный звук</legend><label class="option"><input type="radio" name="audio_mode" value="mix" checked><span><strong>Тихий фон</strong><small>Оригинал остаётся на громкости 18%</small></span></label><label class="option"><input type="radio" name="audio_mode" value="replace"><span><strong>Полная замена</strong><small>Остаётся только узбекская речь</small></span></label></fieldset></section>
<button class="primary" id="submit" type="submit">Создать узбекский дубляж</button><p class="error" id="error" role="alert"></p></form>
<section class="panel progress" id="progress" hidden><div class="head"><div><span class="dot"></span><span id="status">Подготовка…</span></div><strong id="percent">0%</strong></div><div class="track"><div class="bar" id="bar"></div></div><p class="hint">Не закрывайте страницу до окончания обработки.</p></section>
<section class="panel result" id="result" hidden><p class="title">Готовый дубляж</p><video id="player" controls playsinline></video><a class="download" id="download" download="uzbek-dubbed.mp4">Скачать видео</a></section>
<footer>Gemini API key хранится только на сервере · файлы удаляются через 24 часа</footer></main>
<script>
const form=document.querySelector('#form'),video=document.querySelector('#video'),drop=document.querySelector('#drop'),fileName=document.querySelector('#fileName'),fileMeta=document.querySelector('#fileMeta'),submit=document.querySelector('#submit'),error=document.querySelector('#error'),progress=document.querySelector('#progress'),bar=document.querySelector('#bar'),percent=document.querySelector('#percent'),statusText=document.querySelector('#status'),result=document.querySelector('#result'),player=document.querySelector('#player'),download=document.querySelector('#download');let timer=null;
const size=n=>n<1048576?`${(n/1024).toFixed(0)} КБ`:`${(n/1048576).toFixed(1)} МБ`;function showFile(f){if(f){fileName.textContent=f.name;fileMeta.textContent=size(f.size)}}video.onchange=()=>showFile(video.files[0]);['dragenter','dragover'].forEach(n=>drop.addEventListener(n,e=>{e.preventDefault();drop.classList.add('drag')}));['dragleave','drop'].forEach(n=>drop.addEventListener(n,e=>{e.preventDefault();drop.classList.remove('drag')}));drop.addEventListener('drop',e=>{if(e.dataTransfer.files.length){video.files=e.dataTransfer.files;showFile(video.files[0])}});
function fail(m){error.textContent=m;error.classList.add('show')}function clear(){error.textContent='';error.classList.remove('show')}function update(j){progress.hidden=false;const p=Math.max(0,Math.min(100,Number(j.progress)||0));bar.style.width=p+'%';percent.textContent=p+'%';statusText.textContent=j.message||'Обработка…'}async function parse(r){const d=await r.json().catch(()=>({}));if(!r.ok)throw new Error(d.detail||`Ошибка HTTP ${r.status}`);return d}
async function poll(id){try{const j=await parse(await fetch(`/api/jobs/${id}`));update(j);if(j.status==='completed'){submit.disabled=false;submit.textContent='Создать ещё один дубляж';const url=j.result_url+'?t='+Date.now();player.src=url;download.href=url;result.hidden=false;result.scrollIntoView({behavior:'smooth'});return}if(j.status==='failed'){submit.disabled=false;submit.textContent='Попробовать снова';fail(j.error||'Не удалось создать дубляж');return}timer=setTimeout(()=>poll(id),1500)}catch(e){submit.disabled=false;fail(e.message)}}
form.onsubmit=async e=>{e.preventDefault();clear();result.hidden=true;player.removeAttribute('src');player.load();if(!video.files.length){fail('Выберите видео');return}if(timer)clearTimeout(timer);submit.disabled=true;submit.textContent='Загрузка видео…';update({progress:1,message:'Видео загружается на сервер'});try{const j=await parse(await fetch('/api/jobs',{method:'POST',body:new FormData(form)}));submit.textContent='Создаётся дубляж…';update(j);poll(j.id)}catch(e){submit.disabled=false;submit.textContent='Создать узбекский дубляж';fail(e.message)}};
</script></body></html>'''


if __name__ == "__main__":
    cleanup_old_jobs()
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
