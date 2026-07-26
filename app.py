"""
Uzbek Video Dubbing — всё приложение в одном файле (полностью автоматический режим).

Запуск (Python 3.11):
  1. Установите FFmpeg и убедитесь, что команды ffmpeg/ffprobe доступны.
  2. pip install fastapi "uvicorn[standard]" python-multipart httpx faster-whisper
  3. Создайте НОВЫЙ Vertex AI Express Mode API key (формат "AQ....") и задайте
     его только через окружение (никогда не пишите его в код):
       export GEMINI_API_KEY="ваш_новый_ключ"
  4. python app.py
  5. Откройте http://localhost:8000

Как работает (всё автоматически, править ничего не нужно):
  Загружаете видео -> распознаётся речь по СЛОВАМ -> слова выравниваются по
  аудио -> диаризация размечает говорящих по всему файлу -> речь делится на
  смысловые блоки одного говорящего, а те — на дыхательные группы по реальным
  паузам -> Gemini переводит на узбекский под СЛОГОВОЙ бюджет -> голосом
  Gemini 3.1 озвучивается каждая группа и ставится на своё начало слова ->
  собирается готовое видео.

Четыре независимых уровня (их смешивание было главной причиной рассинхрона):
  слова ASR -> speaker turns -> translation units 4-12 с -> TTS-группы 1-4 с.

Настройки окружения (необязательно):
  GEMINI_TEXT_MODEL=gemini-2.5-flash
  GEMINI_TTS_MODEL=gemini-3.1-flash-tts-preview
  WHISPER_MODEL=medium   # точнее small; для максимума качества можно large-v3 (медленнее)
  WHISPER_DEVICE=cpu
  WHISPER_COMPUTE_TYPE=int8
  FORCED_ALIGNMENT=auto  # off — не пытаться выравнивать слова через whisperx
  DIARIZATION_BACKEND=auto  # auto|pyannote|gemini
  PYANNOTE_MODEL=pyannote/speaker-diarization-community-1
  HF_TOKEN=               # токен Hugging Face для pyannote
  NUM_SPEAKERS=           # если число говорящих известно — задайте точно
  MIN_SPEAKERS= / MAX_SPEAKERS=
  MAX_UPLOAD_MB=500
  MAX_VIDEO_MINUTES=20
  JOB_TTL_HOURS=24
  TTS_CONCURRENCY=4      # сколько фраз озвучивать параллельно
  PORT=8000

Точность липсинка растёт по мере работы: длительности реальных генераций
складываются в data/tts_calibration.json и калибруют модель длительности для
каждого голоса.

Дополнительно (сильно повышает точность, ставится отдельно):
  pip install whisperx           # forced alignment: точные границы слов
  pip install pyannote.audio     # диаризация и разметка наложений
Без них пайплайн работает на метках Whisper и диаризации через Gemini.

API-ключ намеренно не хранится в этом файле и не отправляется в браузер.
"""

from __future__ import annotations

import atexit
import base64
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import sys
import threading
import time
import urllib.parse
import uuid
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

# Подхватываем ключи из файла .env рядом с app.py, если он есть. Это самый
# частый источник ошибки «не задан GEMINI_API_KEY»: ключ кладут в .env, а
# приложение читало только переменные окружения процесса.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv()  # и .env из текущего каталога запуска
except ImportError:
    pass


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
TTS_CONCURRENCY = max(1, int(os.getenv("TTS_CONCURRENCY", "4")))
# Пост-обработка темпа — последнее средство, а не основной инструмент подгонки.
# Сначала правим текст (слоговой бюджет), затем просим TTS говорить быстрее,
# и только остаток добираем растяжением в узком, неслышимом диапазоне.
MAX_SPEED_UP_RATIO = 1.15   # рабочий предел ускорения (выше уже слышно)
MIN_SLOWDOWN_RATIO = 0.92   # рабочий предел замедления
EMERGENCY_SPEED_UP = 1.25   # аварийный режим: только для коротких реплик (<2.5 с)
REGENERATE_OVERFLOW = 1.15  # перегенерируем реплику, если вылезла больше чем на 15%
LOOP_DURATION_FACTOR = 1.8  # длиннее прогноза в 1.8 раза — TTS «зациклился», переозвучиваем
TTS_PASSES = 3              # столько заходов на реплику: тишина в дубляже недопустима
TAIL_KEEP_SECONDS = 2.0     # хвост за пределами видео: слово должно договориться
MAX_MISSING_SPEECH = 0.12   # больше этой доли непереозвученной речи — дубляж бракуем
SOURCE_OVERLAP_EPS = 0.08  # с какого наложения в оригинале считаем это перебиванием
TAIL_TOLERANCE = 0.40      # допустимый хвост за окном, чтобы не рубить слова
ONSET_OFFSET = 0.0         # старт по реально найденному началу речи
# Планировщик размещения реплик (замена жадного сдвига). Значения из практики
# губ-синка: старт в пределах ~120 мс воспринимается «в такт», сдвиг закадровой
# реплики терпим до ~400 мс. Порог различения двух реплик как раздельных ~80 мс.
SPEAKER_GAP_MIN = 0.15     # целевой зазор между репликами РАЗНЫХ говорящих
SPEAKER_GAP_HARD = 0.08    # абсолютный минимум, ниже — звучит как «разом»
SAME_SPEAKER_GAP = 0.03    # зазор между своими же дыхательными группами
MAX_SHIFT_ONSCREEN = 0.15  # реплику «в кадре» дальше не двигаем (губы)
MAX_SHIFT_OFFSCREEN = 0.40 # закадровую/реакцию можно сдвинуть сильнее
OVERLAP_DUCK = 0.35        # приглушение перебиваемого при неизбежном нахлёсте (~ -9 дБ)
MAX_STYLE_LEN = 400
MAX_SCENE_LEN = 1200
MAX_CONTEXT_CHARS = 12000  # ограничение контекста, чтобы ответ не обрывался

VOICES = [
    "Kore", "Zephyr", "Puck", "Charon", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]
DEFAULT_FEMALE_VOICE = "Aoede"  # мягче и естественнее, чем Kore
DEFAULT_MALE_VOICE = "Charon"

ProgressCallback = Callable[[int, str], None]
app = FastAPI(title="Uzbek Video Dubbing", version="3.0.0")
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dubbing")
_whisper_models: dict[str, object] = {}
_whisper_lock = threading.Lock()


# -----------------------------------------------------------------------------
# Gemini API: перевод и TTS
# -----------------------------------------------------------------------------

@dataclass
class Word:
    """Уровень 1: слово с временными метками (ASR + forced alignment)."""

    start: float
    end: float
    text: str
    score: float = 1.0  # уверенность выравнивания (1.0 — метки прямо из ASR)
    speaker: str = ""  # метка говорящего, назначается после диаризации


@dataclass
class SpeakerTurn:
    """Уровень 2: непрерывный участок одного говорящего. Длина не ограничена."""

    speaker: str
    start: float
    end: float
    overlap: float = 0.0  # доля времени, занятая перекрытием с другим голосом

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class DubChunk:
    """Уровень 4: дыхательная группа — один запрос к TTS и одно место на таймлайне.

    Каждая группа ставится на СВОЁ лексическое начало, поэтому реальные паузы
    оригинала сохраняются сами собой, без вставки синтетической тишины.
    """

    start: float  # начало по словам ASR
    end: float
    source_text: str
    text: str = ""  # узбекский текст этой группы
    onset: float = 0.0  # уточнённое лексическое начало (куда ставим озвучку)
    budget: float = 0.0  # сколько секунд реально доступно
    generated: float = 0.0  # измеренная длительность TTS до подгонки
    ratio: float = 1.0  # фактически применённое изменение темпа

    @property
    def duration(self) -> float:
        return max(0.2, self.end - self.start)


@dataclass
class DubSegment:
    """Уровень 3: translation unit — смысловой блок внутри одного speaker turn."""

    index: int
    start: float
    end: float
    source_text: str
    translated_text: str = ""
    speaker: str = "female"  # регистр голоса для TTS: "female" | "male"
    speaker_label: str = ""  # кто говорит: S1, S2, ... (диаризация)
    voice: str = ""  # конкретный TTS-голос этого персонажа
    style: str = ""  # авто-определённая эмоция/интонация реплики
    chunks: list[DubChunk] = field(default_factory=list)
    overlap: float = 0.0  # доля перекрытия с другим говорящим
    short_variant: str = ""  # более короткий вариант перевода (на случай перелива)

    @property
    def duration(self) -> float:
        return max(0.25, self.end - self.start)

    @property
    def speech_budget(self) -> float:
        """Время только под речь: сумма окон дыхательных групп без внутренних пауз."""
        if self.chunks:
            return sum(chunk.duration for chunk in self.chunks)
        return self.duration


class GeminiClient:
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Не задан GEMINI_API_KEY. Задайте его перед запуском: "
                "export GEMINI_API_KEY=\"ваш_ключ\" — или создайте файл .env рядом "
                "с app.py со строкой GEMINI_API_KEY=ваш_ключ (без кавычек и пробелов "
                "вокруг знака =)."
            )
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
        except json.JSONDecodeError:
            pass

        # Ответ мог обрезаться на середине (лимит вывода). Спасаем целые объекты.
        salvaged: list[Any] = []
        for match in re.finditer(r"\{[^{}]*\}", cleaned):
            try:
                salvaged.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                continue
        if salvaged:
            return salvaged
        raise RuntimeError("Gemini вернул некорректный JSON перевода")

    def analyze_scene(self, segments: list[dict[str, Any]]) -> str:
        """Просит Gemini понять сцену целиком: место, отношения, характеры, настроение.

        Этот «режиссёрский разбор» затем подмешивается в перевод и озвучку, чтобы
        актёры играли, а не читали ровным тоном.
        """
        script = "\n".join(f"[{s['index']}] {s['text']}" for s in segments)[:6000]
        prompt = (
            "You are a film dubbing director. Read this dialogue transcript and briefly analyze "
            "the scene so voice actors can perform it naturally. In 4-6 short sentences describe: "
            "the setting and situation, the relationship between the speakers, and the distinct "
            "personality, mood and vocal energy of EACH speaker (how they should sound — e.g. "
            "flustered and defensive, or cocky and pushy). Be concrete and vivid. Reply in "
            "English, plain text only.\n\nTRANSCRIPT:\n" + script
        )
        try:
            data = self._post(
                self.text_model,
                {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generation_config": {"temperature": 0.5},
                },
            )
            return self._response_text(data).strip()[:MAX_SCENE_LEN]
        except RuntimeError:
            return ""  # разбор необязателен — при сбое продолжаем без него

    def _translate_batch(
        self,
        context_lines: str,
        scene: str,
        target: list[dict[str, Any]],
        results: dict[int, dict[str, Any]],
    ) -> None:
        scene_block = f"SCENE BRIEF (для тона и характеров):\n{scene}\n\n" if scene else ""
        prompt = (
            "Ты режиссёр дубляжа и переводчик. Ниже SCENE BRIEF (разбор сцены) и CONTEXT — "
            "весь скрипт по порядку. Переведи ТОЛЬКО реплики из TARGET, играя сцену.\n"
            "Для каждой целевой реплики верни поля:\n"
            "1) translated_text — ТОЧНЫЙ и ЖИВОЙ разговорный перевод на УЗБЕКСКИЙ ЛАТИНИЦЕЙ.\n"
            "   ГЛАВНОЕ — ТОЧНОСТЬ СМЫСЛА: переведи именно то, что человек сказал. Ничего не "
            "выдумывай, не добавляй и не выбрасывай смысловые части, сохраняй имена, числа, "
            "вопрос остаётся вопросом, отрицание — отрицанием.\n"
            "   Пиши живым разговорным языком носителя, с эмоцией и интонацией персонажа, а не "
            "сухим подстрочником.\n"
            "   ДЛИНА СЧИТАЕТСЯ В СЛОГАХ, НЕ В СИМВОЛАХ. target_syllables — сколько слогов "
            "укладывается в окно оригинала, max_syllables — предел. Стремись к "
            "target_syllables и не превышай max_syllables: считай слоги по гласным "
            "(a, e, i, o, u, oʻ), учитывай, что цифры произносятся словами, а узбекские "
            "окончания добавляют слоги. Короче — лучше, чем длиннее.\n"
            "2) short_variant — тот же смысл, но на 20-30% КОРОЧЕ по слогам (убери вводные "
            "слова, повторы и местоимения). Он пойдёт в дело, если основной вариант не влезет "
            "в тайминг. Смысл, имена и числа обязаны сохраниться.\n"
            "3) parts — ТОЛЬКО если у реплики в TARGET есть массив source_parts: верни ровно "
            "столько же узбекских частей, в том же порядке и с тем же распределением смысла. "
            "Части разделены реальными паузами актёра, их длительность сохраняется.\n"
            "4) style — ПОДРОБНАЯ актёрская ремарка на английском (одно живое предложение): "
            "эмоция, подтекст, энергия, темп, отношение персонажа и, если уместно, невербалика "
            '(короткий смешок, вздох, придыхание, заминка). Например: "flustered and defensive, '
            'a nervous little laugh, speaks fast and a bit high". Разным персонажам — заметно '
            "разные ремарки.\n"
            "Верни только JSON-массив "
            '[{"id":0,"translated_text":"...","short_variant":"...","parts":["..."],'
            '"style":"..."}] строго для id из TARGET.\n\n'
            + scene_block
            + "CONTEXT:\n"
            + context_lines
            + "\n\nTARGET:\n"
            + json.dumps(target, ensure_ascii=False)
        )
        data = self._post(
            self.text_model,
            {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generation_config": {
                    "temperature": 0.6,
                    "response_mime_type": "application/json",
                    "max_output_tokens": 8192,
                },
            },
        )
        result = self._json(self._response_text(data))
        if isinstance(result, dict):
            result = result.get("segments", result.get("translations", []))
        if not isinstance(result, list):
            raise RuntimeError("Gemini вернул перевод в неожиданном формате")
        for item in result:
            if not isinstance(item, dict) or "id" not in item:
                continue
            # Пробел или пустая строка — не перевод: такие реплики уйдут на
            # повторную попытку, а не молча превратятся в тишину.
            if str(item.get("translated_text", "")).strip():
                results[int(item["id"])] = item

    def _translate_single(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Простой резервный перевод одной реплики обычным текстом (без JSON).

        Нужен, когда строгий формат ломается на конкретной строке — лучше
        перевести её проще, чем уронить весь дубляж.
        """
        prompt = (
            "Переведи эту реплику на естественный разговорный УЗБЕКСКИЙ язык ЛАТИНИЦЕЙ. "
            f"Уложись примерно в {item['target_syllables']} слогов (считай по гласным). "
            "Верни ТОЛЬКО перевод, без кавычек, пояснений и форматирования:\n"
            + str(item["source"])
        )
        try:
            data = self._post(
                self.text_model,
                {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generation_config": {"temperature": 0.3},
                },
            )
            text = self._response_text(data).strip().strip('"').strip()
        except RuntimeError:
            return None
        if not text:
            return None
        return {"id": item["id"], "translated_text": text, "style": ""}

    def translate(
        self, units: list[DubSegment], voice_map: dict[str, str], scene: str = ""
    ) -> None:
        """Переводит translation units с бюджетом В СЛОГАХ и заполняет их на месте.

        Единица перевода — смысловой блок одного говорящего (4-12 с), а не
        обрубок в 3.2 с, поэтому синтаксис узбекской фразы больше не рвётся.
        """
        # Весь скрипт передаётся как контекст в каждый запрос — перевод получается
        # связным и согласованным, а не «вслепую» по кускам.
        context_lines = json.dumps(
            [
                {"id": unit.index, "speaker": unit.speaker_label, "text": unit.source_text}
                for unit in units
            ],
            ensure_ascii=False,
        )[:MAX_CONTEXT_CHARS]

        payloads: dict[int, dict[str, Any]] = {}
        for unit in units:
            voice = voice_map.get(unit.speaker, voice_map["female"])
            budget = unit.speech_budget
            target = syllable_budget(budget, voice)
            item: dict[str, Any] = {
                "id": unit.index,
                "source": unit.source_text,
                "seconds": round(budget, 2),
                "target_syllables": target,
                "max_syllables": max(target + 2, int(target * 1.15)),
            }
            if len(unit.chunks) > 1:
                item["source_parts"] = [chunk.source_text for chunk in unit.chunks]
            payloads[unit.index] = item

        results: dict[int, dict[str, Any]] = {}

        def run(batch: list[dict[str, Any]], depth: int = 0) -> None:
            """Переводит партию; при сбое или пропусках делит её на половины."""
            missing = [item for item in batch if item["id"] not in results]
            if not missing:
                return
            try:
                self._translate_batch(context_lines, scene, missing, results)
            except RuntimeError:
                if len(missing) == 1 or depth >= 4:
                    raise
            still_missing = [item for item in missing if item["id"] not in results]
            if not still_missing:
                return
            if len(still_missing) == 1 or depth >= 4:
                # Последняя попытка: простой перевод построчно. Не вышло —
                # реплика разбирается ниже вместе с остальными пропусками.
                for item in still_missing:
                    fallback = self._translate_single(item)
                    if fallback:
                        results[item["id"]] = fallback
                return
            middle = len(still_missing) // 2
            run(still_missing[:middle], depth + 1)
            run(still_missing[middle:], depth + 1)

        ordered = [payloads[unit.index] for unit in units]
        step = 20
        for offset in range(0, len(ordered), step):
            run(ordered[offset : offset + step])

        failed: list[DubSegment] = []
        for unit in units:
            if _apply_translation(unit, results.get(unit.index)):
                continue
            # Пустой ответ бывает на «репликах» без слов: музыка, вздох, шум,
            # который распознавание приняло за речь. Пробуем ещё раз простым
            # запросом и только потом отбрасываем — падать всем дубляжом из-за
            # одной такой строки нельзя.
            if _apply_translation(unit, self._translate_single(payloads[unit.index])):
                continue
            failed.append(unit)

        if failed:
            report = "; ".join(
                f"{unit.start:.2f}s «{unit.source_text[:40]}»" for unit in failed[:6]
            )
            print(
                f"[dubbing] без перевода осталось реплик: {len(failed)} ({report})",
                flush=True,
            )
            total_seconds = sum(unit.duration for unit in units)
            lost_seconds = sum(unit.duration for unit in failed)
            kept = [unit for unit in units if unit not in failed]
            if not kept or (
                lost_seconds > 1.5
                and total_seconds > 0
                and lost_seconds / total_seconds > MAX_MISSING_SPEECH
            ):
                raise RuntimeError(
                    f"Gemini не перевёл {len(failed)} реплик(и) — "
                    f"{lost_seconds:.0f}с из {total_seconds:.0f}с речи. Первые: {report}"
                )
            units[:] = kept

    def condense_to_budget(self, units: list[DubSegment], voice_map: dict[str, str]) -> int:
        """Дожимает текст ДО озвучки: переписать короче лучше, чем потом ускорять.

        Порядок из практики дубляжа: сначала адаптация текста, потом подача, и
        только в конце лёгкая компрессия. Здесь — первый шаг: реплики, у которых
        предсказанная длительность вылезает за окно, переписываются под слоговой
        бюджет.
        """
        overflowing: list[DubSegment] = []
        for unit in units:
            voice = voice_map.get(unit.speaker, voice_map["female"])
            if predict_unit_duration(unit, voice, safe=True) > (
                unit.speech_budget * REGENERATE_OVERFLOW
            ):
                overflowing.append(unit)
        if not overflowing:
            return 0

        fixed = 0
        for offset in range(0, len(overflowing), 20):
            batch = overflowing[offset : offset + 20]
            payload = []
            for unit in batch:
                voice = voice_map.get(unit.speaker, voice_map["female"])
                target = syllable_budget(unit.speech_budget, voice)
                payload.append(
                    {
                        "id": unit.index,
                        "source": unit.source_text,
                        "uzbek": unit.translated_text,
                        "target_syllables": target,
                    }
                )
            prompt = (
                "Сократи узбекские фразы так, чтобы каждая укладывалась в "
                "target_syllables СЛОГОВ (считай по гласным a, e, i, o, u, oʻ; цифры "
                "произносятся словами). Убирай только повторы, местоимения, вводные "
                "конструкции и служебные слова; смысл, имена, числа, вопрос и отрицание "
                "сохрани полностью. Второстепенное можно опустить, но обрывать слова и "
                "фразу нельзя. Это устная речь для дубляжа, узбекская латиница. "
                'Верни только JSON-массив [{"id":0,"uzbek":"..."}].\n\nСТРОКИ:\n'
                + json.dumps(payload, ensure_ascii=False)
            )
            try:
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
            except RuntimeError:
                continue
            if isinstance(result, dict):
                result = result.get("segments", result.get("lines", []))
            if not isinstance(result, list):
                continue
            shortened = {
                int(item["id"]): dedupe_repeats(normalize_uzbek(str(item["uzbek"]).strip()))
                for item in result
                if isinstance(item, dict) and item.get("uzbek") and "id" in item
            }
            for unit in batch:
                voice = voice_map.get(unit.speaker, voice_map["female"])
                candidate = shortened.get(unit.index)
                if not candidate:
                    continue
                current = predict_unit_duration(unit, voice, safe=True)
                improved = predict_speech_duration(candidate, voice, safe=True)
                if improved < current:
                    unit.translated_text = candidate
                    for chunk, text in zip(
                        unit.chunks, split_text_by_chunks(candidate, unit.chunks)
                    ):
                        chunk.text = text
                    fixed += 1
        return fixed

    def identify_speakers(
        self, audio: Path, segments: list[dict[str, Any]]
    ) -> tuple[dict[int, str], dict[str, str]]:
        """Резервная диаризация через Gemini: метки говорящих для черновых реплик.

        Это НЕ основной путь: LLM не даёт frame-level вероятностей, embeddings и
        разметки наложений, поэтому на коротких репликах ошибается. Если доступен
        pyannote, используется он. Результат здесь всегда проходит temporal
        smoothing, а тембр голоса (низкий/высокий) считается отдельно по F0 —
        мнение модели о поле используется только в спорной зоне.
        """
        try:
            raw = audio.read_bytes()
        except OSError:
            return {}, {}
        if len(raw) > 18 * 1024 * 1024:
            return {}, {}  # слишком длинное аудио для одного запроса

        listing = "\n".join(
            f"id={item['index']} {item['start']:.2f}s-{item['end']:.2f}s: {item['text']}"
            for item in segments
        )
        prompt = (
            "Listen carefully to the attached dialogue audio and perform speaker diarization.\n"
            "1) Assign each line below to a speaker: S1, S2, S3... The SAME person must always "
            "get the SAME label through the whole audio. Judge by voice timbre, not by content.\n"
            "2) For each speaker, state the gender of the VOICE: male or female.\n"
            "Every id must appear exactly once. Return only JSON in this exact shape:\n"
            '{"turns":[{"id":0,"speaker":"S1"},{"id":1,"speaker":"S2"}],'
            '"speakers":[{"speaker":"S1","gender":"male"},'
            '{"speaker":"S2","gender":"female"}]}\n\nLINES:\n' + listing
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(raw).decode("ascii")}},
                    ],
                }
            ],
            "generation_config": {
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        }
        try:
            data = self._post(self.text_model, payload)
            result = self._json(self._response_text(data))
        except RuntimeError:
            return {}, {}

        turns_raw: Any = []
        speakers_raw: Any = []
        if isinstance(result, dict):
            turns_raw = result.get("turns", result.get("lines", []))
            speakers_raw = result.get("speakers", [])
        elif isinstance(result, list):
            # Модель могла вернуть плоский список — разберём его как turns.
            turns_raw = result

        turns: dict[int, str] = {}
        if isinstance(turns_raw, list):
            for item in turns_raw:
                if not isinstance(item, dict) or "id" not in item:
                    continue
                label = str(item.get("speaker", "")).strip().upper()
                if not label:
                    gender = str(item.get("gender", "")).strip().lower()
                    label = gender.upper() if gender in {"male", "female"} else ""
                if not label:
                    continue
                try:
                    turns[int(item["id"])] = label
                except (TypeError, ValueError):
                    continue

        speaker_genders: dict[str, str] = {}
        if isinstance(speakers_raw, list):
            for item in speakers_raw:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("speaker", "")).strip().upper()
                gender = str(item.get("gender", "")).strip().lower()
                if label and gender in {"male", "female"}:
                    speaker_genders[label] = gender
        # Если модель сразу вернула пол вместо метки — используем его как метку.
        for label in set(turns.values()):
            if label in {"MALE", "FEMALE"} and label not in speaker_genders:
                speaker_genders[label] = label.lower()
        return turns, speaker_genders

    def polish(self, segments: list[DubSegment]) -> None:
        """Второй проход: Gemini перечитывает свой узбекский текст и исправляет огрехи.

        Ловит то, что часто портит первый проход: задвоенные слова, кальки с
        русского/английского, неестественные обороты, ошибки в латинице.
        Вычитка идёт ПО ДЫХАТЕЛЬНЫМ ГРУППАМ с контекстом всей реплики, чтобы не
        разрушить привязку частей к реальным паузам оригинала.
        """
        pairs = [
            (segment, position, chunk)
            for segment in segments
            for position, chunk in enumerate(segment.chunks)
            if chunk.text.strip()
        ]
        for offset in range(0, len(pairs), 30):
            batch = pairs[offset : offset + 30]
            payload = [
                {
                    "id": number,
                    "context": segment.source_text,
                    "source": chunk.source_text,
                    "uzbek": chunk.text,
                }
                for number, (segment, _, chunk) in enumerate(batch, start=offset)
            ]
            prompt = (
                "Ты редактор-носитель узбекского языка, вычитываешь текст дубляжа. context — "
                "вся реплика целиком (для понимания), source — именно та часть, которую нужно "
                "проверить. Для каждой "
                "строки СНАЧАЛА проверь главное: точно ли uzbek передаёт смысл source — не "
                "потерян ли смысловой кусок, не искажён ли смысл, сохранены ли имена, числа, "
                "вопрос/отрицание. Если смысл неверный, перепиши строку правильно. Затем "
                "исправь задвоенные и лишние слова, кальки с русского/английского, "
                "неестественные обороты, ошибки грамматики и узбекской латиницы. Сохрани "
                "примерно ту же длину и разговорный стиль — это устная речь для озвучки. "
                "Если строка уже хороша, верни её без изменений. "
                'Верни только JSON-массив [{"id":0,"uzbek":"..."}].\n\nСТРОКИ:\n'
                + json.dumps(payload, ensure_ascii=False)
            )
            try:
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
            except RuntimeError:
                continue  # вычитка необязательна — при сбое оставляем первый перевод
            if isinstance(result, dict):
                result = result.get("segments", result.get("lines", []))
            if not isinstance(result, list):
                continue
            fixed = {
                int(item["id"]): str(item["uzbek"]).strip()
                for item in result
                if isinstance(item, dict) and item.get("uzbek") and "id" in item
            }
            for number, (segment, _, chunk) in enumerate(batch, start=offset):
                better = fixed.get(number)
                if better:
                    chunk.text = dedupe_repeats(normalize_uzbek(better))
                    segment.translated_text = _clean_words(
                        " ".join(part.text for part in segment.chunks if part.text)
                    )

    def tts(
        self,
        text: str,
        voice: str,
        output_path: Path,
        duration: float,
        style_note: str = "",
        scene: str = "",
        speaker: str = "",
        context: str = "",
        pace: str = "normal",
    ) -> None:
        # Пол задаём и в конфиге, и словами в промпте: одной настройки модели
        # оказалось недостаточно — она озвучивала все реплики одним голосом.
        gender_line = ""
        if speaker == "female":
            gender_line = (
                "You are voicing a WOMAN. Use a clearly FEMININE voice — higher pitch, "
                "lighter and softer timbre. Never sound like a man.\n"
            )
        elif speaker == "male":
            gender_line = (
                "You are voicing a MAN. Use a clearly MASCULINE voice — lower pitch, "
                "fuller chest timbre. Never sound like a woman.\n"
            )
        prompt = (
            "You are a top film dubbing voice actor performing a real character in a live scene "
            "— NOT a text-to-speech reader. Perform the line after MATN in fluent, natural, "
            "conversational Uzbek as a REAL PERSON would say it in this moment.\n"
            + gender_line
            + "Absolute rules:\n"
            "- Sound fully human and alive: rich emotion, expressive intonation, natural rhythm "
            "with micro-pauses and small changes of pace. NEVER flat, monotone or robotic.\n"
            "- Fully embody the emotion and attitude in DIRECTION (e.g. flustered, defensive, "
            "cocky, teasing, nervous, excited). Let it clearly colour the voice.\n"
            "- Where it fits the emotion, add subtle natural non-verbal touches — a short breath, "
            "a small laugh or scoff, a brief hesitation — but keep ALL the Uzbek words intact.\n"
            "- Pronounce every word fully and clearly to the very end; never cut words.\n"
            "- Do not add any pause before the first word and do not trail off with silence "
            "at the end: the timing of pauses is set by the film, not by you.\n"
            "- Do not read SCENE/DIRECTION/FULL LINE aloud; output speech only.\n"
        )
        if pace == "faster":
            # Просьба к подаче — второй шаг после правки текста и ДО любого DSP.
            prompt += (
                "- PACE: speak noticeably faster and more energetic than usual, keep it "
                f"natural, no dragging: the line must fit about {duration:.1f}s.\n"
            )
        else:
            prompt += (
                "- Speak at a calm, relaxed, slightly slower-than-average conversational pace. "
                "Take your time, leave natural little pauses between phrases. Do NOT rush or "
                f"compress words (the line has about {duration:.1f}s available).\n"
            )
        if scene:
            prompt += f"SCENE: {scene}\n"
        if style_note.strip():
            prompt += f"DIRECTION: {style_note.strip()}\n"
        if context.strip() and context.strip() != text.strip():
            # Контекст всей реплики держит интонацию: дыхательная группа не
            # звучит как отдельная оборванная фраза.
            prompt += (
                f"FULL LINE (context only, do NOT speak it): {context.strip()}\n"
                "Speak ONLY the part after MATN, with intonation that fits its place in "
                "the full line.\n"
            )
        prompt += f"MATN:\n{text}"

        # Голый промпт на крайний случай: иногда развёрнутый промпт даёт от
        # Vertex «HTTP 400: invalid argument», и реплика утекает в оригинал.
        # Минимальный запрос модель принимает почти всегда.
        bare_prompt = f"{gender_line}Speak this Uzbek line naturally and clearly:\n{text}"

        def build_payload(text_prompt: str) -> dict[str, Any]:
            return {
                "contents": [{"role": "user", "parts": [{"text": text_prompt}]}],
                "generation_config": {
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "voice_config": {"prebuilt_voice_config": {"voice_name": voice}}
                    },
                },
            }

        inline = None
        detail = ""
        attempts = 4
        for attempt in range(attempts):
            # На последней попытке — голый промпт (обходит invalid-argument).
            payload = build_payload(bare_prompt if attempt == attempts - 1 else prompt)
            try:
                data = self._post(self.tts_model, payload)
            except RuntimeError as exc:
                # В т.ч. перемежающийся HTTP 400: не бросаем сразу, а повторяем —
                # иначе одна реплика молча остаётся на языке оригинала.
                detail = str(exc)
                if attempt < attempts - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"Gemini TTS не вернул аудио после {attempts} попыток. {detail}"
                ) from exc
            candidates = data.get("candidates") or []
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                inline = next(
                    (
                        part.get("inlineData") or part.get("inline_data")
                        for part in parts
                        if part.get("inlineData") or part.get("inline_data")
                    ),
                    None,
                )
                if inline and inline.get("data"):
                    break
                # Диагностика: причина завершения и текст, если модель ответила словами.
                reason = candidates[0].get("finishReason") or candidates[0].get(
                    "finish_reason", "UNKNOWN"
                )
                returned_text = "".join(p.get("text", "") for p in parts).strip()
                detail = f"finishReason={reason}"
                if returned_text:
                    detail += f", ответ моделью текстом: {returned_text[:200]}"
            else:
                detail = "пустой ответ (нет candidates)"
            inline = None
            if attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))

        if not inline or not inline.get("data"):
            raise RuntimeError(
                f"Gemini TTS не вернул аудио после {attempts} попыток. "
                f"{detail}. Проверьте доступ к модели {self.tts_model} и лимиты; "
                "при частых сбоях снизьте TTS_CONCURRENCY."
            )
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


def audio_channels(path: Path) -> int:
    result = run_command(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=channels", "-of", "csv=p=0", str(path),
        ],
        timeout=60,
    )
    try:
        return int(result.stdout.strip().rstrip(","))
    except ValueError:
        return 0


def extract_audio(video: Path, output: Path) -> None:
    """Моно 16 кГц для распознавания и диаризации.

    У многоканального источника (5.1) диалог живёт в центральном канале: берём
    его напрямую, иначе музыка и эффекты из остальных каналов ухудшают и ASR, и
    диаризацию. Простой downmix оставляем только для моно/стерео.
    """
    channels = audio_channels(video)
    if channels >= 6:
        filters = ["-af", "pan=mono|c0=FC"]
        print(f"[dubbing] источник {channels}-канальный: беру центральный канал", flush=True)
    else:
        filters = ["-ac", "1"]
    run_command(
        [
            "ffmpeg", "-y", "-i", str(video), "-vn", *filters, "-ar", "16000",
            "-c:a", "pcm_s16le", str(output),
        ]
    )


def whisper_model():
    name = os.getenv("WHISPER_MODEL", "medium")
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


SENTENCE_END = (".", "!", "?", "…")

# -----------------------------------------------------------------------------
# Четыре РАЗНЫХ уровня разбиения (раньше это была одна «фраза» на всё сразу).
#
#   1) слова ASR                  — атомы времени, приходят из word timestamps;
#   2) speaker turn               — участок одного говорящего (диаризация),
#                                   БЕЗ искусственного лимита длины;
#   3) translation unit           — предложение/смысловой блок ~4-12 с,
#                                   всегда внутри ОДНОГО speaker turn;
#   4) TTS chunk (дыхательная группа) — то, что реально уходит в TTS одним
#                                   куском: 1-4 с между реальными паузами.
#
# Смешивание этих уровней и было главным дефектом: лимит 3.2 с рвал и
# акустический контекст говорящего, и синтаксис перевода.
# -----------------------------------------------------------------------------
UNIT_TARGET_SECONDS = 8.0    # к такой длине стремимся для translation unit
UNIT_MAX_SECONDS = 12.0      # жёсткий предел одного смыслового блока
UNIT_FLUSH_SECONDS = 3.5     # после конца предложения закрываем unit от этой длины
UNIT_SPLIT_GAP = 0.6         # пауза, по которой обязательно начинаем новый unit
TURN_MERGE_GAP = 0.7         # соседние реплики одного говорящего сливаем в turn
SHORT_TURN_SECONDS = 1.2     # такие «одиночные» метки считаем ненадёжными
# Пауза >= 250 мс синхронизируется программно (её слышно и часто под жест);
# всё, что короче, остаётся на совести пунктуации и самого TTS.
CHUNK_PAUSE_MIN = 0.25
CHUNK_MAX_SECONDS = 8.0      # предел дыхательной группы для одного TTS-запроса
UTTERANCE_GAP = 0.5          # черновая нарезка для диаризационного промпта
UTTERANCE_MAX_SECONDS = 6.0
ONSET_SEARCH_BACK = 0.20     # насколько раньше первого слова ищем его атаку
ONSET_SEARCH_FORWARD = 0.30  # и насколько позже, когда слова выровнены alignment
ONSET_SEARCH_RAW = 2.50      # без alignment метки Whisper «уезжают» вперёд на секунды
NONLEXICAL_MAX = 0.50        # изолированный всплеск такой длины — вздох/смешок
NONLEXICAL_GAP = 0.20        # если после него пауза, это была не речь
ONSET_LEAD = 0.03            # запас перед найденной атакой согласного


PITCH_RATE = 8000       # частота для анализа питча (достаточно для F0)
PITCH_MIN_HZ = 70.0
PITCH_MAX_HZ = 400.0
PITCH_SPLIT_HZ = 170.0        # граница низкий/высокий регистр в спорных случаях
MALE_CONFIDENT_HZ = 155.0     # ниже — уверенно низкий регистр
FEMALE_CONFIDENT_HZ = 185.0   # выше — уверенно высокий регистр
PITCH_MIN_VOICED = 1.0        # минимум озвученного материала на говорящего, с
GENDER_CONFIDENT_VOICED = 3.0 # уверенное решение о поле — от 3 с материала
ANCHOR_MIN_SECONDS = 2.5      # «эталонный» участок говорящего: не короче этого
ANCHOR_MAX_OVERLAP = 0.10     # и почти без наложения чужого голоса
BIMODAL_SPLIT_HZ = 40.0       # две моды F0 дальше друг от друга — метка склеила двоих
DISTINCT_MIN_F0_HZ = 15.0     # два голоса одного пола в сцене должны отличаться на столько
VOICE_FINGERPRINTS_PATH = JOBS_DIR / "voice_fingerprints.json"


def make_pitch_audio(video: Path, output: Path) -> None:
    """Готовит аудио для анализа питча: моно 8 кГц + полоса 70-400 Гц.

    Полосовой фильтр убирает музыку, шум и обертоны выше основного тона —
    без него автокорреляция часто ошибается на живой речи.
    """
    run_command(
        [
            "ffmpeg", "-y", "-i", str(video), "-vn",
            "-af", "highpass=f=65,lowpass=f=420,dynaudnorm",
            "-ac", "1", "-ar", str(PITCH_RATE), "-c:a", "pcm_s16le", str(output),
        ]
    )


def _frame_pitch(signal: list[float], sample_rate: int, min_corr: float = 0.60) -> float:
    """Питч одного кадра через нормализованную автокорреляцию с коррекцией октавы.

    Кадр принимается только при достаточно высоком пике автокорреляции: шум и
    шипящие не должны попадать в статистику F0.
    """
    count = len(signal)
    mean = sum(signal) / count
    centred = [value - mean for value in signal]
    power = sum(value * value for value in centred)
    if power < 1e-3:
        return 0.0

    min_lag = max(2, int(sample_rate / PITCH_MAX_HZ))
    max_lag = min(count // 2, int(sample_rate / PITCH_MIN_HZ))
    if max_lag <= min_lag:
        return 0.0

    scores: dict[int, float] = {}
    best_lag, best_score = 0, 0.0
    for lag in range(min_lag, max_lag + 1):
        total = 0.0
        for index in range(count - lag):
            total += centred[index] * centred[index + lag]
        norm = total / power
        scores[lag] = norm
        if norm > best_score:
            best_score, best_lag = norm, lag

    if not best_lag or best_score < min_corr:
        return 0.0

    # Коррекция октавы: автокорреляция любит удвоенный период (вдвое ниже тон).
    half = best_lag // 2
    if half >= min_lag and scores.get(half, 0.0) >= best_score * 0.8:
        best_lag = half
    return sample_rate / best_lag


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _segment_pitch_values(samples: array, sample_rate: int, limit: int = 40) -> list[float]:
    """Все надёжные F0-кадры участка. Кадр 40 мс, шаг 10 мс, порог корреляции 0.6.

    Возвращаем именно кадры, а не одно число: решение о регистре голоса
    принимается по агрегату ВСЕХ реплик говорящего, а не по одной фразе.
    """
    frame = max(64, int(sample_rate * 0.04))  # 40 мс
    hop = max(16, int(sample_rate * 0.01))  # 10 мс
    if len(samples) < frame:
        return []

    energies: list[tuple[float, int]] = []
    for start in range(0, len(samples) - frame + 1, hop):
        window = samples[start : start + frame]
        energies.append((sum(abs(value) for value in window) / frame, start))
    if not energies:
        return []
    loudest = max(energy for energy, _ in energies)
    if loudest < 50:
        return []

    candidates = [start for energy, start in energies if energy >= loudest * 0.35]
    if len(candidates) > limit:  # ограничиваем работу, распределяя кадры по участку
        step = len(candidates) / limit
        candidates = [candidates[int(i * step)] for i in range(limit)]

    values: list[float] = []
    for min_corr in (0.60, 0.40):  # второй проход мягче — если материал шумный
        values = []
        for start in candidates:
            signal = [float(value) for value in samples[start : start + frame]]
            pitch = _frame_pitch(signal, sample_rate, min_corr)
            if PITCH_MIN_HZ <= pitch <= PITCH_MAX_HZ:
                values.append(pitch)
        if len(values) >= 8:
            break
    return values


def _load_mono(audio: Path) -> tuple[array, int]:
    with wave.open(str(audio), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise RuntimeError("Ожидается моно 16-бит WAV")
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples, rate


@dataclass
class VoiceStats:
    """Акустический профиль голоса говорящего, агрегированный по всем репликам."""

    f0: float = 0.0                 # медианный F0, взвешенный по длительности
    voiced: float = 0.0             # секунд озвученного материала
    per_turn: list[tuple[float, float]] = field(default_factory=list)  # (секунды, F0)
    bimodal: bool = False           # похоже, что метка склеила двух людей


def _weighted_median(pairs: list[tuple[float, float]]) -> float:
    """Медиана значений с весами (вес — длительность реплики)."""
    if not pairs:
        return 0.0
    ordered = sorted(pairs, key=lambda item: item[1])
    total = sum(weight for weight, _ in ordered)
    if total <= 0:
        return _median([value for _, value in ordered])
    cumulative = 0.0
    for weight, value in ordered:
        cumulative += weight
        if cumulative >= total / 2:
            return value
    return ordered[-1][1]


def _is_bimodal(pairs: list[tuple[float, float]]) -> bool:
    """1-D 2-means по F0: две устойчивые моды дальше BIMODAL_SPLIT_HZ — склейка."""
    values = [value for _, value in pairs if value > 0]
    if len(values) < 4 or max(values) - min(values) < BIMODAL_SPLIT_HZ:
        return False
    low, high = min(values), max(values)
    lows: list[float] = []
    highs: list[float] = []
    for _ in range(20):
        lows = [value for value in values if abs(value - low) <= abs(value - high)]
        highs = [value for value in values if abs(value - low) > abs(value - high)]
        if not lows or not highs:
            return False
        low, high = sum(lows) / len(lows), sum(highs) / len(highs)
    minority = min(len(lows), len(highs)) / len(values)
    return high - low >= BIMODAL_SPLIT_HZ and minority >= 0.2


def speaker_voice_stats(audio: Path, diarization: Diarization) -> dict[str, VoiceStats]:
    """Профиль КАЖДОГО ГОВОРЯЩЕГО по всем его «эталонным» участкам.

    F0 не идентифицирует человека (высокий мужской и низкий женский голос
    пересекаются), поэтому питч не участвует в решении «кто говорит» — только в
    выборе TTS-тембра. Решение принимается ОДИН РАЗ на говорящего голосованием,
    взвешенным по длительности, чтобы короткая реплика не могла его перевесить.
    """
    try:
        samples, rate = _load_mono(audio)
    except (wave.Error, OSError, RuntimeError):
        return {}

    anchors: dict[str, list[SpeakerTurn]] = {}
    for turn in diarization.turns:
        anchors.setdefault(turn.speaker, []).append(turn)

    stats: dict[str, VoiceStats] = {}
    hop_seconds = 0.01
    for label, turns in anchors.items():
        clean = [
            turn
            for turn in turns
            if turn.duration >= ANCHOR_MIN_SECONDS and turn.overlap <= ANCHOR_MAX_OVERLAP
        ]
        chosen = clean or sorted(turns, key=lambda item: item.duration, reverse=True)[:6]
        per_turn: list[tuple[float, float]] = []
        voiced = 0.0
        for turn in sorted(chosen, key=lambda item: item.duration, reverse=True)[:10]:
            start = max(0, int(turn.start * rate))
            end = min(len(samples), int(turn.end * rate))
            if end - start < rate // 4:
                continue
            values = _segment_pitch_values(samples[start:end], rate, limit=80)
            seconds = len(values) * hop_seconds
            if seconds <= 0:
                continue
            voiced += seconds
            per_turn.append((seconds, _median(values)))
        stats[label] = VoiceStats(
            f0=_weighted_median(per_turn),
            voiced=voiced,
            per_turn=per_turn,
            bimodal=_is_bimodal(per_turn),
        )
    return stats


def voice_registers(
    diarization: Diarization, stats: dict[str, VoiceStats]
) -> dict[str, str]:
    """Выбирает TTS-тембр (низкий/высокий) на говорящего, а не на реплику.

    Это выбор голоса для озвучки, а не «определение пола человека». Решение
    одно на весь фильм и принимается по агрегату всех реплик, поэтому голос
    персонажа не «прыгает» посреди диалога, а короткая реплика не может его
    сменить. В спорной зоне 155-185 Гц или при малом объёме материала опираемся
    на мнение модели о тембре, слышавшей аудио.
    """
    registers: dict[str, str] = {}
    uncertain: list[str] = []
    for label in diarization.speakers:
        profile = stats.get(label)
        pitch = profile.f0 if profile else 0.0
        confident = bool(profile and profile.voiced >= GENDER_CONFIDENT_VOICED)
        if pitch <= 0:
            uncertain.append(label)
        elif confident and pitch < MALE_CONFIDENT_HZ:
            registers[label] = "male"
        elif confident and pitch > FEMALE_CONFIDENT_HZ:
            registers[label] = "female"
        elif not confident and pitch < MALE_CONFIDENT_HZ - 10:
            registers[label] = "male"       # даже на малом материале явно низкий
        elif not confident and pitch > FEMALE_CONFIDENT_HZ + 10:
            registers[label] = "female"     # и это явно высокий
        else:
            uncertain.append(label)         # спорно — решаем ниже

    for label in uncertain:
        opinion = diarization.genders.get(label)
        if opinion in {"male", "female"}:
            registers[label] = opinion
            continue
        pitch = stats[label].f0 if label in stats else 0.0
        if pitch > 0:
            registers[label] = "male" if pitch < PITCH_SPLIT_HZ else "female"
            continue
        registers[label] = next(iter(registers.values()), "male")
    return registers


def load_voice_fingerprints() -> dict[str, dict[str, float]]:
    """Акустические «отпечатки» TTS-голосов (F0 и т.п.), посчитанные заранее.

    Файл заполняется калибровкой: один раз генерируем эталонную фразу каждым
    голосом и измеряем F0. Пусто — работаем на двух голосах из настроек.
    """
    try:
        data = json.loads(VOICE_FINGERPRINTS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(voice): value
        for voice, value in data.items()
        if isinstance(value, dict) and value.get("f0")
    }


def assign_character_voices(
    diarization: Diarization,
    stats: dict[str, VoiceStats],
    registers: dict[str, str],
    voice_map: dict[str, str],
    fingerprints: dict[str, dict[str, float]],
) -> dict[str, str]:
    """Подбирает КАЖДОМУ персонажу отдельный голос из доступных.

    Пол — жёсткий фильтр (кросс-гендерный подбор запрещён). Внутри пола голос
    выбирается ближайшим по F0 к оригинальному актёру; два персонажа не
    получают один голос, а два голоса одного пола различаются по F0 не меньше
    DISTINCT_MIN_F0_HZ. Нет отпечатков — падаем на два голоса из настроек.
    """
    fallback = {
        label: voice_map.get(registers.get(label, "male"), voice_map["female"])
        for label in diarization.speakers
    }
    if not fingerprints:
        return fallback

    def voice_gender(info: dict[str, float]) -> str:
        return "male" if float(info.get("f0", 0.0)) < PITCH_SPLIT_HZ else "female"

    pool: dict[str, list[str]] = {"male": [], "female": []}
    for voice, info in fingerprints.items():
        pool[voice_gender(info)].append(voice)

    # Крупные роли выбирают первыми — им достаётся самый похожий голос.
    order = sorted(
        diarization.speakers,
        key=lambda label: stats.get(label, VoiceStats()).voiced,
        reverse=True,
    )
    assigned: dict[str, str] = {}
    used: set[str] = set()
    for label in order:
        gender = registers.get(label, "male")
        candidates = [voice for voice in pool.get(gender, []) if voice not in used]
        if not candidates:
            assigned[label] = fallback[label]
            continue
        target = stats.get(label, VoiceStats()).f0 or (140.0 if gender == "male" else 210.0)

        def score(voice: str, target: float = target, gender: str = gender) -> float:
            info = fingerprints[voice]
            distance = abs(math.log(max(info["f0"], 1.0)) - math.log(max(target, 1.0)))
            clash = 0.0
            for other in used:
                other_info = fingerprints.get(other)
                if other_info and voice_gender(other_info) == gender:
                    if abs(other_info["f0"] - info["f0"]) < DISTINCT_MIN_F0_HZ:
                        clash += 1.0
            return distance + clash

        best = min(candidates, key=score)
        assigned[label] = best
        used.add(best)
    return assigned


def _frame_levels(
    samples: array, rate: int, start: float, end: float
) -> list[tuple[float, float]]:
    """Уровень сигнала по кадрам 10 мс: [(время, амплитуда), ...]."""
    frame = max(1, rate // 100)
    first = max(0, int(start * rate))
    last = min(len(samples), int(end * rate))
    levels: list[tuple[float, float]] = []
    position = first
    while position + frame <= last:
        window = samples[position : position + frame]
        levels.append((position / rate, float(max(abs(value) for value in window))))
        position += frame
    return levels


def _speech_runs(
    levels: list[tuple[float, float]], threshold: float
) -> list[tuple[float, float]]:
    """Непрерывные участки «похоже на речь» в виде (начало, конец) в секундах."""
    runs: list[tuple[float, float]] = []
    start: float | None = None
    last = 0.0
    for moment, level in levels:
        if level >= threshold:
            if start is None:
                start = moment
            last = moment + 0.01
        elif start is not None:
            if last - start >= 0.04:  # короче 40 мс — щелчок, не звук речи
                runs.append((start, last))
            start = None
    if start is not None and last - start >= 0.04:
        runs.append((start, last))

    merged: list[tuple[float, float]] = []
    for run_start, run_end in runs:
        if merged and run_start - merged[-1][1] < 0.05:
            merged[-1] = (merged[-1][0], run_end)
        else:
            merged.append((run_start, run_end))
    return merged


def refine_lexical_onset(
    samples: array,
    rate: int,
    word_start: float,
    lower: float,
    upper: float,
    forward: float = ONSET_SEARCH_FORWARD,
) -> float:
    """Уточняет начало ПЕРВОГО СЛОВА, а не «первого громкого звука».

    Ключевое отличие от прежней логики: поиск ограничен окрестностью слова,
    полученного из ASR/forced alignment. Поэтому вздох, смешок или кашель за
    секунду до речи в принципе не могут стать началом реплики — они попросту
    вне окна поиска. Внутри окна ищем атаку: устойчивое превышение локального
    шумового порога, и отступаем на 30 мс назад, чтобы не срезать согласный.
    """
    search_from = max(lower, word_start - ONSET_SEARCH_BACK)
    search_to = min(upper, word_start + forward)
    if search_to - search_from < 0.03:
        return word_start

    levels = _frame_levels(samples, rate, search_from, search_to)
    if not levels:
        return word_start
    # Порог считаем по ВСЕМУ окну поиска, а не по кусочку у word_start: иначе,
    # когда речь начинается на 1-2 с позже (в начале тишина), пик меряется в
    # тишине, и начало ошибочно остаётся на месте метки Whisper.
    window_levels = sorted(level for _, level in levels)
    noise = window_levels[int(len(window_levels) * 0.2)]
    peak = window_levels[-1]
    if peak < 150:
        return word_start
    threshold = max(noise * 3.5, peak * 0.12, 120.0)

    runs = _speech_runs(levels, threshold)
    for number, (run_start, run_end) in enumerate(runs):
        following = runs[number + 1] if number + 1 < len(runs) else None
        if (
            following is not None
            and run_end - run_start <= NONLEXICAL_MAX
            and following[0] - run_end >= NONLEXICAL_GAP
        ):
            # Короткий всплеск, после которого пауза — вздох, смешок или стук.
            # Слово так не звучит, поэтому cue открываем не здесь.
            continue
        return max(lower, min(run_start - ONSET_LEAD, search_to))
    return word_start


def refine_onsets(
    audio: Path, units: list[DubSegment], diarization: Diarization, aligned: bool = False
) -> int:
    """Ставит каждой дыхательной группе её лексическое начало (dub_start).

    С forced alignment метки слов точные, поэтому окно поиска узкое. Без него
    Whisper часто открывает реплику на секунду раньше, поэтому окно шире, а
    изолированные всплески (вздох, смешок) пропускаются.
    """
    try:
        samples, rate = _load_mono(audio)
    except (wave.Error, OSError, RuntimeError):
        return 0

    forward = ONSET_SEARCH_FORWARD if aligned else ONSET_SEARCH_RAW
    refined = 0
    for unit in units:
        turn_start, turn_end = diarization.turn_bounds(unit.start, unit.end)
        for position, chunk in enumerate(unit.chunks):
            lower = max(0.0, turn_start if position == 0 else unit.chunks[position - 1].end)
            upper = min(chunk.end - 0.05, turn_end if turn_end > chunk.start else chunk.end)
            if upper <= lower:
                chunk.onset = chunk.start
                continue
            # Внутри реплики метки надёжнее (сдвигается обычно только первое
            # слово), поэтому широкое окно даём лишь первой группе.
            onset = refine_lexical_onset(
                samples,
                rate,
                chunk.start,
                lower,
                upper,
                forward if position == 0 else ONSET_SEARCH_FORWARD,
            )
            if abs(onset - chunk.onset) > 0.005:
                refined += 1
            chunk.onset = max(0.0, onset)
        if unit.chunks:
            unit.start = unit.chunks[0].onset
    return refined


# -----------------------------------------------------------------------------
# Модель длительности узбекской речи: слоги вместо символов
#
# Символы не отражают ни числа слогов, ни раскрытия цифр, ни агглютинативных
# окончаний, ни скорости конкретного TTS-голоса. Поэтому длину перевода теперь
# планируем через слоги и предсказанную длительность, а коэффициенты
# калибруются на РЕАЛЬНЫХ генерациях конкретного голоса.
# -----------------------------------------------------------------------------

UZ_VOWELS = "aeiou"
UZ_DIGRAPHS = ("sh", "ch", "ng")
STRONG_BREAKS = (".", "!", "?", "…", ";", ":", "—")

# Стартовый prior (до накопления калибровки), из отраслевых оценок:
#   D = 0.12 + 0.17*слоги + 0.015*согласные + 0.03*слова
PRIOR_COEFFS = (0.12, 0.17, 0.015, 0.03)
PAUSE_PER_COMMA = 0.18
PAUSE_PER_BREAK = 0.30
CALIBRATION_MIN_SAMPLES = 40   # меньше — доверяем prior, а не шуму
CALIBRATION_MAX_SAMPLES = 4000
CALIBRATION_RIDGE = 1e-3
CALIBRATION_PATH = JOBS_DIR / "tts_calibration.json"
_calibration_lock = threading.Lock()
_calibration_cache: dict[str, Any] = {}

UZ_UNITS = ["", "bir", "ikki", "uch", "toʻrt", "besh", "olti", "yetti", "sakkiz", "toqqiz"]
UZ_TENS = [
    "", "oʻn", "yigirma", "oʻttiz", "qirq", "ellik",
    "oltmish", "yetmish", "sakson", "toqson",
]
UZ_SIGNS = {
    "%": " foiz ",
    "$": " dollar ",
    "€": " yevro ",
    "₽": " rubl ",
    "&": " va ",
    "№": " raqam ",
}
# Сокращения читаются словами: «15 kg» — это 6 слогов, а не 5 символов.
UZ_ABBREVIATIONS = {
    "kg": "kilogramm",
    "km": "kilometr",
    "sm": "santimetr",
    "mm": "millimetr",
    "ml": "millilitr",
    "gr": "gramm",
    "soat": "soat",
    "min": "minut",
    "sek": "sekund",
    "yil": "yil",
}


def _number_to_uzbek(value: int) -> str:
    """Раскрывает число словами: «15» — это 2 слога, а не 2 символа."""
    if value < 0:
        return "minus " + _number_to_uzbek(-value)
    if value == 0:
        return "nol"
    parts: list[str] = []
    for scale, name in ((1_000_000_000, "milliard"), (1_000_000, "million"), (1000, "ming")):
        if value >= scale:
            count = value // scale
            value %= scale
            prefix = _number_to_uzbek(count) if count > 1 else ""
            parts.append(f"{prefix} {name}".strip())
    if value >= 100:
        hundreds = value // 100
        value %= 100
        prefix = UZ_UNITS[hundreds] if hundreds > 1 else ""
        parts.append(f"{prefix} yuz".strip())
    if value >= 10:
        parts.append(UZ_TENS[value // 10])
        value %= 10
    if value:
        parts.append(UZ_UNITS[value])
    return " ".join(part for part in parts if part)


def normalize_uzbek(text: str) -> str:
    """Приводит узбекский текст к единому виду перед подсчётом и озвучкой.

    Единый апостроф в oʻ/gʻ, раскрытые числа, знаки и сокращения — иначе одна и
    та же фраза даёт разные оценки длительности и разное чтение в TTS.
    """
    normalized = re.sub(r"[’‘`´ʼ']", "ʻ", text)
    for sign, word in UZ_SIGNS.items():
        normalized = normalized.replace(sign, word)
    normalized = re.sub(
        r"\d+", lambda match: f" {_number_to_uzbek(int(match.group(0)))} ", normalized
    )
    normalized = re.sub(
        r"\b(" + "|".join(UZ_ABBREVIATIONS) + r")\b",
        lambda match: UZ_ABBREVIATIONS[match.group(1).lower()],
        normalized,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", normalized).strip()


def uzbek_features(text: str) -> dict[str, int]:
    """Слоги, согласные фонемы, слова и знаки — вход модели длительности."""
    normalized = normalize_uzbek(text).lower()
    words = [word for word in re.split(r"[^0-9a-zʻ]+", normalized) if word]

    syllables = 0
    consonants = 0
    long_words = 0
    for word in words:
        # Диграфы sh/ch/ng — одна согласная, а не две.
        collapsed = word
        for digraph in UZ_DIGRAPHS:
            collapsed = collapsed.replace(digraph, "c")
        collapsed = collapsed.replace("gʻ", "g").replace("oʻ", "o")
        nuclei = sum(1 for letter in collapsed if letter in UZ_VOWELS)
        nuclei = nuclei or 1  # слово без гласной всё равно произносится
        syllables += nuclei
        consonants += sum(1 for letter in collapsed if letter.isalpha() and letter not in UZ_VOWELS)
        if nuclei >= 4:
            long_words += 1

    return {
        "syllables": syllables,
        "consonants": consonants,
        "words": len(words),
        "long_words": long_words,
        "commas": text.count(","),
        "breaks": sum(text.count(sign) for sign in STRONG_BREAKS),
    }


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """Гаусс с выбором ведущего элемента — для нормальных уравнений регрессии."""
    size = len(vector)
    rows = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) < 1e-12:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for row in range(column + 1, size):
            factor = rows[row][column] / rows[column][column]
            for position in range(column, size + 1):
                rows[row][position] -= factor * rows[column][position]
    solution = [0.0] * size
    for column in range(size - 1, -1, -1):
        total = rows[column][size] - sum(
            rows[column][position] * solution[position] for position in range(column + 1, size)
        )
        solution[column] = total / rows[column][column]
    return solution


def _fit_duration_model(samples: list[dict[str, Any]]) -> tuple[list[float], float] | None:
    """Ridge-регрессия длительности по измеренным генерациям одного голоса.

    Возвращает коэффициенты [b0, слоги, согласные, слова] и 80-й перцентиль
    остатка — запас, который используем как «квантильный» прогноз против
    перелива (лучше немного недоговорить бюджет, чем вылезти за окно).
    """
    if len(samples) < CALIBRATION_MIN_SAMPLES:
        return None
    design = [
        [1.0, float(item["syllables"]), float(item["consonants"]), float(item["words"])]
        for item in samples
    ]
    target = [float(item["speech"]) for item in samples]
    size = 4
    matrix = [
        [
            sum(row[left] * row[right] for row in design)
            + (CALIBRATION_RIDGE if left == right else 0.0)
            for right in range(size)
        ]
        for left in range(size)
    ]
    vector = [
        sum(row[index] * value for row, value in zip(design, target)) for index in range(size)
    ]
    coefficients = _solve(matrix, vector)
    if coefficients is None or coefficients[1] <= 0:
        return None  # бессмысленная модель (слоги не могут сокращать речь)
    residuals = sorted(
        value - sum(coefficient * feature for coefficient, feature in zip(coefficients, row))
        for row, value in zip(design, target)
    )
    safety = max(0.0, residuals[int(len(residuals) * 0.8)])
    return coefficients, safety


def _load_calibration() -> dict[str, Any]:
    with _calibration_lock:
        if _calibration_cache:
            return _calibration_cache
        try:
            data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        _calibration_cache.update(data if isinstance(data, dict) else {})
        return _calibration_cache


def record_tts_duration(voice: str, text: str, measured: float) -> None:
    """Складывает каждую генерацию в калибровочный набор этого голоса.

    Так предсказание длительности со временем становится точным именно для
    ваших голосов и стиля, а не «в среднем по языку».
    """
    if measured <= 0.2:
        return
    features = uzbek_features(text)
    if features["syllables"] <= 0:
        return
    sample = {
        "syllables": features["syllables"],
        "consonants": features["consonants"],
        "words": features["words"],
        "speech": round(measured, 3),
    }
    with _calibration_lock:
        try:
            data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        samples = data.setdefault(voice, [])
        if isinstance(samples, list):
            samples.append(sample)
            del samples[:-CALIBRATION_MAX_SAMPLES]
        data[voice] = samples
        try:
            CALIBRATION_PATH.write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            return
        _calibration_cache.clear()
        _calibration_cache.update(data)


def duration_model(voice: str) -> tuple[list[float], float]:
    """Коэффициенты для голоса: калибровка, если данных хватает, иначе prior."""
    data = _load_calibration()
    samples = data.get(voice)
    if isinstance(samples, list):
        fitted = _fit_duration_model([item for item in samples if isinstance(item, dict)])
        if fitted:
            return fitted
    return list(PRIOR_COEFFS), 0.0


def predict_speech_duration(text: str, voice: str, safe: bool = False) -> float:
    """Предсказанная длительность произнесения (без внутренних пауз)."""
    features = uzbek_features(text)
    coefficients, safety = duration_model(voice)
    predicted = (
        coefficients[0]
        + coefficients[1] * features["syllables"]
        + coefficients[2] * features["consonants"]
        + coefficients[3] * features["words"]
    )
    if safe:
        predicted += safety
    return max(0.2, predicted)


def predict_total_duration(text: str, voice: str, safe: bool = False) -> float:
    """С учётом пунктуационных пауз — когда фраза озвучивается одним куском."""
    features = uzbek_features(text)
    return (
        predict_speech_duration(text, voice, safe)
        + PAUSE_PER_COMMA * features["commas"]
        + PAUSE_PER_BREAK * max(0, features["breaks"] - 1)
    )


def predict_unit_duration(unit: DubSegment, voice: str, safe: bool = False) -> float:
    """Прогноз по реплике: сумма дыхательных групп (пауз между ними тут нет —
    они берутся из оригинала при раскладке на таймлайне)."""
    parts = [chunk.text for chunk in unit.chunks if chunk.text.strip()]
    if parts:
        return sum(predict_total_duration(part, voice, safe) for part in parts)
    return predict_total_duration(unit.translated_text, voice, safe)


def syllable_budget(seconds: float, voice: str) -> int:
    """Сколько слогов реально влезает в окно — это и есть цель для перевода."""
    coefficients, _ = duration_model(voice)
    # Средние соотношения узбекской речи: ~1.5 согласной и ~0.4 слова на слог.
    per_syllable = coefficients[1] + 1.5 * coefficients[2] + 0.4 * coefficients[3]
    if per_syllable <= 0.01:
        per_syllable = PRIOR_COEFFS[1]
    return max(1, int((max(0.3, seconds) - coefficients[0]) / per_syllable))


CALIBRATION_PHRASE = (
    "Salom, bugun havo juda yaxshi. Keling, birga suhbatlashamiz va yangiliklarni "
    "muhokama qilamiz. Menimcha, bu ajoyib fikr."
)


def calibrate_voice_fingerprints(client: GeminiClient, voices: list[str]) -> dict[str, dict]:
    """Один раз измеряет F0 каждого TTS-голоса на эталонной фразе и кэширует.

    Это то, что превращает подбор голоса из «муж/жен» в «ближайший по тембру к
    актёру»: без отпечатков assign_character_voices работает на двух голосах.
    Требует доступа к TTS, поэтому запускается отдельно (эндпоинт
    /api/calibrate-voices), а не на каждом дубляже.
    """
    fingerprints: dict[str, dict] = load_voice_fingerprints()
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        for voice in voices:
            raw = work / f"{voice}-raw.wav"
            trimmed = work / f"{voice}.wav"
            try:
                client.tts(CALIBRATION_PHRASE, voice, raw, 8.0, "", "", "")
                trim_edge_silence(raw, trimmed)
                samples, rate = _load_mono(trimmed)
            except (RuntimeError, wave.Error, OSError) as exc:
                print(f"[calibrate] {voice}: пропущен ({exc})", flush=True)
                continue
            values = _segment_pitch_values(samples, rate, limit=200)
            if len(values) < 8:
                print(f"[calibrate] {voice}: мало озвученных кадров", flush=True)
                continue
            fingerprints[voice] = {"f0": round(_median(values), 1)}
            print(f"[calibrate] {voice}: F0={fingerprints[voice]['f0']} Гц", flush=True)
    try:
        VOICE_FINGERPRINTS_PATH.write_text(
            json.dumps(fingerprints, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        raise RuntimeError(f"Не удалось сохранить отпечатки голосов: {exc}") from exc
    return fingerprints


def _clean_words(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.!?…:;])", r"\1", text)


def dedupe_repeats(text: str) -> str:
    """Убирает «эхо»: подряд повторённое слово или фразу.

    Модель иногда выдаёт «Ajrashdingizmi? Ajrashdingizmi?» или «Amin emasman...
    Amin emasman» — на слух это выглядит как сбой, а не как игра. Схлопываем
    повтор фразы из 2+ слов и повтор одного длинного слова; короткие
    выразительные удвоения («juda juda») оставляем.
    """
    words = text.split()
    if len(words) < 2:
        return text

    def core(word: str) -> str:
        return re.sub(r"[^0-9a-zʻ]", "", word.lower())

    result: list[str] = []
    position = 0
    while position < len(words):
        collapsed = False
        # Сначала длинные повторы: «a b c a b c» -> «a b c».
        for size in range(min(6, (len(words) - position) // 2), 0, -1):
            first = [core(word) for word in words[position : position + size]]
            second = [core(word) for word in words[position + size : position + 2 * size]]
            if not all(first) or first != second:
                continue
            if size == 1 and len(first[0]) < 5:
                continue  # короткое слово могли повторить намеренно
            kept = list(words[position : position + size])
            # Пунктуацию берём у ВТОРОЙ копии: она стоит на своём месте в фразе,
            # иначе после схлопывания остаётся висящая запятая или многоточие.
            tail = re.search(r"[^0-9A-Za-zʻ]+$", words[position + 2 * size - 1])
            kept[-1] = re.sub(r"[^0-9A-Za-zʻ]+$", "", kept[-1]) + (
                tail.group(0) if tail else ""
            )
            result.extend(kept)
            position += 2 * size
            collapsed = True
            break
        if not collapsed:
            result.append(words[position])
            position += 1
    return _clean_words(" ".join(result))


def _words_from_segment(start: float, end: float, text: str) -> list[Word]:
    """Аварийный разбор: если ASR не дал пословных меток, раскладываем слова
    по длине токенов. Хуже настоящего alignment, но структура не ломается."""
    tokens = [token for token in text.split() if token]
    if not tokens or end <= start:
        return []
    weights = [max(1, len(token)) for token in tokens]
    total = float(sum(weights))
    words: list[Word] = []
    position = start
    for token, weight in zip(tokens, weights):
        span = (end - start) * weight / total
        words.append(Word(start=position, end=position + span, text=token, score=0.0))
        position += span
    return words


def transcribe(audio: Path, language: str | None) -> tuple[list[Word], str]:
    """Уровень 1: слова с временными метками. Никакой нарезки на «фразы» здесь нет.

    Раньше слова тут же склеивались в фразы по 3.2 с и выбрасывались; теперь
    пословные метки живут до самого конца: из них считаются лексическое начало,
    внутренние паузы, границы дыхательных групп и назначение говорящего.
    """
    segments_iterator, info = whisper_model().transcribe(
        str(audio),
        language=language,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=False,
    )

    words: list[Word] = []
    for segment in segments_iterator:
        collected: list[Word] = []
        for word in getattr(segment, "words", None) or []:
            token = (word.word or "").strip()
            if token and word.end > word.start:
                score = getattr(word, "probability", None)
                collected.append(
                    Word(
                        start=float(word.start),
                        end=float(word.end),
                        text=token,
                        score=float(score) if score is not None else 1.0,
                    )
                )
        if collected:
            words.extend(collected)
            continue
        seg_text = (segment.text or "").strip()
        if seg_text and segment.end > segment.start:
            words.extend(_words_from_segment(float(segment.start), float(segment.end), seg_text))

    # Отбрасываем «слова» без букв: ноты, тире, звёздочки субтитров.
    words = [word for word in words if re.search(r"[^\W\d_]", word.text, flags=re.UNICODE)]
    words.sort(key=lambda item: item.start)
    if not words:
        raise RuntimeError("В видео не найдена речь")
    detected = str(getattr(info, "language", "") or language or "")
    return words, detected


_alignment_state: dict[str, bool] = {"applied": False}


def _alignment_enabled() -> bool:
    return os.getenv("FORCED_ALIGNMENT", "auto").strip().lower() not in {"0", "off", "no", "false"}


def forced_align(audio: Path, words: list[Word], language: str) -> list[Word]:
    """Forced alignment (WhisperX/CTC): уточняет границы слов по самому аудио.

    Whisper ставит метки «на глазок» и часто открывает реплику на вздохе или
    смешке за секунду до первого слова. Alignment-модель выравнивает УЖЕ
    известный текст с аудио, поэтому начало реплики попадает на реальный
    первый лексический звук. Если whisperx не установлен — работаем на метках
    Whisper, точность ниже, но пайплайн не падает.
    """
    if not words or not _alignment_enabled():
        return words
    try:
        import whisperx  # type: ignore[import-not-found]
    except Exception:
        print(
            "[dubbing] forced alignment недоступен (нет whisperx) — "
            "работаем на пословных метках Whisper",
            flush=True,
        )
        return words

    groups = group_words(words, max_gap=UTTERANCE_GAP, max_span=UTTERANCE_MAX_SECONDS)
    payload = [
        {
            "start": group[0].start,
            "end": group[-1].end,
            "text": _clean_words(" ".join(word.text for word in group)),
        }
        for group in groups
    ]
    try:
        device = os.getenv("WHISPER_DEVICE", "cpu")
        model, metadata = whisperx.load_align_model(
            language_code=(language or "en")[:2], device=device
        )
        aligned = whisperx.align(payload, model, metadata, str(audio), device)
    except Exception as exc:  # модель языка может отсутствовать — это не фатально
        print(f"[dubbing] forced alignment не выполнен: {exc}", flush=True)
        return words

    result: list[Word] = []
    for item in aligned.get("word_segments") or []:
        token = str(item.get("word", "")).strip()
        start, end = item.get("start"), item.get("end")
        if not token or start is None or end is None or float(end) <= float(start):
            continue
        result.append(
            Word(
                start=float(start),
                end=float(end),
                text=token,
                score=float(item.get("score") or 0.0),
            )
        )
    if len(result) < max(3, int(len(words) * 0.6)):
        print("[dubbing] alignment вернул слишком мало слов — оставляем метки Whisper", flush=True)
        return words
    result.sort(key=lambda item: item.start)
    print(f"[dubbing] forced alignment: уточнено слов {len(result)}", flush=True)
    _alignment_state["applied"] = True
    return result


def group_words(
    words: list[Word],
    max_gap: float,
    max_span: float,
    sentence_flush: float = 0.0,
    boundary: Callable[[Word, Word], bool] | None = None,
) -> list[list[Word]]:
    """Общая нарезка потока слов. Один инструмент — разные параметры для разных
    уровней: черновые utterance для диаризации, translation units, дыхательные
    группы. Раньше все уровни делил один и тот же лимит 3.2 с."""
    groups: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current:
            gap = word.start - current[-1].end
            span = word.end - current[0].start
            ends_sentence = current[-1].text.endswith(SENTENCE_END)
            reached = current[-1].end - current[0].start
            split = gap >= max_gap or span > max_span
            if sentence_flush and ends_sentence and reached >= sentence_flush:
                split = True
            if boundary is not None and boundary(current[-1], word):
                split = True
            if split:
                groups.append(current)
                current = []
        current.append(word)
    if current:
        groups.append(current)
    return groups


def utterances_for_diarization(words: list[Word]) -> list[dict[str, Any]]:
    """Черновые реплики для диаризации: делим только по заметным паузам.

    Диаризацию нельзя кормить обрывками по 0.5-1 с — на таком куске просто нет
    акустических данных. Поэтому здесь куски длиннее, чем translation units.
    """
    groups = group_words(words, max_gap=UTTERANCE_GAP, max_span=UTTERANCE_MAX_SECONDS)
    result: list[dict[str, Any]] = []
    for group in groups:
        text = _clean_words(" ".join(word.text for word in group))
        if not text:
            continue
        result.append(
            {
                "index": len(result),
                "start": group[0].start,
                "end": group[-1].end,
                "text": text,
                "words": group,
            }
        )
    if not result:
        raise RuntimeError("В видео не найдена речь")
    return result


# -----------------------------------------------------------------------------
# Уровень 2: диаризация (кто говорит) — по ВСЕМУ аудио, а не по коротким фразам
# -----------------------------------------------------------------------------


@dataclass
class Diarization:
    """Результат диаризации: speaker turns + участки наложения голосов."""

    turns: list[SpeakerTurn] = field(default_factory=list)
    overlaps: list[tuple[float, float]] = field(default_factory=list)
    genders: dict[str, str] = field(default_factory=dict)  # мнение модели — только подсказка
    backend: str = "none"

    @property
    def speakers(self) -> list[str]:
        return sorted({turn.speaker for turn in self.turns})

    def speaker_at(self, start: float, end: float) -> str:
        """Говорящий с максимальным перекрытием по времени."""
        best_label, best_overlap = "", 0.0
        for turn in self.turns:
            shared = min(end, turn.end) - max(start, turn.start)
            if shared > best_overlap:
                best_label, best_overlap = turn.speaker, shared
        if best_label:
            return best_label
        # Ни один turn не покрывает интервал — берём ближайший по времени.
        nearest, distance = "", float("inf")
        centre = (start + end) / 2
        for turn in self.turns:
            gap = min(abs(turn.start - centre), abs(turn.end - centre))
            if gap < distance:
                nearest, distance = turn.speaker, gap
        return nearest

    def overlap_ratio(self, start: float, end: float) -> float:
        span = max(1e-6, end - start)
        shared = 0.0
        for over_start, over_end in self.overlaps:
            shared += max(0.0, min(end, over_end) - max(start, over_start))
        return min(1.0, shared / span)

    def turn_bounds(self, start: float, end: float) -> tuple[float, float]:
        """Границы turn, внутри которого лежит интервал — за них онсет не выносим."""
        label = self.speaker_at(start, end)
        for turn in self.turns:
            if turn.speaker == label and turn.start - 0.05 <= start and end <= turn.end + 0.05:
                return turn.start, turn.end
        return start, end


def _merge_turns(raw: list[tuple[float, float, str]]) -> list[SpeakerTurn]:
    """Склеивает соседние участки одного говорящего в непрерывный turn.

    Никакого ограничения длины: turn — это столько, сколько человек говорил.
    """
    turns: list[SpeakerTurn] = []
    for start, end, label in sorted(raw, key=lambda item: item[0]):
        if end <= start:
            continue
        if turns and turns[-1].speaker == label and start - turns[-1].end <= TURN_MERGE_GAP:
            turns[-1].end = max(turns[-1].end, end)
            continue
        turns.append(SpeakerTurn(speaker=label, start=start, end=end))
    return turns


def _overlap_regions(raw: list[tuple[float, float, str]]) -> list[tuple[float, float]]:
    """Участки, где одновременно звучат два разных говорящих (cross-talk)."""
    regions: list[tuple[float, float]] = []
    ordered = sorted(raw, key=lambda item: item[0])
    for position, (start, end, label) in enumerate(ordered):
        for other_start, other_end, other_label in ordered[position + 1 :]:
            if other_start >= end:
                break
            if other_label == label:
                continue
            shared_start, shared_end = max(start, other_start), min(end, other_end)
            if shared_end - shared_start >= 0.10:  # короче 100 мс — не считаем
                regions.append((shared_start, shared_end))
    regions.sort()
    merged: list[tuple[float, float]] = []
    for start, end in regions:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _patch_huggingface_hub_use_auth_token_compat() -> None:
    """Совместимость pyannote.audio 3.x со свежим huggingface_hub (>=1.0).

    huggingface_hub v1.0 убрал параметр use_auth_token у hf_hub_download и
    родственных функций (заменён на token). pyannote.audio 3.4.0 передаёт
    use_auth_token напрямую в Pipeline.from_pretrained и дальше во внутренние
    вызовы hf_hub_download -> "unexpected keyword argument 'use_auth_token'".
    Оборачиваем сами функции huggingface_hub, а не правим сторонний код
    pyannote — так работает независимо от того, где внутри pyannote всплывёт
    этот параметр.
    """
    try:
        import huggingface_hub  # type: ignore[import-not-found]
    except Exception:
        return
    for name in ("hf_hub_download", "snapshot_download"):
        original = getattr(huggingface_hub, name, None)
        if original is None or getattr(original, "_use_auth_token_patched", False):
            continue

        def make_wrapper(func):
            def wrapper(*args, **kwargs):
                if "use_auth_token" in kwargs:
                    kwargs["token"] = kwargs.pop("use_auth_token")
                return func(*args, **kwargs)

            wrapper._use_auth_token_patched = True
            return wrapper

        setattr(huggingface_hub, name, make_wrapper(original))
    print("[dubbing] добавлена заглушка use_auth_token->token для huggingface_hub", flush=True)


def _patch_torchaudio_audiometadata_compat() -> None:
    """Совместимость pyannote.audio 3.x со свежим torchaudio (>=2.9).

    torchaudio.AudioMetaData был deprecated в 2.8 и удалён в 2.9+, а
    pyannote.audio 3.4.0 всё ещё импортирует его напрямую из torchaudio ->
    "module 'torchaudio' has no attribute 'AudioMetaData'". Возвращаем
    отсутствующий класс на место лёгкой заглушкой перед импортом pyannote,
    не понижая версию torchaudio/torch (это потянуло бы конфликт версий).
    """
    try:
        import torchaudio  # type: ignore[import-not-found]
    except Exception:
        return
    if hasattr(torchaudio, "AudioMetaData"):
        return
    try:
        from dataclasses import dataclass as _dataclass

        @_dataclass
        class AudioMetaData:  # noqa: N801 - имя обязано совпадать со старым API
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int = 0
            encoding: str = ""

        torchaudio.AudioMetaData = AudioMetaData
        print("[dubbing] добавлена заглушка torchaudio.AudioMetaData для pyannote", flush=True)
    except Exception:
        pass

    # torchaudio 2.9+ вместе с AudioMetaData убрал и list_audio_backends —
    # pyannote/зависимости иногда опрашивают список бэкендов при старте.
    if not hasattr(torchaudio, "list_audio_backends"):
        try:
            torchaudio.list_audio_backends = lambda: ["soundfile"]
        except Exception:
            pass


def _extract_pyannote_tracks(result: Any) -> list[tuple[float, float, str]]:
    """Достаёт (start, end, speaker_label) из результата pyannote-пайплайна.

    pyannote.audio 3.x возвращал pyannote.core.Annotation (метод
    .itertracks(yield_label=True)). pyannote.audio 4.x оборачивает результат в
    новый класс DiarizeOutput с несколькими режимами (обычный/exclusive) —
    сам Annotation лежит внутри одного из его атрибутов. Перебираем известные
    варианты по порядку; если ни один не подошёл, печатаем структуру объекта,
    чтобы это можно было точно диагностировать без гадания.
    """
    annotation = result
    if not hasattr(annotation, "itertracks"):
        for attr in (
            "speaker_diarization",
            "exclusive_speaker_diarization",
            "annotation",
            "diarization",
        ):
            candidate = getattr(result, attr, None)
            if candidate is not None and hasattr(candidate, "itertracks"):
                annotation = candidate
                break
    if not hasattr(annotation, "itertracks"):
        available = [name for name in dir(result) if not name.startswith("_")]
        raise RuntimeError(
            f"неизвестный формат результата pyannote (тип {type(result).__name__}, "
            f"атрибуты: {available})"
        )
    return [
        (float(segment.start), float(segment.end), str(label))
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]


def pyannote_diarization(audio: Path) -> Diarization | None:
    """Диаризация pyannote по всему файлу — рекомендуемый путь.

    Модель даёт speaker turns, embeddings-кластеризацию и разметку наложений,
    то есть именно то, чего не может дать LLM по короткой реплике. Запускается
    на ЦЕЛОМ аудио: короткая реплика наследует личность говорящего от его
    длинных реплик, а не угадывается заново.
    """
    backend = os.getenv("DIARIZATION_BACKEND", "auto").strip().lower()
    if backend not in {"auto", "pyannote"}:
        return None
    _patch_torchaudio_audiometadata_compat()
    _patch_huggingface_hub_use_auth_token_compat()
    try:
        from pyannote.audio import Pipeline  # type: ignore[import-not-found]
    except Exception as exc:
        # ВАЖНО: печатаем причину всегда, а не только при backend="pyannote".
        # Раньше при DIARIZATION_BACKEND=auto (по умолчанию) любая ошибка
        # импорта — не только "не установлен", но и реальный сбой версий —
        # проглатывалась без единого сообщения в лог.
        print(f"[dubbing] pyannote.audio не импортировался ({exc}) — диаризация через Gemini", flush=True)
        return None

    model = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1")
    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or "").strip() or None
    options: dict[str, int] = {}
    exact = os.getenv("NUM_SPEAKERS", "").strip()
    if exact.isdigit() and int(exact) > 0:
        # Если число говорящих известно — задаём точно, это заметно снижает DER.
        options["num_speakers"] = int(exact)
    else:
        minimum = os.getenv("MIN_SPEAKERS", "").strip()
        maximum = os.getenv("MAX_SPEAKERS", "").strip()
        if minimum.isdigit():
            options["min_speakers"] = int(minimum)
        if maximum.isdigit():
            options["max_speakers"] = int(maximum)
    try:
        # Разные версии pyannote.audio ожидают либо token=, либо use_auth_token=
        # в своей публичной сигнатуре — пробуем оба, чтобы не зависеть от версии.
        try:
            pipeline = Pipeline.from_pretrained(model, token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(model, use_auth_token=token)
        if pipeline is None:
            raise RuntimeError("pyannote не отдал pipeline (нужен доступ к модели и HF_TOKEN)")
        result = pipeline(str(audio), **options)
        raw = _extract_pyannote_tracks(result)
    except Exception as exc:
        print(f"[dubbing] pyannote не сработал ({exc}) — диаризация через Gemini", flush=True)
        return None
    if not raw:
        return None
    return Diarization(
        turns=_merge_turns(raw),
        overlaps=_overlap_regions(raw),
        backend="pyannote",
    )


def smooth_labels(
    utterances: list[dict[str, Any]], labels: dict[int, str]
) -> dict[int, str]:
    """Temporal smoothing: одиночная короткая реплика не может «сменить» говорящего.

    Именно этот случай ломался раньше: на 0.5-2 с модель угадывала тембр и
    выдавала нового говорящего посреди монолога. Если сосед слева и справа —
    один и тот же человек, короткая реплика достаётся ему.
    """
    smoothed = dict(labels)
    for position, item in enumerate(utterances):
        index = item["index"]
        current = smoothed.get(index)
        if not current:
            continue
        duration = float(item["end"]) - float(item["start"])
        if duration > SHORT_TURN_SECONDS:
            continue
        previous = smoothed.get(utterances[position - 1]["index"]) if position else None
        following = (
            smoothed.get(utterances[position + 1]["index"])
            if position + 1 < len(utterances)
            else None
        )
        if previous and previous == following and previous != current:
            smoothed[index] = previous
    # Реплика без метки наследует метку предыдущей — «unknown» лучше не плодить.
    last = ""
    for item in utterances:
        index = item["index"]
        if smoothed.get(index):
            last = smoothed[index]
        elif last:
            smoothed[index] = last
    return smoothed


def gemini_diarization(
    client: GeminiClient, audio: Path, utterances: list[dict[str, Any]]
) -> Diarization:
    """Резервная диаризация: Gemini слушает всё аудио и метит черновые реплики.

    Слабее pyannote (нет frame-level posterior, embeddings и разметки наложений),
    поэтому сверху обязательно идёт temporal smoothing.
    """
    labels, genders = client.identify_speakers(audio, utterances)
    if not labels:
        return Diarization(backend="none")
    labels = smooth_labels(utterances, labels)
    raw = [
        (float(item["start"]), float(item["end"]), labels[item["index"]])
        for item in utterances
        if labels.get(item["index"])
    ]
    if not raw:
        return Diarization(backend="none")
    return Diarization(turns=_merge_turns(raw), genders=genders, backend="gemini")


def diarize(
    client: GeminiClient, audio: Path, duration: float, utterances: list[dict[str, Any]]
) -> Diarization:
    result = pyannote_diarization(audio)
    if result is None or not result.turns:
        result = gemini_diarization(client, audio, utterances)
    if not result.turns:
        # Совсем ничего не получилось — считаем, что говорящий один.
        result = Diarization(
            turns=[SpeakerTurn(speaker="S1", start=0.0, end=max(duration, 0.1))],
            backend="single",
        )
    for turn in result.turns:
        turn.overlap = result.overlap_ratio(turn.start, turn.end)
    return result


def assign_word_speakers(words: list[Word], diarization: Diarization) -> None:
    for word in words:
        word.speaker = diarization.speaker_at(word.start, word.end)


# -----------------------------------------------------------------------------
# Уровни 3 и 4: translation units и дыхательные группы
# -----------------------------------------------------------------------------

CLAUSE_END = (",", ";", ":", "—", "-")


def build_chunks(words: list[Word]) -> list[DubChunk]:
    """Режет unit на дыхательные группы по РЕАЛЬНЫМ паузам между словами.

    Пауза оригинала здесь не «съедается» и не отдаётся на волю TTS: каждая
    группа потом ставится на своё лексическое начало, поэтому пауза
    воспроизводится ровно той длины, которую сделал актёр.
    """
    groups = group_words(words, max_gap=CHUNK_PAUSE_MIN, max_span=CHUNK_MAX_SECONDS)
    chunks: list[DubChunk] = []
    for group in groups:
        text = _clean_words(" ".join(word.text for word in group))
        if not text:
            continue
        chunks.append(
            DubChunk(
                start=group[0].start,
                end=group[-1].end,
                source_text=text,
                onset=group[0].start,
            )
        )
    return chunks


def has_lexical_content(text: str) -> bool:
    """Есть ли в строке настоящие слова.

    Распознавание регулярно выдаёт «реплики» без слов: музыкальные символы,
    многоточия, отдельные знаки. Переводить и озвучивать там нечего.
    """
    # Достаточно одной буквы: короткие «Ha», «A?», «Yoq» — настоящая речь, и
    # терять их нельзя. Отсекаем только то, где букв нет вовсе.
    return bool(re.search(r"[^\W\d_]", text, flags=re.UNICODE))


def _apply_translation(unit: DubSegment, payload: dict[str, Any] | None) -> bool:
    """Раскладывает ответ переводчика по реплике. False — переводить нечего."""
    if not payload:
        return False
    translated = dedupe_repeats(normalize_uzbek(str(payload.get("translated_text", "")).strip()))
    if not has_lexical_content(translated):
        return False
    unit.translated_text = translated
    unit.short_variant = dedupe_repeats(
        normalize_uzbek(str(payload.get("short_variant", "")).strip())
    )
    unit.style = str(payload.get("style", "")).strip()[:MAX_STYLE_LEN]

    parts = payload.get("parts")
    texts: list[str] = []
    if isinstance(parts, list) and len(parts) == len(unit.chunks):
        texts = [dedupe_repeats(normalize_uzbek(str(part).strip())) for part in parts]
    if not all(text.strip() for text in texts):
        texts = split_text_by_chunks(translated, unit.chunks)
    for chunk, text in zip(unit.chunks, texts):
        chunk.text = text
    # Модель иногда дублирует один и тот же текст в двух частях — тогда одна
    # фраза звучала бы подряд дважды. В этом случае раскладываем сами.
    if any(
        following.text and following.text == previous.text
        for previous, following in zip(unit.chunks, unit.chunks[1:])
    ):
        for chunk, text in zip(unit.chunks, split_text_by_chunks(translated, unit.chunks)):
            chunk.text = text
    if not any(chunk.text.strip() for chunk in unit.chunks):
        # Раскладка не удалась — произносим реплику одним куском, но не молчим.
        unit.chunks[0].text = translated
    return True


def split_text_by_chunks(text: str, chunks: list[DubChunk]) -> list[str]:
    """Резервная раскладка перевода по дыхательным группам.

    Используется, если модель не вернула готовые части: делим по словам
    пропорционально длительности групп, стараясь попасть на знак препинания.
    """
    if len(chunks) <= 1:
        return [text]
    words = text.split()
    if len(words) < len(chunks):
        # Слов меньше, чем групп: озвучиваем всё первой группой, остальные молчат.
        return [text] + [""] * (len(chunks) - 1)

    total = sum(chunk.duration for chunk in chunks) or 1.0
    parts: list[str] = []
    position = 0
    for number, chunk in enumerate(chunks):
        remaining_chunks = len(chunks) - number - 1
        if remaining_chunks == 0:
            parts.append(" ".join(words[position:]))
            break
        share = chunk.duration / total
        take = max(1, round(share * len(words)))
        take = min(take, len(words) - position - remaining_chunks)
        # Подтягиваем границу к ближайшему знаку препинания (±1 слово).
        for shift in (0, 1, -1):
            candidate = position + take + shift
            if position < candidate < len(words) - remaining_chunks:
                if words[candidate - 1].rstrip().endswith(CLAUSE_END + SENTENCE_END):
                    take += shift
                    break
        parts.append(" ".join(words[position : position + take]))
        position += take
    return parts


def build_units(words: list[Word], diarization: Diarization) -> list[DubSegment]:
    """Собирает translation units: смысловой блок ~4-12 с внутри ОДНОГО говорящего.

    Границы: смена говорящего, пауза >= UNIT_SPLIT_GAP, конец предложения после
    UNIT_FLUSH_SECONDS, запятая после UNIT_TARGET_SECONDS, жёсткий предел
    UNIT_MAX_SECONDS. Ни одного искусственного лимита в 3.2 с больше нет —
    перевод получает целое предложение, а не обрубок.
    """

    def boundary(previous: Word, following: Word) -> bool:
        if previous.speaker != following.speaker:
            return True  # unit никогда не пересекает границу speaker turn
        return False

    groups: list[list[Word]] = []
    for group in group_words(
        words,
        max_gap=UNIT_SPLIT_GAP,
        max_span=UNIT_MAX_SECONDS,
        sentence_flush=UNIT_FLUSH_SECONDS,
        boundary=boundary,
    ):
        # Длинный блок без точек дорезаем по запятой, чтобы не уехать к 12 с.
        current: list[Word] = []
        for word in group:
            if (
                current
                and word.end - current[0].start > UNIT_TARGET_SECONDS
                and current[-1].text.rstrip().endswith(CLAUSE_END)
            ):
                groups.append(current)
                current = []
            current.append(word)
        if current:
            groups.append(current)

    units: list[DubSegment] = []
    for group in groups:
        text = _clean_words(" ".join(word.text for word in group))
        # Реплики без слов (музыка, «♪», отдельные знаки, обрывки шума) в дубляж
        # не идут: переводить там нечего, а тишина под них не нужна.
        if not has_lexical_content(text):
            continue
        chunks = build_chunks(group)
        if not chunks:
            continue
        start, end = group[0].start, group[-1].end
        label = diarization.speaker_at(start, end) or "S1"
        units.append(
            DubSegment(
                index=len(units),
                start=start,
                end=end,
                source_text=text,
                speaker_label=label,
                chunks=chunks,
                overlap=diarization.overlap_ratio(start, end),
            )
        )
    if not units:
        raise RuntimeError("В видео не найдена речь")
    return units


def assign_chunk_budgets(units: list[DubSegment], duration: float) -> None:
    """Бюджет каждой дыхательной группы: до начала следующей группы (любого unit).

    Считаем по всей ленте, а не внутри реплики: так узбекская группа не
    наезжает на следующую и при этом договаривается до конца.
    """
    chunks = sorted(
        (chunk for unit in units for chunk in unit.chunks), key=lambda item: item.onset
    )
    for position, chunk in enumerate(chunks):
        if position + 1 < len(chunks):
            limit = chunks[position + 1].onset
        else:
            limit = duration
        window = max(0.4, limit - chunk.onset)
        # Небольшой хвост за окно допустим (слова договариваются), но тянуться
        # дольше, чем говорил человек, реплика не должна.
        chunk.budget = max(0.4, min(window, chunk.duration + TAIL_TOLERANCE))


@dataclass
class Placement:
    """Куда встала дыхательная группа и что при этом пришлось сделать."""

    key: tuple[int, int]
    start: float
    length: float
    ducked: bool = False  # осталось неизбежное наложение — перебиваемого приглушаем

    @property
    def end(self) -> float:
        return self.start + self.length


def schedule_chunks(
    items: list[tuple[tuple[int, int], float, float, str, float, float, bool]],
) -> list[Placement]:
    """Планировщик размещения реплик на таймлайне (замена жадного сдвига).

    Вход по каждой группе: (key, ideal_start, length, speaker, src_start,
    src_end, onscreen), где src_start/src_end — границы реплики в ОРИГИНАЛЕ.

    Правила уступок: реплику НИКОГДА не начинаем раньше её лексического начала и
    НИКОГДА не обрываем. Чтобы двое не звучали разом, ответ сдвигаем вправо — но
    не дальше допуска (в кадре 0.15 с, за кадром 0.40 с). Если и сдвига не
    хватает, наложение оставляем и приглушаем перебиваемого (ducked=True), а не
    двигаем без предела — так избегаем и «наездов», и лавины сдвигов до конца
    сцены. Наложение, которое БЫЛО в оригинале, сохраняется как есть.
    """
    ordered = sorted(items, key=lambda item: item[1])
    placements: list[Placement] = []
    previous_end: float | None = None
    previous_speaker = ""
    previous_src_end: float | None = None
    for key, ideal, length, speaker, src_start, src_end, onscreen in ordered:
        src_overlap = (
            previous_src_end is not None and src_start < previous_src_end - SOURCE_OVERLAP_EPS
        )
        ducked = False
        if previous_end is None or src_overlap:
            # Первая реплика или наложение из оригинала — ставим на своё место.
            start = ideal
        else:
            gap = SAME_SPEAKER_GAP if speaker == previous_speaker else SPEAKER_GAP_MIN
            earliest = previous_end + gap
            if earliest <= ideal:
                start = ideal
            else:
                shift_cap = ideal + (MAX_SHIFT_ONSCREEN if onscreen else MAX_SHIFT_OFFSCREEN)
                start = min(earliest, shift_cap)
                # Даже после предельного сдвига осталось наложение — приглушаем.
                if start + 1e-6 < previous_end + SPEAKER_GAP_HARD:
                    ducked = True
        placements.append(Placement(key=key, start=max(0.0, start), length=length, ducked=ducked))
        previous_end = max(previous_end or 0.0, start + length)
        previous_speaker = speaker
        previous_src_end = max(previous_src_end or 0.0, src_end)
    return placements


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


_stretch_filter_cache: dict[str, bool] = {}


def has_rubberband() -> bool:
    """Есть ли в сборке FFmpeg фильтр rubberband.

    Обычный phase vocoder размывает транзиенты и согласные; rubberband с
    обработкой транзиентов звучит заметно лучше при растяжении больше пары
    процентов. Если его нет — остаёмся на atempo в безопасном диапазоне.
    """
    if "rubberband" not in _stretch_filter_cache:
        try:
            result = run_command(["ffmpeg", "-hide_banner", "-filters"], timeout=30)
            _stretch_filter_cache["rubberband"] = bool(
                re.search(r"^\s*\S+\s+rubberband\s", result.stdout, re.MULTILINE)
            )
        except RuntimeError:
            _stretch_filter_cache["rubberband"] = False
    return _stretch_filter_cache["rubberband"]


def stretch_filter(ratio: float) -> str:
    if has_rubberband():
        return f"rubberband=tempo={ratio:.6f}:pitchq=quality:transients=crisp"
    return atempo_filter(ratio)


def fit_ratio(actual: float, budget: float, source_seconds: float, speech_speed: float) -> float:
    """Сколько нужно изменить темп — с жёсткими рабочими границами.

    Растяжение здесь — ПОСЛЕДНЯЯ ступень: текст уже подогнан по слогам, а TTS
    уже просили говорить быстрее. Поэтому диапазон узкий: 0.92-1.15, и только
    для коротких реплик (<2.5 с) разрешён аварийный предел 1.25 — на них
    ускорение почти не слышно, а вылет за окно слышен сразу.
    """
    ratio = 1.0
    if actual > budget:
        ratio = actual / max(budget, 0.2)
    elif actual < source_seconds * 0.92:
        # Реплика короче окна: чуть растягиваем, чтобы губы не «доигрывали» молча.
        ratio = actual / max(source_seconds, 0.25)
    ceiling = EMERGENCY_SPEED_UP if source_seconds < 2.5 else MAX_SPEED_UP_RATIO
    ratio = min(max(ratio, MIN_SLOWDOWN_RATIO), ceiling)
    # Пользовательская скорость речи — поверх, но всё ещё в разумных пределах.
    return min(max(ratio * speech_speed, 0.85), max(ceiling, 1.15))


def normalize_and_fit(
    source: Path,
    output: Path,
    target_seconds: float,
    max_seconds: float,
    speech_speed: float = 1.0,
) -> float:
    """Финальная подгонка длины. Возвращает применённое изменение темпа."""
    actual = media_duration(source)
    ratio = fit_ratio(actual, max(max_seconds, 0.3), target_seconds, speech_speed)

    chain: list[str] = []
    if abs(ratio - 1.0) > 0.02:
        chain.append(stretch_filter(ratio))
    final_length = actual / ratio
    # Более длинные микрофейды полностью убирают щелчки на стыках реплик.
    chain.append("afade=t=in:st=0:d=0.02")
    chain.append(f"afade=t=out:st={max(0.0, final_length - 0.08):.3f}:d=0.08")
    run_command(
        [
            "ffmpeg", "-y", "-i", str(source), "-af", ",".join(chain),
            "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(output),
        ],
        timeout=180,
    )
    return ratio


def read_mono_pcm(path: Path) -> array:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 24000:
            raise RuntimeError("Внутренняя ошибка формата WAV")
        samples = array("h")
        samples.frombytes(wav.readframes(wav.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


TARGET_CLIP_RMS = 3600.0  # целевая громкость каждой реплики (из 32767)
MAX_CLIP_GAIN = 3.5       # выше — звук перегружается и хрипит
PEAK_CEILING = 29000.0    # запас до предела, чтобы не было клиппинга


def normalize_clip_level(clip: array) -> array:
    """Выравнивает громкость реплик, но без перегрузки и хрипа.

    Раньше тихие реплики (часто женские) усиливались слишком сильно и звучали
    искажённо. Теперь усиление ограничено и дополнительно проверяется по пику.
    """
    if not clip:
        return clip
    energy = 0.0
    peak = 1.0
    for value in clip:
        sample = float(value)
        energy += sample * sample
        if abs(sample) > peak:
            peak = abs(sample)
    rms = (energy / len(clip)) ** 0.5
    if rms < 1.0:
        return clip
    # Не превышаем ни целевую громкость, ни безопасный пик.
    gain = min(TARGET_CLIP_RMS / rms, PEAK_CEILING / peak, MAX_CLIP_GAIN)
    if abs(gain - 1.0) < 0.05:
        return clip
    return array(
        "h", [max(-32768, min(32767, int(value * gain))) for value in clip]
    )


def trim_edge_silence(source: Path, output: Path) -> None:
    """Убирает случайную тишину В НАЧАЛЕ И В КОНЦЕ сгенерированной группы.

    Паузы задаёт фильм, а не TTS: каждая дыхательная группа ставится на своё
    лексическое начало, поэтому её собственные краевые тишины только сбивают
    синхрон. Внутренние паузы группы при этом не трогаем.
    """
    run_command(
        [
            "ffmpeg", "-y", "-i", str(source),
            # Хвост режем через areverse: stop_periods=-1 вырезал бы и ПАУЗЫ
            # ВНУТРИ группы, а они часть игры и синхрона.
            "-af",
            "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.02,"
            "areverse,"
            "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,"
            "areverse",
            "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(output),
        ],
        timeout=120,
    )
    # Если после обрезки файл пустой (вся реплика тихая) — вернём исходный.
    try:
        if media_duration(output) < 0.05:
            shutil.copyfile(source, output)
    except RuntimeError:
        shutil.copyfile(source, output)


def _generate_chunk(
    client: GeminiClient,
    segment: DubSegment,
    chunk: DubChunk,
    position: int,
    voice_map: dict[str, str],
    scene: str,
    segments_dir: Path,
    speech_speed: float = 1.0,
    simple: bool = False,
) -> tuple[Path, str, str]:
    """Озвучивает одну дыхательную группу с контекстом всей реплики.

    `simple=True` — аварийный режим последнего захода: без сцены, ремарки и
    контекста. Такой запрос модель почти всегда выполняет, а тишина в дубляже
    хуже, чем менее выразительная подача.

    Порядок борьбы с переливом строго такой:
      1) текст уже подогнан под слоговой бюджет (condense_to_budget);
      2) если сгенерированное всё равно длиннее бюджета более чем на 15% —
         просим TTS говорить быстрее и, если есть, берём короткий вариант;
      3) только остаток добираем растяжением в диапазоне 0.92-1.15.
    """
    name = f"{segment.index:04d}-{position:02d}"
    raw = segments_dir / f"{name}-raw.wav"
    trimmed = segments_dir / f"{name}-trim.wav"
    fitted = segments_dir / f"{name}.wav"
    # Голос персонажа выбран один раз на говорящего (assign_character_voices);
    # voice_map — резерв, если конкретный голос не назначен.
    voice = segment.voice or voice_map.get(segment.speaker, voice_map["female"])
    # ВАЖНО: если у части нет своего текста, она молчит — но НЕ произносит всю
    # реплику целиком. Иначе одна и та же фраза звучала дважды подряд.
    spoken_text = chunk.text.strip()
    if not spoken_text:
        if position == 0:
            spoken_text = segment.translated_text
        else:
            raise RuntimeError("У части реплики нет текста для озвучки")
    budget = max(0.4, chunk.budget or chunk.duration)
    predicted = predict_total_duration(spoken_text, voice)

    client.tts(
        spoken_text,
        voice,
        raw,
        budget,
        "" if simple else segment.style,
        "" if simple else scene,
        segment.speaker,
        context="" if simple else segment.translated_text,
    )
    trim_edge_silence(raw, trimmed)
    measured = media_duration(trimmed)
    record_tts_duration(voice, spoken_text, measured)

    # Две причины переозвучить: (1) TTS «зациклился» и повторил фразу — это видно
    # по длительности вдвое больше прогноза; (2) реплика не влезает в окно.
    looped = measured > max(predicted * LOOP_DURATION_FACTOR, predicted + 1.0)
    if looped or measured > budget * REGENERATE_OVERFLOW:
        retry_text = spoken_text
        if (
            not looped
            and len(segment.chunks) == 1
            and segment.short_variant
            and predict_speech_duration(segment.short_variant, voice)
            < predict_speech_duration(spoken_text, voice)
        ):
            retry_text = segment.short_variant
        retry_raw = segments_dir / f"{name}-raw2.wav"
        retry_trimmed = segments_dir / f"{name}-trim2.wav"
        try:
            client.tts(
                retry_text,
                voice,
                retry_raw,
                budget,
                "" if simple else segment.style,
                "" if simple else scene,
                segment.speaker,
                # При зацикливании убираем контекст: он и провоцирует повтор.
                context="" if (simple or looped) else segment.translated_text,
                pace="normal" if looped else "faster",
            )
            trim_edge_silence(retry_raw, retry_trimmed)
            retry_measured = media_duration(retry_trimmed)
            record_tts_duration(voice, retry_text, retry_measured)
            if retry_measured < measured:
                trimmed, measured, spoken_text = retry_trimmed, retry_measured, retry_text
                if looped:
                    print(
                        f"[dubbing] переозвучена зацикленная фраза {name} "
                        f"({measured:.1f}с вместо прогноза {predicted:.1f}с)",
                        flush=True,
                    )
        except RuntimeError:
            pass  # перегенерация необязательна: остаётся первый вариант

    chunk.generated = measured
    chunk.ratio = normalize_and_fit(trimmed, fitted, chunk.duration, budget, speech_speed)
    return fitted, spoken_text, voice


def render_timeline(
    duration: float,
    segments: list[DubSegment],
    client: GeminiClient,
    voice_map: dict[str, str],
    scene: str,
    work_dir: Path,
    progress: ProgressCallback,
    speech_speed: float = 1.0,
) -> Path:
    sample_rate = 24000
    # Запас в конце: реплики больше не обрезаются и могут выходить за окно.
    timeline = array("h", [0]) * (int(duration * sample_rate) + sample_rate * 12)
    segments_dir = work_dir / "segments"
    segments_dir.mkdir(exist_ok=True)
    transcript: list[dict[str, Any]] = []

    ordered = sorted(segments, key=lambda s: s.start)
    # Озвучиваем ДЫХАТЕЛЬНЫМИ ГРУППАМИ: каждая ставится на своё лексическое
    # начало, поэтому внутренние паузы оригинала воспроизводятся как есть, а не
    # отдаются на усмотрение TTS.
    tasks: list[tuple[DubSegment, int, DubChunk]] = [
        (unit, position, chunk)
        for unit in ordered
        for position, chunk in enumerate(unit.chunks)
        if chunk.text.strip()
    ]
    if not tasks:
        raise RuntimeError("Нет текста для озвучки")

    males = sum(1 for unit in ordered if unit.speaker == "male")
    print(
        f"[dubbing] реплик: {len(ordered)} (низкий регистр {males}, "
        f"высокий {len(ordered) - males}), дыхательных групп: {len(tasks)}",
        flush=True,
    )
    for unit in ordered:
        mark = "М" if unit.speaker == "male" else "Ж"
        voice = voice_map.get(unit.speaker, voice_map["female"])
        predicted = predict_speech_duration(unit.translated_text, voice)
        print(
            f"[dubbing]   {unit.start:6.2f}s {mark} {unit.speaker_label:>3} "
            f"групп={len(unit.chunks)} окно={unit.speech_budget:4.1f}с "
            f"прогноз={predicted:4.1f}с "
            f"слогов={uzbek_features(unit.translated_text)['syllables']:3d} "
            f"{unit.translated_text[:40]}",
            flush=True,
        )

    generated: dict[tuple[int, int], tuple[Path, str, str]] = {}
    errors: dict[tuple[int, int], str] = {}
    retry: list[tuple[DubSegment, int, DubChunk]] = []
    total = len(tasks)
    done = 0
    workers = max(1, min(TTS_CONCURRENCY, total))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tts") as pool:
        futures = {
            pool.submit(
                _generate_chunk,
                client,
                unit,
                chunk,
                position,
                voice_map,
                scene,
                segments_dir,
                speech_speed,
            ): (unit, position, chunk)
            for unit, position, chunk in tasks
        }
        for future in as_completed(futures):
            unit, position, chunk = futures[future]
            try:
                generated[(unit.index, position)] = future.result()
            except Exception as exc:
                retry.append((unit, position, chunk))
                errors[(unit.index, position)] = str(exc)
            done += 1
            progress(45 + round(done / total * 40), f"Озвучено {done} из {total} фраз")

    # Дальше — упорные повторные заходы. Пропущенная реплика означает, что в
    # этом месте останется оригинальная английская речь, то есть «дырявый»
    # дубляж со смесью языков. Это хуже любой другой ошибки, поэтому пробуем
    # последовательно, с растущей паузой, а последний заход — в упрощённом
    # режиме (без сцены, ремарки и контекста).
    for attempt in range(2, TTS_PASSES + 1):
        if not retry:
            break
        pending, retry = sorted(retry, key=lambda item: item[2].onset), []
        for number, (unit, position, chunk) in enumerate(pending, start=1):
            progress(
                86,
                f"Повторная озвучка пропущенных фраз {number} из {len(pending)} "
                f"(заход {attempt} из {TTS_PASSES})",
            )
            time.sleep(1.5 * attempt)
            try:
                generated[(unit.index, position)] = _generate_chunk(
                    client,
                    unit,
                    chunk,
                    position,
                    voice_map,
                    scene,
                    segments_dir,
                    speech_speed,
                    simple=attempt >= TTS_PASSES,
                )
                errors.pop((unit.index, position), None)
            except Exception as exc:
                errors[(unit.index, position)] = str(exc)
                retry.append((unit, position, chunk))

    if not generated:
        first = "; ".join(list(errors.values())[:3])
        raise RuntimeError(f"Не удалось озвучить ни одну реплику. {first}")

    speech_total = sum(chunk.duration for _, _, chunk in tasks)
    missing_seconds = 0.0
    if errors:
        lines: list[str] = []
        for unit, position, chunk in sorted(tasks, key=lambda item: item[2].onset):
            message = errors.get((unit.index, position))
            if not message:
                continue
            missing_seconds += chunk.duration
            lines.append(
                f"{chunk.onset:8.2f}s-{chunk.end:.2f}s остался оригинал "
                f"({unit.speaker_label}): {chunk.text[:50]} | {message}"
            )
        report = "\n".join(lines)
        (work_dir / "skipped.txt").write_text(report + "\n", encoding="utf-8")
        print(
            f"[dubbing] НЕ озвучено фраз: {len(errors)} "
            f"({missing_seconds:.1f}с из {speech_total:.1f}с речи)\n{report}",
            flush=True,
        )
        # Смесь языков — не «частичный успех», а брак: лучше явная ошибка с
        # таймкодами, чем видео, где половина диалога осталась по-английски.
        # Одну короткую фразу пережить можно, но не куски по несколько секунд.
        if (
            speech_total > 0
            and missing_seconds > 1.5
            and missing_seconds / speech_total > MAX_MISSING_SPEECH
        ):
            first = "; ".join(sorted({message for message in errors.values()})[:2])
            raise RuntimeError(
                f"Дубляж получился «дырявым»: {missing_seconds:.0f}с речи из "
                f"{speech_total:.0f}с остались бы на оригинальном языке "
                f"({len(errors)} фраз). Причина: {first}. Запустите снова или "
                "снизьте TTS_CONCURRENCY — смешивать языки в одном ролике нельзя."
            )
    skipped_count = len({index for index, _ in errors})

    # Планируем размещение сразу для всех озвученных групп (просмотр вперёд по
    # всей сцене), а не жадно по одной — так реплики не «наезжают» и не
    # накапливают лавину сдвигов.
    clips: dict[tuple[int, int], array] = {}
    schedule_input: list[tuple[tuple[int, int], float, float, str, float, float, bool]] = []
    for unit, position, chunk in tasks:
        key = (unit.index, position)
        if key not in generated:
            continue
        fitted, _, _ = generated[key]
        clip = normalize_clip_level(read_mono_pcm(fitted))
        clips[key] = clip
        # «В кадре» считаем реплику без наложения в оригинале: там важен губ-синк,
        # поэтому сдвиг жёстко ограничен; закадровую/реакцию двигаем свободнее.
        onscreen = unit.overlap < 0.2
        schedule_input.append(
            (
                key,
                max(0.0, chunk.onset + ONSET_OFFSET),
                len(clip) / sample_rate,
                unit.speaker_label,
                chunk.start,
                chunk.end,
                onscreen,
            )
        )
    placements = {item.key: item for item in schedule_chunks(schedule_input)}

    stretched = 0
    ducked_count = 0
    for unit, position, chunk in sorted(tasks, key=lambda item: item[2].onset):
        key = (unit.index, position)
        placement = placements.get(key)
        if placement is None or key not in clips:
            continue
        _, spoken_text, voice = generated[key]
        clip = clips[key]
        start_sample = max(0, int(placement.start * sample_rate))
        available = min(len(clip), len(timeline) - start_sample)
        for index in range(available):
            sample_position = start_sample + index
            existing = timeline[sample_position]
            # Неизбежное наложение: приглушаем того, кого перебивают, а не рубим.
            if existing and clip[index]:
                existing = int(existing * OVERLAP_DUCK)
            mixed = existing + clip[index]
            timeline[sample_position] = max(-32768, min(32767, mixed))
        if placement.ducked:
            ducked_count += 1
        if chunk.ratio > MAX_SPEED_UP_RATIO + 0.01:
            stretched += 1
        transcript.append(
            {
                "start": round(placement.start, 3),
                "end": round(placement.end, 3),
                "source_start": round(chunk.start, 3),
                "onset_shift": round(placement.start - chunk.onset, 3),
                "source": chunk.source_text,
                "uzbek": spoken_text,
                "speaker": unit.speaker,
                "speaker_label": unit.speaker_label,
                "voice": voice,
                "style": unit.style,
                "part": position + 1,
                "parts": len(unit.chunks),
                "predicted": round(predict_speech_duration(spoken_text, voice), 3),
                "generated": round(chunk.generated, 3),
                "budget": round(chunk.budget, 3),
                "tempo": round(chunk.ratio, 3),
                "overlap": round(unit.overlap, 2),
                "ducked": placement.ducked,
            }
        )

    previous_end = max((item.end for item in placements.values()), default=0.0)
    transcript.sort(key=lambda item: item["start"])
    if ducked_count:
        print(
            f"[dubbing] неизбежных наложений (с приглушением): {ducked_count}",
            flush=True,
        )
    ratios = [chunk.ratio for _, _, chunk in tasks if chunk.ratio]
    shifts = sorted(abs(float(item["onset_shift"])) for item in transcript)
    if ratios:
        print(
            "[dubbing] темп: медиана "
            f"{_median(ratios):.3f}, за рабочим диапазоном {stretched} из {len(ratios)}",
            flush=True,
        )
    if shifts:
        # Диагностика синхрона: насколько озвучка сдвинута от начала слова.
        print(
            f"[dubbing] сдвиг старта: медиана {_median(shifts) * 1000:.0f} мс, "
            f"P90 {shifts[int(len(shifts) * 0.9)] * 1000:.0f} мс",
            flush=True,
        )
    write_qa_report(ordered, transcript, work_dir / "qa.txt")

    output = work_dir / "dubbed.wav"
    # Последняя реплика не обрезается по концу видео: слово должно договориться.
    # Раньше фраза на 44-й секунде 45-секундного ролика рубилась на полуслове.
    keep_seconds = min(duration + TAIL_KEEP_SECONDS, max(duration, previous_end + 0.2))
    final_samples = timeline[: int(keep_seconds * sample_rate)]
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
    write_srt(transcript, work_dir / "uzbek.srt")
    if skipped_count:
        (work_dir / "skipped_count.txt").write_text(str(skipped_count), encoding="utf-8")
    return output


def write_qa_report(
    units: list[DubSegment], transcript: list[dict[str, Any]], output: Path
) -> None:
    """Список мест, которые стоит проверить руками, вместо тихой «уверенности».

    Неуверенные и перекрывающиеся участки в дубляже всегда отправляют на
    отдельный QA: одного голоса на cross-talk из моно-микса корректно не
    восстановить, а сильно ускоренная реплика слышна.
    """
    lines: list[str] = []
    for unit in units:
        if unit.overlap > 0.20:
            lines.append(
                f"{unit.start:8.2f}s наложение голосов {unit.overlap * 100:.0f}% "
                f"({unit.speaker_label}): {unit.source_text[:60]}"
            )
    for item in transcript:
        if item.get("ducked"):
            lines.append(
                f"{float(item['start']):8.2f}s неизбежное наложение (приглушено) "
                f"{item.get('speaker_label', '')}: {str(item['uzbek'])[:55]}"
            )
        if float(item["tempo"]) > MAX_SPEED_UP_RATIO + 0.01:
            lines.append(
                f"{float(item['start']):8.2f}s ускорение {float(item['tempo']):.2f}x "
                f"— перепишите текст короче: {str(item['uzbek'])[:60]}"
            )
        elif abs(float(item["onset_shift"])) > 0.25:
            lines.append(
                f"{float(item['start']):8.2f}s старт сдвинут на "
                f"{float(item['onset_shift']) * 1000:.0f} мс: {str(item['uzbek'])[:60]}"
            )
    if not lines:
        lines.append("Замечаний нет: наложений, переускорений и сдвигов старта не найдено.")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _srt_time(value: float) -> str:
    value = max(0.0, value)
    hours, rest = divmod(int(value), 3600)
    minutes, seconds = divmod(rest, 60)
    millis = int(round((value - int(value)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def write_srt(transcript: list[dict[str, Any]], output: Path) -> None:
    """Сохраняет узбекские субтитры в формате SRT."""
    blocks: list[str] = []
    for number, item in enumerate(transcript, start=1):
        text = str(item.get("uzbek", "")).strip()
        if not text:
            continue
        start = _srt_time(float(item["start"]))
        end = _srt_time(float(item["end"]))
        blocks.append(f"{number}\n{start} --> {end}\n{text}\n")
    output.write_text("\n".join(blocks), encoding="utf-8")


def mux_video(
    video: Path,
    dubbed: Path,
    output: Path,
    mode: str,
    original_volume: float = 0.25,
    dub_volume: float = 1.0,
) -> None:
    should_mix = mode == "mix" and has_audio(video)
    original_volume = min(max(original_volume, 0.0), 1.5)
    dub_volume = min(max(dub_volume, 0.2), 2.0)

    # Обработка голоса как на студии: мягкий компрессор выравнивает динамику,
    # loudnorm приводит к вещательному уровню громкости.
    voice_chain = (
        "acompressor=threshold=-20dB:ratio=3:attack=8:release=180:makeup=2,"
        f"loudnorm=I=-16:TP=-1.5:LRA=11,volume={dub_volume:.2f}"
    )

    def command(video_codec: list[str]) -> list[str]:
        base = ["ffmpeg", "-y", "-i", str(video), "-i", str(dubbed)]
        if should_mix:
            # sidechaincompress = ducking: оригинал автоматически притухает ровно
            # там, где звучит узбекская речь, поэтому дубляж всегда разборчив.
            audio = [
                "-filter_complex",
                f"[1:a:0]{voice_chain},asplit=2[dub][key];"
                f"[0:a:0]volume={original_volume:.2f}[orig];"
                "[orig][key]sidechaincompress="
                "threshold=0.005:ratio=20:attack=5:release=300[duck];"
                "[duck][dub]amix=inputs=2:duration=longest:normalize=0[aout]",
                "-map", "0:v:0", "-map", "[aout]",
            ]
        else:
            audio = [
                "-filter_complex", f"[1:a:0]{voice_chain}[aout]",
                "-map", "0:v:0", "-map", "[aout]",
            ]
        # Без -shortest: если последняя реплика чуть выходит за конец видео,
        # она доигрывает целиком, а не обрывается на полуслове.
        return base + audio + video_codec + [
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
        ]

    try:
        run_command(command(["-c:v", "copy"]))
    except RuntimeError:
        output.unlink(missing_ok=True)
        run_command(command(["-c:v", "libx264", "-preset", "medium", "-crf", "20"]))


# -----------------------------------------------------------------------------
# Полностью автоматический процесс дубляжа
# -----------------------------------------------------------------------------

STYLE_PRESETS = {
    "auto": "",
    "cinema": (
        "Cinematic drama dubbing: rich emotional range, expressive delivery, "
        "strong character presence, dramatic pacing."
    ),
    "vlog": (
        "Casual vlog dubbing: friendly, upbeat, spontaneous and conversational, "
        "like talking to a friend on camera."
    ),
    "news": (
        "News/documentary dubbing: clear, confident, authoritative and measured, "
        "neutral but engaged tone."
    ),
    "comedy": (
        "Comedy dubbing: playful, lively, exaggerated timing, teasing energy "
        "and expressive punchlines."
    ),
}


def auto_dubbing_pipeline(
    input_path: Path,
    output_path: Path,
    language: str | None,
    voice_map: dict[str, str],
    audio_mode: str,
    style_preset: str,
    original_volume: float,
    dub_volume: float,
    speech_speed: float,
    progress: ProgressCallback,
) -> int:
    """Возвращает количество реплик, которые не удалось озвучить."""
    work_dir = input_path.parent
    progress(5, "Проверяется видео")
    duration = media_duration(input_path)
    if duration > MAX_VIDEO_MINUTES * 60:
        raise RuntimeError(f"Видео длиннее лимита {MAX_VIDEO_MINUTES} минут")
    if not has_audio(input_path):
        raise RuntimeError("В видео нет аудиодорожки")

    source_audio = work_dir / "source.wav"
    progress(12, "Извлекается аудио")
    extract_audio(input_path, source_audio)

    # Уровень 1: слова. Сначала ASR, затем forced alignment — из выровненных слов
    # берутся и лексическое начало, и карта внутренних паузаций.
    progress(20, "Распознаётся речь")
    words, detected_language = transcribe(source_audio, language)
    progress(26, "Выравниваются слова по аудио")
    words = forced_align(source_audio, words, language or detected_language)
    utterances = utterances_for_diarization(words)

    client = GeminiClient()
    try:
        # Уровень 2: кто говорит. По ВСЕМУ аудио, без искусственных лимитов длины.
        progress(30, "Определяются говорящие")
        diarization = diarize(client, source_audio, duration, utterances)
        assign_word_speakers(words, diarization)

        # Регистр голоса (низкий/высокий) — один раз на говорящего, по агрегату F0.
        pitch_audio = work_dir / "pitch.wav"
        try:
            make_pitch_audio(input_path, pitch_audio)
        except RuntimeError:
            pitch_audio = source_audio
        stats = speaker_voice_stats(pitch_audio, diarization)
        registers = voice_registers(diarization, stats)
        # Диагностика: stats[label].voiced — это материал ПОСЛЕ строгого фильтра
        # эталонных участков (>=2.5с, наложение <=10%), а не всё время говорящего.
        # Печатаем общее время отдельно, чтобы отличить «pyannote дал мало» от
        # «фильтр эталонов отсеял почти всё» — это разные проблемы и разный фикс.
        total_by_speaker: dict[str, float] = {}
        turns_by_speaker: dict[str, int] = {}
        for turn in diarization.turns:
            total_by_speaker[turn.speaker] = total_by_speaker.get(turn.speaker, 0.0) + turn.duration
            turns_by_speaker[turn.speaker] = turns_by_speaker.get(turn.speaker, 0) + 1
        print(
            "[dubbing] всего речи на говорящего (до фильтра эталонов): "
            + ", ".join(
                f"{label}={total_by_speaker.get(label, 0.0):.1f}с "
                f"({turns_by_speaker.get(label, 0)} turn'ов)"
                for label in diarization.speakers
            ),
            flush=True,
        )
        character_voices = assign_character_voices(
            diarization, stats, registers, voice_map, load_voice_fingerprints()
        )
        print(
            f"[dubbing] диаризация ({diarization.backend}): "
            + ", ".join(
                f"{label}={registers.get(label, '?')} "
                f"({stats[label].f0 if label in stats else 0:.0f} Гц, "
                f"{stats[label].voiced if label in stats else 0:.1f}с"
                + (", БИМОДАЛЬНО" if label in stats and stats[label].bimodal else "")
                + f")->{character_voices.get(label, '?')}"
                for label in diarization.speakers
            )
            + f"; наложений: {len(diarization.overlaps)}",
            flush=True,
        )
        bimodal = [label for label in diarization.speakers if label in stats and stats[label].bimodal]
        if bimodal:
            print(
                "[dubbing] ВНИМАНИЕ: возможно, в метках "
                f"{', '.join(bimodal)} склеены два голоса — проверьте qa.txt",
                flush=True,
            )

        # Уровень 3 и 4: смысловые блоки внутри одного говорящего и дыхательные группы.
        units = build_units(words, diarization)
        for unit in units:
            unit.speaker = registers.get(unit.speaker_label, "male")
            unit.voice = character_voices.get(unit.speaker_label, "")
        progress(34, "Уточняется начало речи по словам")
        refined = refine_onsets(
            source_audio, units, diarization, aligned=_alignment_state["applied"]
        )
        assign_chunk_budgets(units, duration)
        print(
            f"[dubbing] реплик {len(units)}, групп "
            f"{sum(len(unit.chunks) for unit in units)}, онсетов уточнено {refined}",
            flush=True,
        )

        progress(38, "Анализируется сцена и характеры")
        scene = client.analyze_scene(
            [
                {"index": unit.index, "start": unit.start, "end": unit.end,
                 "text": unit.source_text}
                for unit in units
            ]
        )
        preset = STYLE_PRESETS.get(style_preset, "")
        if preset:
            scene = f"{preset}\n{scene}".strip()

        progress(40, f"Переводятся {len(units)} реплик на узбекский")
        client.translate(units, voice_map, scene)
        progress(43, "Вычитывается узбекский текст")
        client.polish(units)
        # Подгонка длины ТЕКСТОМ — до озвучки и до любого изменения темпа.
        condensed = client.condense_to_budget(units, voice_map)
        if condensed:
            print(f"[dubbing] переписано под слоговой бюджет: {condensed} реплик", flush=True)

        progress(45, "Создаётся узбекская озвучка")
        dubbed = render_timeline(
            duration, units, client, voice_map, scene, work_dir, progress, speech_speed
        )
    finally:
        client.close()

    progress(90, "Собирается готовое видео")
    mux_video(input_path, dubbed, output_path, audio_mode, original_volume, dub_volume)
    progress(98, "Проверяется результат")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("FFmpeg не создал итоговое видео")
    media_duration(output_path)

    skipped_file = work_dir / "skipped_count.txt"
    if skipped_file.is_file():
        try:
            return int(skipped_file.read_text(encoding="utf-8").strip())
        except ValueError:
            return 0
    return 0


def process_job(
    job_id: str,
    input_path: Path,
    language: str | None,
    voice_map: dict[str, str],
    audio_mode: str,
    style_preset: str,
    original_volume: float,
    dub_volume: float,
    speech_speed: float,
) -> None:
    output_path = input_path.parent / "uzbek-dubbed.mp4"

    def progress(percent: int, message: str) -> None:
        update_job(job_id, status="processing", progress=percent, message=message)

    try:
        skipped = auto_dubbing_pipeline(
            input_path,
            output_path,
            language,
            voice_map,
            audio_mode,
            style_preset,
            original_volume,
            dub_volume,
            speech_speed,
            progress,
        )
        message = "Дубляж готов"
        if skipped:
            message = (
                f"Дубляж готов, но {skipped} реплик(и) остались на языке оригинала — "
                "таймкоды в skipped.txt"
            )
        update_job(
            job_id,
            status="completed",
            progress=100,
            message=message,
            skipped=skipped,
            result_path=str(output_path),
        )
    except Exception as exc:
        update_job(job_id, status="failed", message="Не удалось обработать видео", error=str(exc))


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
        "skipped": job.get("skipped", 0),
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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    return (
        HTML_PAGE.replace("__MAX_MB__", str(MAX_UPLOAD_MB))
        .replace("__MAX_MINUTES__", str(MAX_VIDEO_MINUTES))
        .replace("__VOICES__", json.dumps(VOICES))
        .replace("__FEMALE__", DEFAULT_FEMALE_VOICE)
        .replace("__MALE__", DEFAULT_MALE_VOICE)
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/voices")
def list_voices() -> dict[str, Any]:
    fingerprints = load_voice_fingerprints()
    return {
        "voices": VOICES,
        "female": DEFAULT_FEMALE_VOICE,
        "male": DEFAULT_MALE_VOICE,
        "calibrated": sorted(fingerprints),
    }


@app.post("/api/calibrate-voices")
def calibrate_voices() -> dict[str, Any]:
    """Замеряет F0 всех голосов и кэширует — включает подбор голоса по тембру."""
    if not os.getenv("GEMINI_API_KEY", "").strip():
        raise HTTPException(status_code=503, detail="На сервере не задан GEMINI_API_KEY")
    client = GeminiClient()
    try:
        fingerprints = calibrate_voice_fingerprints(client, VOICES)
    finally:
        client.close()
    return {"calibrated": len(fingerprints), "fingerprints": fingerprints}


@app.post("/api/jobs", status_code=202)
async def create_job(
    video: UploadFile = File(...),
    source_language: str = Form("auto"),
    female_voice: str = Form(DEFAULT_FEMALE_VOICE),
    male_voice: str = Form(DEFAULT_MALE_VOICE),
    audio_mode: str = Form("mix"),
    style_preset: str = Form("auto"),
    original_volume: float = Form(0.25),
    dub_volume: float = Form(1.0),
    speech_speed: float = Form(0.95),
) -> dict[str, Any]:
    if not os.getenv("GEMINI_API_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="Не задан GEMINI_API_KEY. Задайте export GEMINI_API_KEY=... "
            "или положите .env рядом с app.py и перезапустите сервер.",
        )
    if source_language not in {"auto", "ru", "en", "uz"}:
        raise HTTPException(status_code=400, detail="Неподдерживаемый исходный язык")
    if female_voice not in VOICES or male_voice not in VOICES:
        raise HTTPException(status_code=400, detail="Неизвестный голос")
    if audio_mode not in {"mix", "replace"}:
        raise HTTPException(status_code=400, detail="Неизвестный режим звука")
    if style_preset not in STYLE_PRESETS:
        raise HTTPException(status_code=400, detail="Неизвестный стиль озвучки")

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
        {"female": female_voice, "male": male_voice},
        audio_mode,
        style_preset,
        original_volume,
        dub_volume,
        min(max(speech_speed, 0.75), 1.25),
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


@app.get("/api/jobs/{job_id}/subtitles")
def subtitles(job_id: str) -> FileResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Задача не найдена")
        result_path = job.get("result_path")
        if job["status"] != "completed" or not result_path:
            raise HTTPException(status_code=409, detail="Субтитры ещё не готовы")
    path = Path(result_path).parent / "uzbek.srt"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Субтитры не найдены")
    return FileResponse(path, media_type="text/plain; charset=utf-8", filename="uzbek.srt")


@atexit.register
def shutdown() -> None:
    _executor.shutdown(wait=False, cancel_futures=True)


# -----------------------------------------------------------------------------
# Весь frontend: HTML + CSS + JavaScript (одна кнопка, всё автоматически)
# -----------------------------------------------------------------------------

HTML_PAGE = r'''<!doctype html>
<html lang="ru">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ovoz — авто-дубляж на узбекский</title>
<style>
:root{--bg:#12121a;--raised:#1a1a24;--panel:#1e1e2a;--line:#30303f;--text:#ede9e3;--dim:#9292a1;--gold:#d4a574;--red:#c85049}
*{box-sizing:border-box}html,body{margin:0;min-height:100%}
body{background:radial-gradient(ellipse 900px 520px at 15% -10%,rgba(212,165,116,.1),transparent 60%),var(--bg);color:var(--text);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;padding:46px 20px 80px}
.wrap{width:min(100%,820px);margin:auto}
header{margin-bottom:36px}
.eyebrow{color:var(--gold);font-size:11px;letter-spacing:.18em;text-transform:uppercase;display:flex;align-items:center;gap:10px}
.eyebrow:before{content:"";width:7px;height:7px;border-radius:50%;background:var(--red);box-shadow:0 0 8px var(--red)}
h1{font-family:Georgia,serif;font-size:clamp(36px,7vw,58px);line-height:1;letter-spacing:-.025em;margin:13px 0;font-weight:600}
h1 em{color:var(--gold);font-weight:500}
.sub{color:var(--dim);max-width:640px;font-size:14px;line-height:1.7}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:24px;margin-bottom:18px}
.title{margin:0 0 17px;color:var(--dim);font-size:11px;letter-spacing:.14em;text-transform:uppercase;display:flex;justify-content:space-between;gap:16px}
.drop{min-height:170px;border:1px dashed #48485a;background:var(--raised);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;cursor:pointer;text-align:center;padding:24px}
.drop:hover,.drop.drag{border-color:var(--gold);background:rgba(212,165,116,.06)}
.drop input{display:none}.plus{color:var(--gold);font-size:28px}.drop strong{font-size:14px}
.drop small,.option small,.hint{display:block;color:var(--dim);font-size:11px;line-height:1.5}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
label,legend{color:var(--dim);font-size:10px;letter-spacing:.08em;text-transform:uppercase}
select{width:100%;margin-top:8px;background:var(--raised);border:1px solid var(--line);color:var(--text);font:13px ui-monospace,monospace;padding:12px;border-radius:3px;outline:none}
select:focus{border-color:var(--gold)}
fieldset{border:0;padding:0;margin:22px 0 0}legend{margin-bottom:9px}
.option{display:flex;align-items:center;gap:10px;background:var(--raised);border:1px solid var(--line);padding:12px;margin-top:8px;cursor:pointer}
.option:has(input:checked){border-color:var(--gold)}.option input{accent-color:var(--gold);width:auto;margin:0}
.option span{display:flex;align-items:baseline;justify-content:space-between;width:100%;gap:16px}.option strong{color:var(--text);font-size:12px}
.primary,.download{display:block;width:100%;border:0;border-radius:3px;background:var(--gold);color:#12121a;padding:16px;font:700 12px ui-monospace,monospace;letter-spacing:.1em;text-transform:uppercase;text-align:center;cursor:pointer;text-decoration:none}
.primary:disabled{background:var(--line);color:var(--dim);cursor:wait}
.adv{margin-top:14px;color:var(--dim);font-size:11px;cursor:pointer;text-decoration:underline}
input[type=range]{width:100%;margin-top:10px;accent-color:var(--gold)}
.ghost-dl{background:transparent;border:1px solid var(--line);color:var(--dim);margin-top:10px}
.error{color:#ee736c;font-size:12px;line-height:1.6;display:none;margin-top:12px}.error.show{display:block}
.progress,.result{margin-top:18px}
.head{display:flex;align-items:center;justify-content:space-between;font-size:12px}
.dot{display:inline-block;width:8px;height:8px;background:var(--red);border-radius:50%;margin-right:10px;box-shadow:0 0 8px var(--red);animation:pulse 1.2s infinite}
.track{height:5px;background:var(--raised);margin:17px 0 10px;overflow:hidden}.bar{width:0;height:100%;background:var(--gold);transition:width .35s}
video{display:block;width:100%;max-height:460px;background:#08080c;margin-bottom:16px}
footer{color:var(--dim);text-align:center;font-size:10px;margin-top:30px}
@keyframes pulse{50%{opacity:.35}}
@media(max-width:620px){body{padding:30px 14px 60px}.panel{padding:18px}.grid{grid-template-columns:1fr}.option span{display:block}.limit{display:none}}
</style>
</head><body><main class="wrap">
<header><p class="eyebrow">Gemini 3.1 TTS · Uzbek dubbing</p><h1>Видео говорит <em>по-узбекски</em>.</h1><p class="sub">Загрузите видео — всё остальное автоматически: распознавание речи, перевод, определение говорящего (мужской/женский голос) и эмоции каждой реплики. Ничего править не нужно.</p></header>

<form id="form">
<section class="panel"><p class="title"><span>01 · Видео</span><span class="limit">до __MAX_MB__ МБ · __MAX_MINUTES__ минут</span></p>
<label class="drop" id="drop" for="video"><input id="video" name="video" type="file" accept="video/mp4,video/quicktime,video/webm,.mkv" required><span class="plus">＋</span><strong id="fileName">Выберите видео</strong><small id="fileMeta">MP4, MOV, WEBM или MKV</small></label>
<div style="margin-top:16px"><label for="lang">Исходный язык</label>
<select id="lang" name="source_language"><option value="auto">Определить автоматически</option><option value="ru">Русский</option><option value="en">Английский</option><option value="uz">Узбекский</option></select></div>

<p class="adv" id="advToggle">Дополнительно: выбрать голоса и звук ▾</p>
<div id="advBox" hidden>
  <div class="grid" style="margin-top:6px">
    <div><label for="femaleVoice">Женский голос</label><select id="femaleVoice" name="female_voice"></select></div>
    <div><label for="maleVoice">Мужской голос</label><select id="maleVoice" name="male_voice"></select></div>
  </div>
  <div style="margin-top:16px"><label for="stylePreset">Стиль озвучки</label>
  <select id="stylePreset" name="style_preset">
    <option value="auto">Авто (по сцене)</option>
    <option value="cinema">Кино / драма</option>
    <option value="vlog">Влог / разговорный</option>
    <option value="news">Новости / документальный</option>
    <option value="comedy">Комедия</option>
  </select></div>
  <fieldset><legend>Оригинальный звук</legend>
    <label class="option"><input type="radio" name="audio_mode" value="mix" checked><span><strong>Тихий фон</strong><small>Оригинал приглушается под речью</small></span></label>
    <label class="option"><input type="radio" name="audio_mode" value="replace"><span><strong>Полная замена</strong><small>Только узбекская речь</small></span></label>
  </fieldset>
  <div class="grid" style="margin-top:16px">
    <div><label for="origVol">Громкость оригинала: <span id="origVolVal">25%</span></label>
    <input type="range" id="origVol" name="original_volume" min="0" max="1.2" step="0.05" value="0.25"></div>
    <div><label for="dubVol">Громкость дубляжа: <span id="dubVolVal">100%</span></label>
    <input type="range" id="dubVol" name="dub_volume" min="0.4" max="1.6" step="0.05" value="1"></div>
  </div>
  <div style="margin-top:16px"><label for="speed">Скорость речи: <span id="speedVal">0.95x</span> <small style="display:inline;text-transform:none;letter-spacing:0">меньше — медленнее и спокойнее</small></label>
  <input type="range" id="speed" name="speech_speed" min="0.75" max="1.15" step="0.05" value="0.95"></div>
</div>
</section>
<button class="primary" id="submit" type="submit">Создать узбекский дубляж</button><p class="error" id="error" role="alert"></p></form>

<section class="panel progress" id="progress" hidden><div class="head"><div><span class="dot"></span><span id="status">Подготовка…</span></div><strong id="percent">0%</strong></div><div class="track"><div class="bar" id="bar"></div></div><p class="hint">Не закрывайте страницу до окончания обработки.</p></section>

<section class="panel result" id="result" hidden><p class="title">Готовый дубляж</p><video id="player" controls playsinline></video><a class="download" id="download" download="uzbek-dubbed.mp4">Скачать видео</a><a class="download ghost-dl" id="srtLink" download="uzbek.srt">Скачать субтитры (.srt)</a></section>
<footer>Gemini API key хранится только на сервере · файлы удаляются через 24 часа</footer></main>

<script>
const VOICES=__VOICES__,DEF_FEMALE="__FEMALE__",DEF_MALE="__MALE__";
const $=s=>document.querySelector(s);
const form=$('#form'),video=$('#video'),drop=$('#drop'),fileName=$('#fileName'),fileMeta=$('#fileMeta'),submit=$('#submit'),error=$('#error');
const progress=$('#progress'),bar=$('#bar'),percent=$('#percent'),statusText=$('#status');
const advToggle=$('#advToggle'),advBox=$('#advBox'),femaleVoice=$('#femaleVoice'),maleVoice=$('#maleVoice');
const result=$('#result'),player=$('#player'),download=$('#download'),srtLink=$('#srtLink');
const origVol=$('#origVol'),dubVol=$('#dubVol'),origVolVal=$('#origVolVal'),dubVolVal=$('#dubVolVal');
const speed=$('#speed'),speedVal=$('#speedVal');
let timer=null;

const pct=v=>Math.round(Number(v)*100)+'%';
origVol.oninput=()=>origVolVal.textContent=pct(origVol.value);
dubVol.oninput=()=>dubVolVal.textContent=pct(dubVol.value);
speed.oninput=()=>speedVal.textContent=Number(speed.value).toFixed(2)+'x';

VOICES.forEach(v=>{femaleVoice.add(new Option(v,v));maleVoice.add(new Option(v,v));});
femaleVoice.value=DEF_FEMALE;maleVoice.value=DEF_MALE;
advToggle.onclick=()=>{advBox.hidden=!advBox.hidden;advToggle.textContent=(advBox.hidden?'Дополнительно: выбрать голоса и звук ▾':'Скрыть дополнительно ▴')};

const size=n=>n<1048576?`${(n/1024).toFixed(0)} КБ`:`${(n/1048576).toFixed(1)} МБ`;
function showFile(f){if(f){fileName.textContent=f.name;fileMeta.textContent=size(f.size)}}
video.onchange=()=>showFile(video.files[0]);
['dragenter','dragover'].forEach(n=>drop.addEventListener(n,e=>{e.preventDefault();drop.classList.add('drag')}));
['dragleave','drop'].forEach(n=>drop.addEventListener(n,e=>{e.preventDefault();drop.classList.remove('drag')}));
drop.addEventListener('drop',e=>{if(e.dataTransfer.files.length){video.files=e.dataTransfer.files;showFile(video.files[0])}});

function fail(m){error.textContent=m;error.classList.add('show')}
function clearErr(){error.textContent='';error.classList.remove('show')}
function update(j){progress.hidden=false;const p=Math.max(0,Math.min(100,Number(j.progress)||0));bar.style.width=p+'%';percent.textContent=p+'%';statusText.textContent=j.message||'Обработка…'}
async function parse(r){const d=await r.json().catch(()=>({}));if(!r.ok)throw new Error(d.detail||`Ошибка HTTP ${r.status}`);return d}

async function poll(id){
  try{
    const j=await parse(await fetch(`/api/jobs/${id}`));update(j);
    if(j.status==='completed'){
      submit.disabled=false;submit.textContent='Создать ещё один дубляж';progress.hidden=true;
      const url=j.result_url+'?t='+Date.now();player.src=url;download.href=url;
      srtLink.href=`/api/jobs/${id}/subtitles?t=`+Date.now();
      if(j.skipped>0)fail(`Внимание: ${j.skipped} реплик(и) не удалось озвучить — в этих местах будет тишина. Попробуйте запустить снова или снизить TTS_CONCURRENCY.`);
      result.hidden=false;result.scrollIntoView({behavior:'smooth'});return;
    }
    if(j.status==='failed'){submit.disabled=false;submit.textContent='Попробовать снова';progress.hidden=true;fail(j.error||'Не удалось создать дубляж');return}
    timer=setTimeout(()=>poll(id),1500);
  }catch(e){submit.disabled=false;fail(e.message)}
}

form.onsubmit=async e=>{
  e.preventDefault();clearErr();result.hidden=true;player.removeAttribute('src');player.load();
  if(!video.files.length){fail('Выберите видео');return}
  if(timer)clearTimeout(timer);
  submit.disabled=true;submit.textContent='Загрузка видео…';
  update({progress:1,message:'Видео загружается на сервер'});
  try{
    const j=await parse(await fetch('/api/jobs',{method:'POST',body:new FormData(form)}));
    submit.textContent='Создаётся дубляж…';update(j);poll(j.id);
  }catch(e){submit.disabled=false;submit.textContent='Создать узбекский дубляж';fail(e.message)}
};
</script></body></html>'''


if __name__ == "__main__":
    cleanup_old_jobs()
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
