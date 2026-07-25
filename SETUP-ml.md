# Точность дубляжа: установка ML-стека (Python 3.12)

Три главные претензии супервайзера — «голос за секунду до артикуляции», «мужчину
озвучили женским голосом» и «реплики наезжают» — упираются в один блокер:
**forced alignment и диаризация не установлены**, потому что среда собрана под
Python 3.14, где нет колёс `ctranslate2`/`faster-whisper`.

Ваш лог это подтвердил:

```
ERROR: Could not find a version that satisfies the requirement ctranslate2==4.4.0
ERROR: Ignored the following versions that require a different python version: ... <3.13 ...
```

## Решение: отдельный venv на Python 3.12

Не пытайтесь поставить всё на 3.14. DSP на чистом Python (наш код) работает на
любой версии, а тяжёлый ML держите в venv на 3.12.

```bash
# 1. Поставьте Python 3.12 (macOS): brew install python@3.12
# 2. Отдельное окружение под ML:
python3.12 -m venv .venv-ml
. .venv-ml/bin/activate
pip install -r requirements-ml.txt

# 3. Диаризация pyannote: примите условия модели и задайте токен
#    https://huggingface.co/pyannote/speaker-diarization-community-1
export HF_TOKEN=hf_xxx
# Если число говорящих известно (для вашего клипа их двое):
export NUM_SPEAKERS=2

# 4. Запуск основного приложения из этого же 3.12-окружения:
export GEMINI_API_KEY=...
python app.py
```

## Что включится автоматически

`app.py` подхватывает пакеты через soft-import — код менять не нужно:

- `whisperx` → forced alignment. Онсет реплики привязывается к первому реальному
  слову, а не к вздоху/смеху за секунду до речи. Лог: `forced alignment: уточнено слов N`.
- `pyannote.audio` → диаризация по всему аудио с разметкой наложений. Лог:
  `диаризация (pyannote): S1=male (128 Гц)...`. Без неё — резерв через Gemini,
  который и путал говорящих.

Переменные окружения:

| Переменная | Смысл |
|---|---|
| `FORCED_ALIGNMENT` | `auto` (по умолчанию) / `off` |
| `DIARIZATION_BACKEND` | `auto` / `pyannote` / `gemini` |
| `PYANNOTE_MODEL` | по умолчанию `pyannote/speaker-diarization-community-1` |
| `HF_TOKEN` | токен Hugging Face для pyannote |
| `NUM_SPEAKERS` | точное число говорящих (сильно снижает ошибки) |
| `MIN_SPEAKERS` / `MAX_SPEAKERS` | если точное число неизвестно |

## Подбор голоса по тембру (не обязательно, но улучшает расстановку)

Один раз измерьте F0 всех 30 голосов Gemini TTS — тогда каждому персонажу
подбирается ближайший по тембру голос, а не просто «мужской/женский»:

```bash
curl -X POST http://localhost:8000/api/calibrate-voices
```

Результат кэшируется в `data/voice_fingerprints.json`. После этого
`assign_character_voices` выбирает голоса из 30 доступных с жёстким фильтром по
полу и разносит двух мужчин/двух женщин в одной сцене по F0 (≥15 Гц).

## Контроль качества

Каждый дубляж пишет `qa.txt` рядом с результатом: наложения, переускорения,
сдвиги старта, приглушённые перебивания, подозрение на склейку двух голосов в
одну метку диаризации. Логи `[dubbing] сдвиг старта: медиана … P90 …` и
`[dubbing] темп: медиана …` показывают регресс без прослушивания.
