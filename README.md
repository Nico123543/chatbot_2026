# Lokaler Unternehmens-Chatbot (Offline, Ollama + RAG)

Ein vollständig lokaler Chatbot für Unternehmenswissen mit Avatar-UI.

## Features
- 100% lokal ausführbar auf dem Laptop (kein Cloud-Zwang)
- RAG mit `LangChain + ChromaDB` (persistente Vektordatenbank)
- LLM über `Ollama` (z. B. `phi3`, `llama3.2`)
- Chat-UI mit Avatar-Status (`Idle` / `Talking`) via Gradio
- Speech-to-Text (Mikrofon -> Textfeld, Whisper lokal)
- Text-to-Speech (Antwort -> Audioausgabe, macOS `say`)
- Ingest von `PDF`, `TXT`, `MD` aus lokaler Wissensbasis

## Projektstruktur

```text
.
├── app.py
├── ingest.py
├── requirements.txt
├── knowledge_base/
├── vector_db/
└── static/
```

## Voraussetzungen (ARM Mac)
- macOS auf Apple Silicon (arm64)
- Homebrew
- Python 3.11
- Ollama lokal installiert

### Einmalige Systeminstallation

```bash
brew install python@3.11 ollama
brew services start ollama
ollama pull phi3
```

## Installation

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Modell lokal vorbereiten

```bash
ollama list
```

## Wissensdatenbank vorbereiten
1. Dateien (`.pdf`, `.txt`, `.md`) in `knowledge_base/` ablegen.
2. Index erstellen:

```bash
source .venv/bin/activate
python ingest.py
```

## App starten

```bash
source .venv/bin/activate
python app.py --model phi3
```

Danach im Browser öffnen: `http://127.0.0.1:7860`
Wenn `7860` belegt ist, wählt die App automatisch den nächsten freien Port.

## Sprachfunktionen
- `Sprache -> Text`: nimmt Mikrofon-Eingabe auf und schreibt den Text ins Fragefeld.
- Bot-Antworten werden zusätzlich als Audio erzeugt und in `Sprachausgabe` abgespielt.

Optional:
- Stimme für TTS ändern: `TTS_VOICE=Anna` (oder eine lokal verfügbare macOS-Stimme)
- STT komplett offline erzwingen (nur wenn Modell lokal vorhanden): `STT_LOCAL_ONLY=1`

## Offline-Hinweise
- App bindet standardmäßig nur an `127.0.0.1`
- Ollama-Host ist lokal (`127.0.0.1:11434`)
- Embeddings laufen standardmäßig mit `local_files_only=True`

Falls das Embedding-Modell noch nicht lokal vorhanden ist:

```bash
source .venv/bin/activate
EMBEDDING_LOCAL_ONLY=0 python ingest.py
```

Danach wieder offline normal starten.

## Avatar-Dateien
Lege optional folgende Dateien in `static/`:
- `avatar_idle.gif`
- `avatar_talk.gif`

Ohne diese Dateien wird ein lokaler Fallback-Avatar angezeigt.

## Lizenz
MIT (siehe `LICENSE`).
