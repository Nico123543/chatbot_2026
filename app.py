#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path
from typing import Dict, Generator, List

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(BASE_DIR / ".hf_cache"))

import gradio as gr
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

KB_DIR = BASE_DIR / "knowledge_base"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
STATIC_DIR = BASE_DIR / "static"
HF_CACHE_DIR = BASE_DIR / ".hf_cache"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_LOCAL_ONLY = os.getenv("EMBEDDING_LOCAL_ONLY", "1") == "1"

AVATAR_IDLE = STATIC_DIR / "avatar_idle.gif"
AVATAR_TALK = STATIC_DIR / "avatar_talk.gif"

SYSTEM_PROMPT = """Du bist ein hilfreicher Assistent fuer [Firmenname].
Nutze NUR den folgenden Kontext, um die Frage zu beantworten.
Wenn die Antwort im Kontext nicht enthalten ist, sage klar, dass die Information nicht vorliegt.

Kontext:
{context}

Frage:
{question}
"""


def ensure_dirs():
    KB_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)


def create_embeddings():
    model_path = Path(EMBEDDING_MODEL)
    model_name = str(model_path) if model_path.exists() else EMBEDDING_MODEL
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"local_files_only": EMBEDDING_LOCAL_ONLY},
    )


def avatar_html(state: str) -> str:
    idle_exists = AVATAR_IDLE.exists()
    talk_exists = AVATAR_TALK.exists()

    if idle_exists and talk_exists:
        src = AVATAR_TALK if state == "talking" else AVATAR_IDLE
        return f"""
        <div style="display:flex;justify-content:center;align-items:center;height:260px;">
          <img id="avatar" src="/gradio_api/file={src}" alt="Avatar" style="max-height:240px;border-radius:14px;"/>
        </div>
        """

    label = "Talking" if state == "talking" else "Idle"
    bg = "#ffe7c2" if state == "talking" else "#dceeff"
    return f"""
    <div style="display:flex;justify-content:center;align-items:center;height:260px;background:{bg};border-radius:14px;">
      <div style="font-family:sans-serif;text-align:center;">
        <div style="font-size:22px;font-weight:700;">Avatar: {label}</div>
        <div style="font-size:13px;opacity:0.75;margin-top:6px;">
          Lege static/avatar_idle.gif und static/avatar_talk.gif ab.
        </div>
      </div>
    </div>
    """


def load_components(model_name: str):
    embeddings = create_embeddings()
    vector_store = Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
    )
    llm = ChatOllama(model=model_name, temperature=0.2)
    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
    return llm, vector_store, prompt


def build_answer(
    question: str,
    chat_history: List[Dict[str, str]],
    llm: ChatOllama,
    vector_store: Chroma,
    prompt: PromptTemplate,
) -> Generator[tuple[List[Dict[str, str]], str, str], None, None]:
    try:
        docs = vector_store.similarity_search(question, k=4)
    except Exception as exc:
        msg = f"Fehler bei der Vektorsuche: {exc}"
        chat_history = chat_history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": msg},
        ]
        yield chat_history, "", avatar_html("idle")
        return

    context = "\n\n".join(d.page_content for d in docs) if docs else "(kein Kontext gefunden)"
    final_prompt = prompt.format(context=context, question=question)

    working_history = chat_history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""},
    ]
    yield working_history, "", avatar_html("talking")

    try:
        response = llm.invoke(final_prompt).content
    except Exception as exc:
        response = (
            "LLM-Aufruf fehlgeschlagen. Bitte pruefe, ob Ollama laeuft "
            f"und das Modell installiert ist.\n\nTechnischer Fehler: {exc}"
        )

    partial = ""
    for ch in str(response):
        partial += ch
        working_history[-1]["content"] = partial
        yield working_history, "", avatar_html("talking")

    working_history[-1]["content"] = str(response)
    yield working_history, "", avatar_html("idle")


def check_startup(vector_db_dir: Path):
    index_files = list(vector_db_dir.glob("**/*"))
    if not index_files:
        raise RuntimeError(
            "Keine Vektordatenbank gefunden. Bitte zuerst `python ingest.py` ausfuehren."
        )


def create_ui(llm: ChatOllama, vector_store: Chroma, prompt: PromptTemplate):
    def stream_reply(user_msg, history):
        yield from build_answer(
            user_msg,
            history,
            llm=llm,
            vector_store=vector_store,
            prompt=prompt,
        )

    with gr.Blocks(title="Lokaler Unternehmens-Chatbot") as demo:
        gr.Markdown("## Lokaler Unternehmens-Chatbot (Offline mit Ollama + RAG)")
        with gr.Row():
            avatar = gr.HTML(value=avatar_html("idle"), label="Avatar")
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": "Hallo! Wie kann ich helfen?"}],
                height=420,
                label="Chat",
            )
        msg = gr.Textbox(
            label="Frage",
            placeholder="Stelle eine Frage zur Wissensdatenbank...",
            lines=2,
        )
        send = gr.Button("Senden", variant="primary")

        send.click(
            fn=stream_reply,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, avatar],
            api_name="chat",
            queue=True,
            concurrency_limit=1,
            show_progress="hidden",
        )
        msg.submit(
            fn=stream_reply,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, avatar],
            queue=True,
            concurrency_limit=1,
            show_progress="hidden",
        )
    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Lokaler Unternehmens-Chatbot mit Avatar und RAG")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "phi3"), help="Ollama Modellname")
    parser.add_argument("--host", default="127.0.0.1", help="Host fuer Gradio")
    parser.add_argument("--port", type=int, default=7860, help="Port fuer Gradio")
    return parser.parse_args()


def find_open_port(host: str, preferred_port: int, max_tries: int = 20) -> int:
    for port in range(preferred_port, preferred_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Kein freier Port in Bereich {preferred_port}-{preferred_port + max_tries - 1} gefunden.")


def main():
    args = parse_args()
    os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    ensure_dirs()
    check_startup(VECTOR_DB_DIR)
    try:
        llm, vector_store, prompt = load_components(args.model)
    except Exception as exc:
        raise SystemExit(
            "Start fehlgeschlagen. Embedding-Modell lokal nicht verfuegbar.\n"
            "Setze EMBEDDING_MODEL auf einen lokalen Modellpfad oder "
            "setze fuer einmaligen Download EMBEDDING_LOCAL_ONLY=0.\n"
            f"Technischer Fehler: {exc}"
        )
    ui = create_ui(llm, vector_store, prompt)
    ui.queue()
    server_port = find_open_port(args.host, args.port)
    print(f"[INFO] Starte UI auf http://{args.host}:{server_port}")
    ui.launch(server_name=args.host, server_port=server_port, show_error=True, share=False)


if __name__ == "__main__":
    main()
