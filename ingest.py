#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_LOCAL_ONLY = os.getenv("EMBEDDING_LOCAL_ONLY", "1") == "1"


def load_documents(kb_dir: Path):
    docs = []
    if not kb_dir.exists():
        kb_dir.mkdir(parents=True, exist_ok=True)
        return docs

    txt_loader = DirectoryLoader(
        str(kb_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    md_loader = DirectoryLoader(
        str(kb_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    pdf_loader = DirectoryLoader(
        str(kb_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )

    for loader in (txt_loader, md_loader, pdf_loader):
        try:
            docs.extend(loader.load())
        except Exception as exc:
            print(f"[WARN] Fehler beim Laden mit {loader.__class__.__name__}: {exc}")

    return docs


def build_index(kb_dir: Path, vector_db_dir: Path):
    print(f"[INFO] Lade Dokumente aus: {kb_dir}")
    docs = load_documents(kb_dir)
    if not docs:
        print("[ERROR] Keine unterstützten Dateien gefunden (.txt, .md, .pdf).")
        print(f"[HINWEIS] Lege Dateien in {kb_dir} ab und starte erneut.")
        return 1

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Dokument-Chunks erstellt: {len(chunks)}")

    model_path = Path(EMBEDDING_MODEL)
    model_name = str(model_path) if model_path.exists() else EMBEDDING_MODEL
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"local_files_only": EMBEDDING_LOCAL_ONLY},
        )
    except Exception as exc:
        print(f"[ERROR] Embedding-Modell konnte nicht geladen werden: {exc}")
        print(
            "[HINWEIS] Lege ein lokales Embedding-Modell fest mit "
            "EMBEDDING_MODEL=/absoluter/pfad/zum/modell "
            "oder setze EMBEDDING_LOCAL_ONLY=0 fuer den einmaligen Download."
        )
        return 1
    vector_db_dir.mkdir(parents=True, exist_ok=True)

    # Eine neue persistente DB aus den aktuellen Chunks erstellen.
    _ = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(vector_db_dir),
    )
    print(f"[OK] Vektordatenbank erstellt unter: {vector_db_dir}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Erstellt/aktualisiert die lokale Chroma Vektordatenbank.")
    parser.add_argument("--kb-dir", default=str(KB_DIR), help="Pfad zum Knowledge-Base-Ordner")
    parser.add_argument("--db-dir", default=str(VECTOR_DB_DIR), help="Pfad für persistente Chroma DB")
    return parser.parse_args()


def main():
    args = parse_args()
    raise SystemExit(build_index(Path(args.kb_dir), Path(args.db_dir)))


if __name__ == "__main__":
    main()
