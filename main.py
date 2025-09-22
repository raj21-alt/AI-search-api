# main.py
import os
import json
import sqlite3
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Optional AI SDKs (import only if available)
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    import openai
except Exception:
    openai = None

load_dotenv()

# -----------------------
# Config (env variables)
# -----------------------
WP_BASE = os.getenv("WP_BASE", "https://healthresearchdigest.co.uk")  # WordPress site base URL
WP_TASK_ENDPOINT = os.getenv("WP_TASK_ENDPOINT", f"{WP_BASE}/wp-json/wp/v2/tasks")  # CPT REST endpoint (adjust if different)
DB_PATH = os.getenv("DB_PATH", "tasks.db")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")  # "gemini" | "openai" | "mock"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL_GEMINI = os.getenv("EMBED_MODEL_GEMINI", "")  # adjust as needed
EMBED_MODEL_OPENAI = os.getenv("EMBED_MODEL_OPENAI", "text-embedding-3-small")  # example

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------
# FastAPI setup
# -----------------------
app = FastAPI(title="AI Task Search (Tasks indexed from WP)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod limit to WP domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# DB helpers
# -----------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY,
        wp_id INTEGER UNIQUE,
        title TEXT,
        description TEXT,
        category TEXT,
        url TEXT,
        embedding TEXT,   -- JSON array stored as TEXT
        updated_at REAL
    )
    """)
    conn.commit()
    conn.close()

def upsert_task(wp_id:int, title:str, description:str, category:str, url:str, embedding:List[float]):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = time.time()
    emb_json = json.dumps(embedding)
    c.execute("""
      INSERT INTO tasks (wp_id, title, description, category, url, embedding, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(wp_id) DO UPDATE SET
        title=excluded.title,
        description=excluded.description,
        category=excluded.category,
        url=excluded.url,
        embedding=excluded.embedding,
        updated_at=excluded.updated_at
    """, (wp_id, title, description, category, url, emb_json, now))
    conn.commit()
    conn.close()

def fetch_all_indexed_tasks():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT wp_id, title, description, category, url, embedding FROM tasks")
    rows = c.fetchall()
    conn.close()
    tasks = []
    for wp_id, title, description, category, url, emb_json in rows:
        embedding = json.loads(emb_json) if emb_json else None
        tasks.append({
            "wp_id": wp_id,
            "title": title,
            "description": description,
            "category": category,
            "url": url,
            "embedding": embedding
        })
    return tasks

# -----------------------
# Embedding helpers
# -----------------------
def get_embedding_gemini(text: str):
    if not genai:
        raise RuntimeError("google.generativeai is not installed. pip install google-generative-ai")
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")

    genai.configure(api_key=GOOGLE_API_KEY)

    try:
        # Correct embeddings call
        res = genai.embed_content(
            model=EMBED_MODEL_GEMINI or "models/embedding-001",   # e.g., "models/embedding-001"
            content=text
        )
        return res["embedding"]
    except Exception as e:
        raise RuntimeError(f"Gemini embedding failed: {e}")

def get_embedding_openai(text: str):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    res = client.embeddings.create(
        model=EMBED_MODEL_OPENAI,
        input=text
    )
    return res.data[0].embedding


def get_embedding_mock(text: str, dim=1536):
    # Deterministic pseudo-embedding for dev (not semantic)
    # Use a hash -> seed -> random vector so same text => same vector
    import hashlib, random
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    seed = int(h[:8], 16)
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def get_embedding(text: str):
    provider = EMBEDDING_PROVIDER.lower()
    if provider == "gemini":
        try:
            return get_embedding_gemini(text)
        except Exception as e:
            raise RuntimeError(f"Gemini embedding failed: {e}")
    elif provider == "openai":
        try:
            return get_embedding_openai(text)
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding failed: {e}")
    else:
        # mock
        return get_embedding_mock(text)

# -----------------------
# WP fetch & index flow
# -----------------------
def fetch_tasks_from_wp(per_page=100, pages=10):
    """
    Fetch tasks CPT from WordPress REST API.
    Works with both default categories and custom task taxonomies.
    """
    results = []
    page = 1
    while page <= pages:
        params = {"per_page": per_page, "page": page}
        try:
            r = requests.get(WP_TASK_ENDPOINT, params=params, timeout=15)
            if r.status_code >= 400:
                break
            arr = r.json()
        except Exception:
            break
        if not arr:
            break

        for item in arr:
            wp_id = item.get("id")
            title = item.get("title", {}).get("rendered") if isinstance(item.get("title"), dict) else item.get("title")
            description = item.get("excerpt", {}).get("rendered") if isinstance(item.get("excerpt"), dict) else item.get("excerpt") or item.get("content", {}).get("rendered", "")

            # --- CATEGORY HANDLING ---
            # --- TAG HANDLING ---
            tasktag = "Untagged"

            # Case 1: Default WP tags
            if isinstance(item.get("tags"), list) and item["tags"]:
                tasktag = f"Tag-{item['tags'][0]}"

            # Case 2: Custom taxonomy (tasktags)
            elif "_embedded" in item and "wp:term" in item["_embedded"]:
                for tax in item["_embedded"]["wp:term"]:
                    if isinstance(tax, list) and tax:
                        # Find taxonomy object named 'tasktags'
                        if "taxonomy" in tax[0] and tax[0]["taxonomy"] in "tasktags":
                            tasktag = tax[0].get("name", tasktag)

            # --- URL ---
            url = item.get("link") or f"{WP_BASE}/?p={wp_id}"

            results.append({
                "wp_id": wp_id,
                "title": strip_html(title) if title else "",
                "description": strip_html(description) if description else "",
                "category": tasktag,
                "url": url
            })
        page += 1
    return results


def strip_html(s):
    # naive html stripper
    import re
    return re.sub('<[^<]+?>', '', s) if s else s

# -----------------------
# Search helpers
# -----------------------
def cosine_sim(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def vector_search(query_emb, tasks, top_k=4):
    scores = []
    for t in tasks:
        if not t.get("embedding"):
            continue
        s = cosine_sim(query_emb, t["embedding"])
        scores.append((s, t))
    scores.sort(key=lambda x: x[0], reverse=True)
    top = [ {"score": float(s),"title": t["title"], "category": t["category"], "url": t["url"], "description": t["description"]} for s,t in scores[:top_k] ]
    return top

# -----------------------
# API Schemas
# -----------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 4

class ReindexRequest(BaseModel):
    force: bool = False

# -----------------------
# Endpoints
# -----------------------
@app.post("/reindex")
async def reindex(req: ReindexRequest = Body(...)):
    """
    Fetch tasks from WP, compute embeddings, store in SQLite.
    Set EMBEDDING_PROVIDER to 'gemini' for now in env vars.
    """
    tasks = fetch_tasks_from_wp()
    if not tasks:
        raise HTTPException(status_code=400, detail="No tasks fetched from WP. Check WP_TASK_ENDPOINT.")
    # compute embeddings (may be slow) - do sequentially to avoid rate issues
    for idx, t in enumerate(tasks):
        text_for_embedding = f"{t['title']}. {t['description'] or ''} Category: {t['category']}"
        try:
            emb = get_embedding(text_for_embedding)
            # Normalize to list of floats (some SDKs return numpy arrays, etc.)
            emb_list = list(map(float, emb))
        except Exception as e:
            # fallback to mock to avoid failing entire index
            emb_list = get_embedding_mock(text_for_embedding)
        upsert_task(t["wp_id"], t["title"], t["description"], t["category"], t["url"], emb_list)
    return {"ok": True, "indexed": len(tasks)}

@app.post("/search")
async def search(req: SearchRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")
    try:
        q_emb = get_embedding(q)
    except Exception as e:
        # in case of embedding failure, return error
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")
    # load tasks
    tasks = fetch_all_indexed_tasks()
    if not tasks:
        raise HTTPException(status_code=500, detail="No indexed tasks. Run /reindex first.")
    # ensure embeddings are numeric lists
    for t in tasks:
        if t["embedding"] is None:
            t["embedding"] = get_embedding_mock(t["title"] + " " + (t["description"] or ""))
    top = vector_search(q_emb, tasks, top_k=req.top_k)
    return {"results": top}

@app.get("/health")
async def health():
    return {"ok": True, "provider": EMBEDDING_PROVIDER}

# -----------------------
# Startup
# -----------------------
if __name__ == "__main__":
    # initialize DB
    init_db()
    print("DB initialized at", DB_PATH)
    # If you want to auto-reindex on start (dev), uncomment:
    # from time import sleep; sleep(1); import asyncio; asyncio.run(reindex(ReindexRequest(force=True)))
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)


    
