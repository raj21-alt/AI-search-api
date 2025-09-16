from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Search Mock")

# Allow CORS (WordPress frontend can call it)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production: restrict to your WP domain
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 4

MOCK_RESULTS = [
    {"id": 1, "title": "ChatGPT", "url": "/tools/chatgpt", "excerpt": "AI assistant for text and code."},
    {"id": 2, "title": "MidJourney", "url": "/tools/midjourney", "excerpt": "AI for generating images."},
    {"id": 3, "title": "Claude AI", "url": "/tools/claude", "excerpt": "Helpful AI for research and writing."},
    {"id": 4, "title": "Whisper", "url": "/tools/whisper", "excerpt": "OpenAI model for speech-to-text."},
]

@app.post("/search")
async def search(req: SearchRequest):
    # Just return mock results (top_k max)
    return {"results": MOCK_RESULTS[: max(1, min(req.top_k, len(MOCK_RESULTS)))]}
