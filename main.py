from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Search Demo")

# Allow CORS so WordPress frontend can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production, restrict to your WP domain
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 4

# Demo AI tools dataset
DEMO_AI_TOOLS = [
    {"id": 1, "title": "DALL·E 3", "url": "https://openai.com/dall-e", "excerpt": "Generates AI images from text prompts."},
    {"id": 2, "title": "ChatGPT", "url": "https://chat.openai.com", "excerpt": "AI chatbot that can write, debug, summarize text."},
    {"id": 3, "title": "MidJourney", "url": "https://www.midjourney.com", "excerpt": "AI tool for generating creative images."},
    {"id": 4, "title": "Whisper", "url": "https://openai.com/research/whisper", "excerpt": "OpenAI model for speech-to-text transcription."},
    {"id": 5, "title": "Claude AI", "url": "https://www.anthropic.com", "excerpt": "AI assistant for research, writing, and summarization."},
    {"id": 6, "title": "Jasper AI", "url": "https://www.jasper.ai", "excerpt": "AI copywriting assistant for marketing and content."},
    {"id": 7, "title": "Runway", "url": "https://runwayml.com", "excerpt": "AI-powered video and image editing platform."}
]

@app.post("/search")
async def search(req: SearchRequest):
    query = req.query.lower()

    # Simple keyword-based matching for demo
    filtered = []
    for tool in DEMO_AI_TOOLS:
        # Check if any word in title or excerpt matches query
        if any(word in query for word in tool["title"].lower().split() + tool["excerpt"].lower().split()):
            filtered.append(tool)

    # If no match, return top 4 by default
    if not filtered:
        filtered = DEMO_AI_TOOLS[:4]

    return {"results": filtered[:req.top_k]}
