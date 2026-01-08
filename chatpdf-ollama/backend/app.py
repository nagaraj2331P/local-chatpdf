import os
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from pdf_utils import load_pdf
from vector_store import add_text, search

# --------------------------------
# App init
# --------------------------------
app = FastAPI()

# --------------------------------
# Paths
# --------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # chatpdf-ollama
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

os.makedirs(PDF_DIR, exist_ok=True)

# --------------------------------
# Serve frontend
# --------------------------------
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def home():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


# --------------------------------
# Upload PDF
# --------------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    path = os.path.join(PDF_DIR, file.filename)

    # Save PDF
    with open(path, "wb") as f:
        f.write(await file.read())

    # Extract text
    text = load_pdf(path)

    if not text or len(text.strip()) < 50:
        return {
            "status": "error",
            "message": "PDF has no readable text (scanned PDF not supported)"
        }

    # Add to vector store
    add_text(text)

    return {"status": "PDF indexed successfully"}


# --------------------------------
# Ask Question
# --------------------------------
@app.post("/ask")
async def ask(question: str = Form(...)):
    chunks = search(question)

    if not chunks:
        return {"answer": "No relevant information found in the document."}

    # Use top chunks safely
    context = "\n".join(chunks[:3])

    prompt = f"""
Answer using the context only.
If not found, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi",          # ✅ lightweight model
                "prompt": prompt,
                "stream": False
            },
            timeout=300                  # ✅ increased timeout
        )

        if response.status_code != 200:
            return {
                "answer": f"Ollama error: {response.text}"
            }

        data = response.json()
        return {"answer": data.get("response", "No response from model")}

    except requests.exceptions.ConnectionError:
        return {
            "answer": "Ollama server is not running. Please start Ollama."
        }

    except requests.exceptions.Timeout:
        return {
            "answer": "Model is taking too long. Try a simpler question."
        }

    except Exception as e:
        return {
            "answer": f"Unexpected error: {str(e)}"
        }
