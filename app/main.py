from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from fastapi.responses import FileResponse, HTMLResponse
from src.text_splitter.DocTextSplitter import DocTextSplitter

from app.dependencies.LoadModel import LoadModel
from app.dependencies.LoadQAChain import LoadQAChain
from app.dependencies.LoadQARerankChain import LoadQARerankChain
from app.dependencies.LoadVectorstore import LoadVectorstore
from app.dependencies.LoadEmbeddingModel import LoadEmbeddingModel

import shutil
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.on_event("startup")
async def startup_event():
    LoadModel.initialize_model()
    LoadEmbeddingModel.initialize_embeddings()

@app.get("/", response_class=HTMLResponse)
async def get_home():
    return FileResponse("chatbot.html")

UPLOAD_DIRECTORY = "./docs/"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["text/plain", "application/pdf"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        Path(UPLOAD_DIRECTORY).mkdir(parents=True, exist_ok=True)
        file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        DocTextSplitter.LoadAndSplit(file_location)
        LoadVectorstore.init_vectorstore()
        LoadQAChain.init_chain()
        LoadQARerankChain.init_chain()
        os.remove(file_location)

        return {"message": "cargado y vectorizado"}
    
    except Exception as e:
        return {"message": f"error subiendo archivo: {str(e)}"}


@app.post("/generate")
async def generate_text(prompt: str):
    
    generated_text = LoadQAChain.LLMResponse(prompt)
    return {"generated_text": generated_text}

@app.post("/rerank_generate")
async def generate_text(prompt: str):
    generated_text = LoadQARerankChain.LLMResponse(prompt)
    return {"generated_text": generated_text}

@app.post("/change_model")
async def generate_text(model: str):
    if model not in ["phi3", "llama2_7b", "orca_mini"]:
        raise HTTPException(status_code=400, detail="Invalid model type")
    else:
        LoadModel.change_model(model)

    return {"message": "Model changed"}