from fastapi import FastAPI, UploadFile, HTTPException, File
from contextlib import asynccontextmanager
from src.text_splitter.DocTextSplitter import DocTextSplitter
from app.dependencies.LoadModel import LoadModel
from app.dependencies.LoadQAChain import LoadQAChain
import shutil
import os


app = FastAPI()

@app.on_event("startup")
async def startup_event():
    LoadModel.initialize_model("phi3")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    if file.content_type not in ["text/plain", "application/pdf"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    try:
        file_location = f"docs/{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        
        DocTextSplitter.LoadAndSplit(file_location)
        os.remove(file_location)
        return {"message": "cargado y vectorizado"}
    except Exception:
        return {"message": "Hubo un error al subir el archivo"}



@app.post("/generate")
async def generate_text(prompt: str):
    LoadQAChain.init_chain()
    generated_text = LoadQAChain.LLMResponse(prompt)
    return {"generated_text": generated_text}
'''
@app.post("/rerank_generate")
async def generate_text(prompt: str):
    del LoadAppModel.model 
    generated_text = model.generate(prompt)
    return {"generated_text": generated_text}''' 

@app.post("/change_model")
async def generate_text(model: str):
    if model not in ["phi3", "llama2_7b"]:
        raise HTTPException(status_code=400, detail="Invalid model type")
    else:
        LoadModel.change_model(model)

    return {"message": "Model changed"}