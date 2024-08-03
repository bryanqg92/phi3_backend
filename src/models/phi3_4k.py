import torch
from transformers import AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.llms import Ollama

class phi3_model():

    def __init__(self) -> None:
        
        self.model_name = "phi3"
        self.model = Ollama(model=self.model_name,
                            temperature=0.0,
                            )
    
    def getPipeline(self):
        return self.model

        