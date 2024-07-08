from src.models.embeddings.bge import bge_model
from langchain_community.vectorstores import Chroma
from src.text_splitter.DocTextSplitter import DocTextSplitter
import os
import torch

class VectorStore:
    _instance = None
    _DB_VECTORIAL_NAME="documents_to_talentum"
    _CACHE_DIR = "./offload_models/embeddings/bge"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):

        if not os.path.exists(self._DB_VECTORIAL_NAME):
            self.vectorstore = Chroma.from_documents(
                DocTextSplitter.docs, 
                bge_model.model_norm,
                persist_directory=VectorStore._DB_VECTORIAL_NAME
            )
            self.vectorstore.persist()
        else:
            self.vectorstore = Chroma(
                persist_directory=VectorStore._DB_VECTORIAL_NAME,
                embedding_function=bge_model.model_norm,
            )

    def get_vectorstore(self):
        return self.vectorstore

