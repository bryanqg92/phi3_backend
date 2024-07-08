from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from src.text_splitter.DocTextSplitter import DocTextSplitter
import torch

class VectorStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.model_name = "BAAI/bge-reranker-base"
        self.encode_kwargs = {'normalize_embeddings': True}
        self.model_norm = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': torch.device("cuda" if torch.cuda.is_available() else "cpu",)
                            'cache_dir':"offload_models/phi3"},
            encode_kwargs=self.encode_kwargs
            
        )

        self.vectorstore = Chroma.from_documents(
            DocTextSplitter.docs, 
            self.model_norm,
            persist_directory="./chroma_db" 
        )

    def get_vectorstore(self):
        return self.vectorstore

