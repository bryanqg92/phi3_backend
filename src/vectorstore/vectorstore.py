from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

from src.text_splitter.DocTextSplitter import DocTextSplitter
import torch

class vectorstore():

    def __init__(self, docs):
        self.docs = docs

        self.model_name = "BAAI/bge-reranker-base"
        self.encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

        self.model_norm = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': torch.device ("cuda" if torch.cuda.is_available() else "cpu")},
            encode_kwargs=self.encode_kwargs
        )

        self.vectorstore = Chroma.from_documents(DocTextSplitter.docs, self.model_norm)


    def get_vectorstore(self):
        return self.vectorstore