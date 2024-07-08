from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import torch


class bge_model:
        

        _CACHE_DIR = "./offload_models/embeddings/bge"
        _model_name = "BAAI/bge-reranker-base"
        _encode_kwargs = {'normalize_embeddings': True}
        model_norm = None

        @classmethod
        def InitEmbeddings(cls):
            cls.model_norm = HuggingFaceBgeEmbeddings(
                model_name=cls._model_name,
                model_kwargs={
                    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                },
                encode_kwargs=cls._encode_kwargs,
                cache_folder=cls._CACHE_DIR  
            )