from src.models.embeddings.bge import bge_model


class LoadEmbeddingModel:

    @classmethod
    def initialize_embeddings(cls):
        bge_model.InitEmbeddings()