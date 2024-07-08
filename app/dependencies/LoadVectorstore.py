from src.vectorstore.vectorstore import VectorStore

class LoadVectorstore:
    vectorstore_ins = VectorStore().get_vectorstore()

    @classmethod
    def GetVectorstore(cls):
        return cls.vectorstore