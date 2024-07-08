from src.vectorstore.vectorstore import VectorStore

class LoadVectorstore:
    vectorstore_ins = None

    @classmethod
    def GetVectorstore(cls):
        return cls.vectorstore
    
    @classmethod
    def init_vectorstore():
        cls.vectorstore_ins = VectorStore().get_vectorstore()