from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


class DocTextSplitter():
     
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            length_function=len,
            chunk_overlap=20
            ) 
    docs = None

    @classmethod
    def LoadAndSplit(cls, file_path):  
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        cls.docs = cls.text_splitter.split_documents(documents)   
        return cls.docs