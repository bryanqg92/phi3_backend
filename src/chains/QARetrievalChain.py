from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from src.vectorstore.vectorstore import vectorstore


class QARetrievalChain(RetrievalQA):
        
    def __init__(self, llm: HuggingFacePipeline):
        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type = 'stuff',
            retriever = vectorstore.as_retriever(search_kwargs={'k':5}),
        )

    def GetQAChain(self):
        return self.qa