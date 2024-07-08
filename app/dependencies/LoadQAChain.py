from src.chains.QARetrievalChain import QARetrievalChain
from app.dependencies.LoadModel import LoadModel
from app.dependencies.LoadVectorstore import LoadVectorstore

class LoadQAChain:

    qa_chain_ins = None

    @classmethod
    def LLMResponse(self, prompt: str):
        response = self.qa_chain_ins.invoke({"query": prompt})

        if LoadModel.current_get_model_type() == "phi3":
            response = response['result'].split("###")[1].strip()
        else:
            response = response['result'].split("###")[-1].strip()
        return response['result'].split("###")[1].strip()
    
    @classmethod
    def init_chain(cls):
        cls.qa_chain_ins = QARetrievalChain(LoadModel.get_qa_model(), LoadVectorstore.GetVectorstore())