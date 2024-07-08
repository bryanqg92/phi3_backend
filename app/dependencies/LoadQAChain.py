from src.chains.QARetrievalChain import QARetrievalChain
from app.dependencies.LoadModel import LoadModel
from app.dependencies.LoadVectorstore import LoadVectorstore

class LoadQAChain:

    qa_chain_ins = None

    @classmethod
    def LLMResponse(cls, prompt: str):
        response = cls.qa_chain_ins.invoke({"query": prompt})

        if LoadModel.get_model_type() == "phi3":
            response = response['result']
        else:
            response = response['result']
        return response
    @classmethod
    def init_chain(cls):
        cls.qa_chain_ins = QARetrievalChain(LoadModel.get_qa_model(), LoadVectorstore.GetVectorstore()).GetQAChain()
        print(f"aqui las keys{cls.qa_chain_ins.input_keys}")