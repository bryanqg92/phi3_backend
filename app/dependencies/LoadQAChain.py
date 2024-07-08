from src.chains.QARetrievalChain import QARetrievalChain
from app.dependencies.LoadModel import LoadModel
from app.dependencies.LoadVectorstore import LoadVectorstore

class LoadQAChain:

    qa_chain_ins = None

    @classmethod
    def LLMResponse(cls, prompt: str):
        response = cls.qa_chain_ins.invoke({"query": prompt})

        if LoadModel.get_model_type() == "phi3":
            print(response)
            response_list = response['result'].split("###")
            if len(response_list) > 1:
                response = response_list[1].strip()
            else:
                response = response_list[-1].strip()
        else:
            response_list = response['result'].split("###")
            response = response_list[-1].strip()

        return response
    
    @classmethod
    def init_chain(cls):
        cls.qa_chain_ins = QARetrievalChain(LoadModel.get_qa_model(), LoadVectorstore.GetVectorstore()).GetQAChain()
        print(f"aqui las keys{cls.qa_chain_ins.input_keys}")