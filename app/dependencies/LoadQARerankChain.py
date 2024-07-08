from src.chains.QARerankChain import QARerankChain
from app.dependencies.LoadModel import LoadModel
from app.dependencies.LoadVectorstore import LoadVectorstore

class LoadQARerankChain:

    qa_rerank_ins = None

    @classmethod
    def LLMResponse(cls, prompt: str):
        
        response = cls.qa_rerank_ins.invoke({"query": prompt})
        print(response)
        response_list = response['result'].split("###")
        response = response_list[1].strip()
        return response
    
    @classmethod
    def init_chain(cls):
        cls.qa_rerank_ins = QARerankChain(LoadModel.get_qa_model(), LoadVectorstore.GetVectorstore()).GetQAChain()
        print(f"aqui las keys{cls.qa_rerank_ins.input_keys}")