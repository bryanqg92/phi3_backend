from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from src.vectorstore.vectorstore import VectorStore
from langchain.prompts import PromptTemplate


class QARetrievalChain():
    
    def __init__(self, llm: HuggingFacePipeline, vectorstore: VectorStore):
        
        self.custom_prompt_template = """
        Usa la siguiente información para responder a la pregunta del usuario.
        Si no sabes la respuesta, simplemente di que no lo sabes, no intentes inventar una respuesta.

        Contexto: {context}
        Pregunta: {question}

        Solo devuelve la primera respuesta sin devolver todo el conxtexto, responde siempre en español
        a continuación.
        ###
        """
        self.prompt = PromptTemplate(template=self.custom_prompt_template,
                        input_variables=['context', 'question'])

        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type = 'stuff',
            retriever = vectorstore.as_retriever(search_kwargs={'k':5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def GetQAChain(self):
        return self.qa
    