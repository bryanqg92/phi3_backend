from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from src.vectorstore.vectorstore import VectorStore


class QARerankChain():
    
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


        self.compressor = FlashrankRerank()
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=vectorstore.as_retriever(search_kwargs={'k':3}))
        
        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type = 'stuff',
            retriever = self.compression_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def GetQAChain(self):
        return self.qa
    