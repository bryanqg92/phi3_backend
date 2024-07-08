from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from src.vectorstore.vectorstore import VectorStore
from src.text_splitter.DocTextSplitter import DocTextSplitter
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

class QARetrievalReranker:


    def __init__(self, llm: HuggingFacePipeline, vectorstore: VectorStore):

        compressor = LLMChainExtractor.from_llm(llm)
        retriever_with_rerank = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vectorstore.as_retriever(search_kwargs={'k': 5})
        )

        self.custom_prompt_template = """Usa la siguiente información para responder a la pregunta del usuario.
        Si no sabes la respuesta, simplemente di que no lo sabes, no intentes inventar una respuesta.

        Contexto: {context}
        Pregunta: {question}

        Solo devuelve la respuesta útil a continuación y nada más y responde siempre en español
        ###
        """
        self.prompt = PromptTemplate(template=self.custom_prompt_template,
                        input_variables=['context', 'question'])

        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever_with_rerank,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def GetQAChain(self):
        return self.qa