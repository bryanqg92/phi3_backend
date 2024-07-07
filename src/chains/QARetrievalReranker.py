from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from src.vectorstore.vectorstore import vectorstore as vs
from src.text_splitter.DocTextSplitter import DocTextSplitter
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

class QARetrievalReranker:
    
    def __init__(self, llm: HuggingFacePipeline):
        vectorstore_ins = vs(DocTextSplitter.docs)
        vectorstore = vectorstore_ins.get_vectorstore()

        self.custom_prompt_template = """Usa la siguiente información para responder a la pregunta del usuario.
        Si no sabes la respuesta, simplemente di que no lo sabes, no intentes inventar una respuesta.

        Contexto: {context}
        Pregunta: {question}

        Solo devuelve la respuesta útil a continuación y nada más y responde siempre en español
        ###
        """
        self.prompt = PromptTemplate(template=self.custom_prompt_template,
                        input_variables=['context', 'question'])

        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        base_retriever = vectorstore.as_retriever(search_kwargs={'k': 10})
        
        self.retriever = ContextualCompressionRetriever(
            base_compressor=LLMChainExtractor.from_llm(llm),
            base_retriever=base_retriever
        )

        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def rerank_documents(self, query, docs, top_k=3):
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:top_k]]

    def GetQAChain(self):
        return self._answer_question

    def _answer_question(self, query_dict):
        query = query_dict["query"]
        raw_docs = self.retriever.get_relevant_documents(query)
        reranked_docs = self.rerank_documents(query, raw_docs)
        
        context = "\n".join([doc.page_content for doc in reranked_docs])
        
        response = self.qa({"query": query, "context": context})
        return response['result']