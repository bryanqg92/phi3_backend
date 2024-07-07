from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from src.vectorstore.vectorstore import vectorstore as vs
from src.text_splitter.DocTextSplitter import DocTextSplitter
from langchain.prompts import PromptTemplate


class QARetrievalChain():


    
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

        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type = 'stuff',
            retriever = vectorstore.as_retriever(search_kwargs={'k':3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def GetQAChain(self):
        return self.qa
    