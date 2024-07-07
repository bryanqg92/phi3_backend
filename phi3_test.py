from src.models.phi3 import phi3_model
from src.models.llama2_7b import llama2_7b_model
from src.text_splitter.DocTextSplitter import DocTextSplitter
from src.chains.QARetrievalChain import QARetrievalChain
from src.chains.QARetrievalReranker import QARetrievalReranker

def main(prompt:str):

    print("\n   ==> Starting Application")
    print(f"\n   ==> {prompt}")
    model1 = phi3_model()
    print("\n   ==> Loaded Model")

    DocTextSplitter.LoadAndSplit("docs/Informe Final CD.pdf")
    print("\n   ==>Loaded Documents and splitted")

    qa = QARetrievalChain(model1.getPipeline()).GetQAChain()
    print("\n   ==>Loaded QA Chain")

    query = {"query": prompt}
    response = qa.invoke(query)

    print("\n   Response phi: " + response['result'])

if __name__ == "__main__":
    main("Hablame sobre el asistente")