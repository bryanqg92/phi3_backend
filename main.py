from src.models.phi3 import phi3_model
from src.text_splitter.DocTextSplitter import DocTextSplitter
from src.chains.QARetrievalChain import QARetrievalChain
from src.chains.QARetrievalReranker import QARetrievalReranker

def main(prompt:str):

    print("\n   ==> Starting Application")
    print(f"\n   ==> {prompt}")
    model = phi3_model()
    print("\n   ==> Loaded Model")

    DocTextSplitter.LoadAndSplit("docs/Informe Final CD.pdf")
    print("\n   ==>Loaded Documents and splitted")

    #qa = QARetrievalChain(model.getPipeline()).GetQAChain()
    qa = QARetrievalReranker(model.getPipeline()).GetQAChain()
    print("\n   ==>Loaded QA Chain")

    query = {"query": prompt}
    response = qa(query)
    print("\n   Response: " + response['result'].split("###")[-1].strip())

if __name__ == "__main__":
    main("Hay conjuntos difusos?")