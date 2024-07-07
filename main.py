from src.models.phi3 import phi3_model
from src.models.llama2_7b import llama2_7b_model
from src.text_splitter.DocTextSplitter import DocTextSplitter
from src.chains.QARetrievalChain import QARetrievalChain
from src.chains.QARetrievalReranker import QARetrievalReranker

def main(prompt:str):

    print("\n   ==> Starting Application")
    print(f"\n   ==> {prompt}")
    model1 = phi3_model()
    model2 = llama2_7b_model()
    print("\n   ==> Loaded Models")

    DocTextSplitter.LoadAndSplit("docs/Informe Final CD.pdf")
    print("\n   ==>Loaded Documents and splitted")

    qa = QARetrievalChain(model1.getPipeline()).GetQAChain()
    qa2 = QARetrievalReranker(model2.getPipeline()).GetQAChain()
    print("\n   ==>Loaded QA Chain")

    query = {"query": prompt}
    response = qa.invoke(query)
    response2 = qa2.invoke(query)

    print("\n   Response phi: " + response['result'])
    print("\n   Response: LlaMa " + response2['result'])

if __name__ == "__main__":
    main("Hay conjuntos difusos?")