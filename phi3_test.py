import sys
from src.models.phi3 import phi3_model
from src.text_splitter.DocTextSplitter import DocTextSplitter
from src.chains.QARetrievalChain import QARetrievalChain
from src.chains.QARetrievalReranker import QARetrievalReranker

def main(**kwargs):
    prompt = kwargs.get("prompt", "")
    rerank = kwargs.get("rerank", False)

    print("\n   ==> Starting Application")
    print(f"\n   ==> {prompt}")
    model = phi3_model()
    print("\n   ==> Loaded Model")

    DocTextSplitter.LoadAndSplit("docs/Informe Final CD.pdf")
    print("\n   ==> Loaded Documents and splitted")

    if rerank:
        qa = QARetrievalReranker(model.getPipeline()).GetQAChain()
    else:
        qa = QARetrievalChain(model.getPipeline()).GetQAChain()
    print("\n   ==> Loaded QA Chain")

    query = {"query": prompt}
    response = qa.invoke(query)

    print("\n   Response phi: " + response['result'])

if __name__ == "__main__":
    kwargs = {}
    args = sys.argv[1:]
    
    # Parse positional argument
    if len(args) > 0:
        kwargs["prompt"] = args[0]
    
    # Parse optional arguments
    if "--rerank" in args:
        kwargs["rerank"] = True

    main(**kwargs)
