from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from retreiver import process_pdf
from graphnodes import RAGState, make_retrieve_node, make_generate_node


def build_rag(pdf_path, model_name="gemini-1.5-flash", store_path="store.json"):
    retriever = process_pdf(pdf_path, store_path)
    llm = ChatGoogleGenerativeAI(model=model_name)

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", make_retrieve_node(retriever))
    graph.add_node("generate", make_generate_node(llm))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()