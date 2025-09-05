from typing import TypedDict, List

class RAGState(TypedDict):
    question: str
    context: List[str]
    answer: str
    history: List[str]

def make_retrieve_node(retriever):
    def retrieve_node(state: RAGState):
        docs = retriever.get_relevant_documents(state["question"])
        return {"context": [d["page_content"] for d in docs], "history": state["history"]}
    return retrieve_node

def make_generate_node(llm):
    """Node: Generate answer using retrieved context + LLM."""
    def generate_node(state: RAGState):
        context_text = "\n".join(state["context"])
        prompt = f"""You are an expert assistant. 
        Answer concisely and cite evidence from the PDF. 
        If you are unsure, say 'I don't know.'

        Context:
        {context_text}

        Question: {state['question']}
        Answer:"""
        answer = llm.invoke(prompt).content  # gemini returns full response object so return only content
        return {"answer": answer, "history": state["history"]}
    return generate_node