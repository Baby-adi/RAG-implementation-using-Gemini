from pipeline import build_rag
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Chat with your PDF", layout="wide")
st.title("pdfbot")

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    pdf_path = f"uploaded_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Reset store.json for each new file
    if os.path.exists("store.json"):
        os.remove("store.json")

    if "rag_app" not in st.session_state:
        st.session_state.rag_app = build_rag(pdf_path, model_name="gemini-1.5-flash")
        st.session_state.history = []

    user_q = st.chat_input("Ask about the PDF...")

    if user_q:
        result = st.session_state.rag_app.invoke({
            "question": user_q,
            "history": st.session_state.history
        })
        new_turn = f"Q: {user_q}\nA: {result['answer']}"
        st.session_state.history.append(new_turn)

    for turn in st.session_state.history:
        q, a = turn.split("\nA: ")
        st.chat_message("user").write(q.replace("Q: ", ""))
        st.chat_message("assistant").write(a)
else:
    st.info("Upload the PDF to get started.")