import streamlit as st
from main import load_documents, prepare_knowledge_base, build_qa_chain

st.title("Manual Assistant – Violeta d.o.o.")

if "qa_chain" not in st.session_state:
    with st.spinner("Učitavanje manuala..."):
        docs = load_documents("pdfs")
        vectordb = prepare_knowledge_base(docs)
        st.session_state.qa_chain = build_qa_chain(vectordb)

question = st.text_input("Unesi pitanje o stroju: ")

if question:
    response = st.session_state.qa_chain.run(question)
    st.markdown("### Odgovor:")
    st.write(response)
