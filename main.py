import streamlit as st
import os
from data_loader import load_pdf_files
from text_splitter import create_text_splitter
from vector_store import create_vector_store
from retrieval_chain import create_retrieval_chain
from generator import create_generator

st.title("Document Q&A System")

uploaded_files = st.file_uploader(
    "Upload PDF documents", accept_multiple_files=True, type=["pdf"]
)

if uploaded_files:
    if (
        "vectorstore" not in st.session_state
    ):  
        with st.spinner("Processing documents..."):
            try:
                temp_files = []
                for uploaded_file in uploaded_files:
                    temp_path = f"data/temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_files.append(temp_path)
                    print(temp_files)

                docs = load_pdf_files(temp_files)  
                for temp_path in temp_files:
                    os.remove(temp_path)

                vector_store = create_vector_store(docs)

                st.session_state.vectorstore = vector_store  
                st.success("Documents processed successfully!")

            except Exception as e:
                st.error(f"Error processing documents: {e}")
                st.stop()  

    else:
        vector_store = st.session_state.vectorstore
        st.info("Using existing vector store.")

    query = st.text_input("Ask a question about the documents:")

    if query:
        if "retrieval_chain" not in st.session_state:
            with st.spinner("Creating retrieval chain..."):
                try:
                    retrieval_chain = create_retrieval_chain(vector_store)
                    st.session_state.retrieval_chain = retrieval_chain
                except Exception as e:
                    st.error(f"Error creating retrieval chain: {e}")
                    st.stop()
        else:
            retrieval_chain = st.session_state.retrieval_chain
            st.info("Using existing retrieval chain")

        if "generator" not in st.session_state:
            with st.spinner("Creating generator..."):
                try:
                    generator = create_generator(retrieval_chain)
                    st.session_state.generator = generator
                except Exception as e:
                    st.error(f"Error creating generator: {e}")
                    st.stop()
        else:
            generator = st.session_state.generator
            st.info("Using existing generator.")

        with st.spinner("Generating answer..."):
            try:
                result = generator.invoke({"question": query})
                st.write("Answer:")
                st.write(result)
            except Exception as e:
                st.error(f"Error generating answer: {e}")
else:
    st.info("Please upload PDF documents to begin.")
