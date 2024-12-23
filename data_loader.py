import os
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_files(files):
    all_splits = []

    for file in files:
        doc_loader = PyPDFLoader(file)
        docs = doc_loader.load()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents=docs)
        all_splits.extend(splits)
    return all_splits
