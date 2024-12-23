from langchain_community.vectorstores import Chroma
import dotenv
from langchain.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-base-en"

dotenv.load_dotenv()


def create_vector_store(documents):
    embedding = HuggingFaceBgeEmbeddings(model_name=model_name, encode_kwargs={'normalize_embeddings': True})
    return Chroma.from_documents(documents=documents, embedding=embedding)
