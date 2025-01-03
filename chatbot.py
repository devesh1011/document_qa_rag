import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain.load import loads, dumps

load_dotenv()

# st.write(
#     "Has environment variables been set:",
#     os.environ["OPENAI_API_KEY"] == st.secrets["openai_api_key"],
# )

template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)


def save_uploaded_file(uploaded_file):
    data_dir = "data/"
    os.makedirs(data_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = os.path.join(data_dir, f"{timestamp}_{uploaded_file.name}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_docs(filepath):
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    return docs


def split_and_save_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    return chunks


def get_vector_store(chunks):
    embeddings = OpenAIEmbeddings(api_key=st.secrets["openai_api_key"])
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    return vectorstore


def get_unique_union(documents: list[list]):
    """Unique union of retrieved docs"""
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


def multi_query_chain(question: str):
    """Generate multiple versions of the user question for better retrieval."""
    return (
        prompt_perspectives
        | ChatOpenAI(
            temperature=0.1, model="gpt-4", api_key=st.secrets["openai_api_key"]
        )
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )


def conversational_chain(retriever):
    """Create a conversational chain with multi-query retrieval."""
    prompt = hub.pull("rlm/rag-prompt")

    # Define the multi-query chain
    multi_query = RunnablePassthrough() | (
        lambda x: multi_query_chain(x)
    )  # Pass the question directly

    # Combine multi-query with retriever and unique union
    retrieval_chain = multi_query | retriever.map() | get_unique_union

    # Define the LLM and final RAG chain
    llm = ChatOpenAI(
        temperature=0.1, model="gpt-4", api_key=st.secrets["openai_api_key"]
    )
    rag_chain = (
        {"context": retrieval_chain | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def multi_query_chain(question: str):
    """Generate multiple versions of the user question for better retrieval."""
    return (
        prompt_perspectives
        | ChatOpenAI(
            temperature=0.1, model="gpt-4", api_key=st.secrets["openai_api_key"]
        )
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question."}
    ]


def main():
    st.set_page_config(page_title="Conversational PDF Chatbot", page_icon="🤖")
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    with st.sidebar:
        st.title("Menu:")
        uploaded_file = st.file_uploader("Upload a document", type=["pdf"])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if uploaded_file is not None:
                    filepath = save_uploaded_file(uploaded_file)
                    docs = load_docs(filepath)
                    chunks = split_and_save_docs(docs)
                    vector_store = get_vector_store(chunks)
                    st.session_state.vector_store = vector_store
                    st.success("File Uploaded Successfully")
                else:
                    st.warning("Please upload a PDF file.")

    st.title("Chat with PDF files using OpenAI 🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Upload any PDF and start asking me questions!",
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        if st.session_state.vector_store:
            retriever = st.session_state.vector_store.as_retriever()
            chain = conversational_chain(retriever)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chain.invoke(prompt)
                    st.write(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
        else:
            with st.chat_message("assistant"):
                st.write(
                    "Please upload a PDF file and process it before asking questions."
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Please upload a PDF file and process it before asking questions.",
                    }
                )


if __name__ == "__main__":
    main()
