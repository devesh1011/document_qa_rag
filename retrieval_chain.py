from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_openai import ChatOpenAI

def create_retrieval_chain(vector_store):
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    def get_unique_union(documents):
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]

    queries = (
        (prompt_perspectives | ChatOpenAI(temperature=0.1))
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )
    retrieval_chain = queries | vector_store.as_retriever().map() | get_unique_union
    return retrieval_chain
