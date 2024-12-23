from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_openai import ChatOpenAI
import replicate

model = replicate.models.get("meta/meta-llama-3.1-405b-instruct")


def generate_response(prompt):
    inputs = {"prompt": prompt}
    output = model.predict(**inputs)
    return output[0]


def create_generator(retrieval_chain):
    prompt_template = """
    Answer the following question based on this context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retrieval_chain, "question": itemgetter("question")}
        | prompt
        | (lambda x: generate_response(x["text"]))
        | StrOutputParser()
    )
    return chain
