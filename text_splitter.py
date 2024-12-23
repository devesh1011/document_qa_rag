from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_text_splitter(chunk_size=1000, chunk_overlap=100):
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
