from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()
def create_db():
    data = PyPDFLoader(r'C:\Users\fawwa\Desktop\Rag Project\React Native & JavaScript - 400 Practice Questions.pdf')
    docs = data.load()

    splitter  = RecursiveCharacterTextSplitter(
        separators='',
        chunk_size = 500,
        chunk_overlap = 100
    )


    chunks = splitter.split_documents(docs)
    print(len(chunks))

    embedding_model = MistralAIEmbeddings()

    vectorStore = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory='chroma_db'
    )

