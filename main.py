from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatMistralAI(
    model = "mistral-small-2506"
)

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful AI assistant to summarize the following text"),
        ("human", "{data}")
    ]
)

parser = StrOutputParser()

data = PyPDFLoader(r'C:\Users\fawwa\Desktop\Rag Project\React Native & JavaScript - 400 Practice Questions.pdf')
docs = data.load()

splitter  = RecursiveCharacterTextSplitter(
    separators='',
    chunk_size = 500,
    chunk_overlap = 100
)
chunks = splitter.split_documents(docs)
print(len(chunks))
chain = template | model | parser
result = chain.invoke({"data" : docs})

# print(result)
