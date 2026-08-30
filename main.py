from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
create_db()
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

chain = template | model | parser
# print(result)
