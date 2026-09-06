from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def format_document(documents):
    return '\n'.join(
        document.page_content for document in documents
    )
    
    
model = ChatMistralAI(
    model = "mistral-small-2506"
)
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.
Answer the user's question using only the provided context.

If the answer cannot be found in the context, reply:
"Sorry, this information is not available in the provided context."

Do not use outside knowledge."""
        ),
        (
            "human",
            """Context:
{context}

Question:
{query}"""
        )
    ]
)

embedding_model = MistralAIEmbeddings()
vector_stores = Chroma(
    embedding_function = embedding_model,
    persist_directory='chroma_db'
)

retriever = vector_stores.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k" : 10,
        "lambda_mult":0.5
    }
)

parser = StrOutputParser()

rag_chain = (
    {
        "context" : retriever | RunnableLambda(format_document),
        "query" : RunnablePassthrough()
    }
    | template
    | model 
    | parser
)

while True:
    query = input ("You :")
    if query == 'exit':
        break
    result = rag_chain.invoke(query)
    print("Bot",result)

# print(result)}
