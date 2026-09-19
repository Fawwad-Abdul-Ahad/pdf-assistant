from langchain_tavily import TavilySearch
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
news_tool = TavilySearch(max_results=5)

llm = ChatMistralAI(
    model = "labs-leanstral-1-5-1"
)

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful AI assistant.
    Summarize the provided news article.
    If the answer cannot be found in the context, reply:
    "Sorry, this information is not available in the provided context."
    Do not use outside knowledge.
    Context:
    {news}
    """
)

news_result = news_tool.run("latest news of pakistan cricket?")
print(news_result)
parser = StrOutputParser()

chain =  prompt | llm | parser

result = chain.invoke({"news" : news_result})
print(result)