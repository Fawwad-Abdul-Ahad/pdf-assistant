from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=3,
    doc_content_chars_max=4000
)

query = "Artificial Intelligence"

docs = retriever.invoke(query)

for doc in docs:
    print(doc.page_content)
    print("-" * 80)