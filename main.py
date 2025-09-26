
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about restaurants
Here are some relevant reviews : {reviews}
Here is the question to answer : {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n=======================================")
    question = input("Ask your question (q to quit) : ")
    print("\n\n=======================================")
    if question == "q":
        break

    # we invoke the retriever with our input questions, which will then fetch top 5 vectors match
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews" : reviews, "question" : question})
    print(result)


