from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import time

# Initialize model once
model = OllamaLLM(model="llama3.2")

template = """You are an expert at answering questions about restaurants based on customer reviews.

Here are relevant customer reviews:
{reviews}

Question: {question}

Please provide a helpful, concise answer based on the reviews above."""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def generate_response(question):
    """Generate a response for a given question"""
    # Retrieve relevant reviews
    review_docs = retriever.invoke(question)
    
    # Format reviews for the prompt
    reviews_text = "\n\n".join([
        f"Review {i+1} (Rating: {doc.metadata.get('rating', 'N/A')}): {doc.page_content}"
        for i, doc in enumerate(review_docs)
    ])
    
    # Generate response
    result = chain.invoke({
        "reviews": reviews_text,
        "question": question
    })
    
    return result, review_docs

def main():
    print("\nRestaurant Review RAG Agent")
    print("="*50)
    print("Ask questions about restaurants based on reviews")
    print("Type 'q' to quit\n")
    
    while True:
        print("\n" + "="*50)
        question = input("Your question: ")
        print("="*50)
        
        if question.lower() in ['q', 'quit', 'exit']:
            print("\nGoodbye!")
            break
        
        if not question.strip():
            print("Please enter a question")
            continue
        
        print("\nSearching reviews...")
        start_time = time.time()
        
        try:
            result, review_docs = generate_response(question)
            elapsed = (time.time() - start_time) * 1000
            
            print(f"\nAnswer (generated in {elapsed:.0f}ms):")
            print("-" * 50)
            print(result)
            print("-" * 50)
            print(f"\nBased on {len(review_docs)} relevant reviews")
            
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()