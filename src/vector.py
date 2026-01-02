from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Initialize embeddings once (not per query)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"

def initialize_vector_store():
    """Initialize or load the vector store"""
    print("🔧 Initializing vector store...")
    
    df = pd.read_csv("realistic_restaurant_reviews.csv")
    add_documents = not os.path.exists(db_location)
    
    # Create/load vector store
    vector_store = Chroma(
        collection_name="restaurant_reviews",
        persist_directory=db_location,
        embedding_function=embeddings
    )
    
    # Add documents only if database doesn't exist
    if add_documents:
        print(f"Adding {len(df)} documents to vector store...")
        documents = []
        ids = []
        
        for i, row in df.iterrows():
            document = Document(
                page_content=row["Title"] + " " + row["Review"],
                metadata={"rating": row["Rating"], "date": row["Date"]},
                id=str(i)
            )
            ids.append(str(i))
            documents.append(document)
        
        # Adding in batches for better performance
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            print(f"   Added {min(i+batch_size, len(documents))}/{len(documents)}")
        
        print("Vector store created successfully")
    else:
        print(f"Loaded existing vector store with {len(df)} documents")
    
    return vector_store

# Initialize once at module load
vector_store = initialize_vector_store()

# Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

def query_vector_store(query, k=5):
    """Query the vector store and return results"""
    results = retriever.invoke(query)
    return results

# For easy testing
if __name__ == "__main__":
    # Test query
    test_query = "What do people say about the food?"
    print(f"\nTest query: {test_query}")
    results = query_vector_store(test_query)
    print(f"Retrieved {len(results)} documents")
    for i, doc in enumerate(results[:2], 1):
        print(f"\nResult {i}:")
        print(f"  Content: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")