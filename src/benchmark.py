import time
import numpy as np
import pandas as pd
from vector import query_vector_store, vector_store
from main import generate_response

def benchmark_system():
    print("\nRunning comprehensive benchmarks...")
    print("="*60)
    
    # Load dataset info
    df = pd.read_csv('realistic_restaurant_reviews.csv')
    
    queries = [
        "What do people say about the food quality?",
        "Are there any complaints about service?",
        "Which restaurant has the best atmosphere?",
        "Tell me about pricing",
        "What are the common positive reviews?"
    ]
    
    # Warm-up query (first query is always slower)
    print("\n1. Warming up system...")
    _ = query_vector_store(queries[0])
    print("   Warm-up complete")
    
    # Benchmark retrieval only
    print("\n2. Benchmarking retrieval performance...")
    retrieval_times = []
    
    for i, query in enumerate(queries, 1):
        start = time.time()
        results = query_vector_store(query)
        retrieval_time = (time.time() - start) * 1000
        retrieval_times.append(retrieval_time)
        print(f"   Query {i}/5: {retrieval_time:.2f}ms")
    
    # Benchmark end-to-end (retrieval + LLM)
    print("\n3. Benchmarking end-to-end performance (with LLM)...")
    end_to_end_times = []
    
    for i, query in enumerate(queries, 1):
        start = time.time()
        result, _ = generate_response(query)
        total_time = (time.time() - start) * 1000
        end_to_end_times.append(total_time)
        print(f"   Query {i}/5: {total_time:.2f}ms ({len(result)} chars)")
    
    # Calculate LLM times
    llm_times = [e2e - ret for e2e, ret in zip(end_to_end_times, retrieval_times)]
    
    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    print(f"\nRetrieval Performance:")
    print(f"   Average:        {np.mean(retrieval_times):.2f}ms")
    print(f"   Median:         {np.median(retrieval_times):.2f}ms")
    print(f"   95th percentile: {np.percentile(retrieval_times, 95):.2f}ms")
    print(f"   Min/Max:        {np.min(retrieval_times):.2f}ms / {np.max(retrieval_times):.2f}ms")
    
    print(f"\nLLM Generation:")
    print(f"   Average:        {np.mean(llm_times):.2f}ms")
    print(f"   Median:         {np.median(llm_times):.2f}ms")
    print(f"   95th percentile: {np.percentile(llm_times, 95):.2f}ms")
    
    print(f"\nEnd-to-End Response:")
    print(f"   Average:        {np.mean(end_to_end_times):.2f}ms ({np.mean(end_to_end_times)/1000:.2f}s)")
    print(f"   Median:         {np.median(end_to_end_times):.2f}ms")
    print(f"   95th percentile: {np.percentile(end_to_end_times, 95):.2f}ms")
    
    print(f"\nDataset Information:")
    print(f"   Total reviews:   {len(df)}")
    print(f"   Avg review length: {df['Review'].str.len().mean():.0f} characters")
    if 'Rating' in df.columns:
        print(f"   Avg rating:      {df['Rating'].mean():.2f}/5.0")
    print(f"   Retrieval count: 5 reviews per query")
    
    # Save to markdown
    with open('BENCHMARKS.md', 'w') as f:
        f.write("# Performance Benchmarks\n\n")
        f.write("## Test Configuration\n")
        f.write(f"- **Dataset Size**: {len(df)} restaurant reviews\n")
        f.write(f"- **Vector Store**: Chroma with Ollama embeddings (mxbai-embed-large)\n")
        f.write(f"- **LLM**: Llama 3.2 (local via Ollama)\n")
        f.write(f"- **Retrieval Count**: Top 5 similar reviews\n")
        f.write(f"- **Test Queries**: {len(queries)}\n\n")
        
        f.write("## Performance Results\n\n")
        f.write("### Retrieval Performance\n")
        f.write(f"- **Average Latency**: {np.mean(retrieval_times):.0f}ms\n")
        f.write(f"- **Median**: {np.median(retrieval_times):.0f}ms\n")
        f.write(f"- **95th Percentile**: {np.percentile(retrieval_times, 95):.0f}ms\n\n")
        
        f.write("### LLM Generation\n")
        f.write(f"- **Average**: {np.mean(llm_times):.0f}ms\n")
        f.write(f"- **95th Percentile**: {np.percentile(llm_times, 95):.0f}ms\n\n")
        
        f.write("### End-to-End Response Time\n")
        f.write(f"- **Average**: {np.mean(end_to_end_times):.0f}ms ({np.mean(end_to_end_times)/1000:.1f}s)\n")
        f.write(f"- **Median**: {np.median(end_to_end_times):.0f}ms ({np.median(end_to_end_times)/1000:.1f}s)\n")
        f.write(f"- **95th Percentile**: {np.percentile(end_to_end_times, 95):.0f}ms ({np.percentile(end_to_end_times, 95)/1000:.1f}s)\n\n")
        
        f.write("## Test Queries\n")
        for i, q in enumerate(queries, 1):
            f.write(f"{i}. {q}\n")
    
    print("\nResults saved to BENCHMARKS.md")
    print("="*60)

if __name__ == "__main__":
    benchmark_system()