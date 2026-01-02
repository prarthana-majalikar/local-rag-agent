# Performance Benchmarks

## Test Configuration
- **Dataset Size**: 369 restaurant reviews
- **Vector Store**: Chroma with Ollama embeddings (mxbai-embed-large)
- **LLM**: Llama 3.2 (local via Ollama)
- **Retrieval Count**: Top 5 similar reviews
- **Test Queries**: 5

## Performance Results

### Retrieval Performance
- **Average Latency**: 111ms
- **Median**: 101ms
- **95th Percentile**: 153ms

### LLM Generation
- **Average**: 21676ms
- **95th Percentile**: 28327ms

### End-to-End Response Time
- **Average**: 21787ms (21.8s)
- **Median**: 20827ms (20.8s)
- **95th Percentile**: 28479ms (28.5s)

## Test Queries
1. What do people say about the food quality?
2. Are there any complaints about service?
3. Which restaurant has the best atmosphere?
4. Tell me about pricing
5. What are the common positive reviews?
