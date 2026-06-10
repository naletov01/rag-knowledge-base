# RAG Knowledge Base Agent

AI agent that answers questions based on a company's knowledge base.
Uses semantic search and reranking to provide accurate answers.

## How It Works

1. Documents are split into chunks
2. Each chunk is converted into a vector using OpenAI Embeddings
3. Vectors are stored in Chroma (local vector database)
4. When a question is asked, similar chunks are retrieved using semantic search
5. Reranking reorders the results by relevance
6. GPT generates an answer based on the retrieved chunks

## Stack

- Python
- OpenAI API (embeddings + GPT-4o-mini)
- ChromaDB (vector database)
- LangChain (chunking)
- CrossEncoder (reranking)

## Example

Question: How much does delivery cost?

Answer: Delivery costs 80 UAH. Orders over 1500 UAH qualify for free delivery.
Door-to-door courier delivery costs an additional 50 UAH.

Sources: Delivery, FAQ

---

Question: How can I return a defective product?

Answer: For defective products, return shipping is covered by the store.
Contact us at return@brandua.com or send us a direct message.

Sources: Returns

## Installation

bash pip install openai chromadb langchain-text-splitters sentence-transformers python-dotenv 

Create a .env file:

env OPENAI_API_KEY=your-api-key 

Run:

bash python day26miniRAG.py

## Architecture

Documents (TXT/PDF)
↓

Chunking (150 characters, overlap 30)
↓

Embeddings (text-embedding-3-small)
↓

ChromaDB (vector storage)
↓

Semantic Search (top 6 chunks)
↓

Reranking (CrossEncoder)
↓

GPT-4o-mini (answer generation)
↓

Answer + Sources

