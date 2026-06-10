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

<img width="990" height="217" alt="Снимок экрана 2026-06-10 в 11 46 25" src="https://github.com/user-attachments/assets/b350b764-4a80-46ef-9065-94cf03dcbe04" />

<img width="980" height="174" alt="Снимок экрана 2026-06-10 в 11 47 01" src="https://github.com/user-attachments/assets/2b795c29-3527-4d70-87ab-d517126a3935" />

<img width="1044" height="180" alt="Снимок экрана 2026-06-10 в 11 46 42" src="https://github.com/user-attachments/assets/bea32d22-1336-4ce7-af3e-1a0bcc1ab452" />



