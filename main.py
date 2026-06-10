import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def get_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def rerank(question, chunks):
    pairs = [[question, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in ranked], [score for chunk, score in ranked]

# Chroma
chroma_client = chromadb.PersistentClient(path="./chroma_company")
collection = chroma_client.get_or_create_collection(name="company_kb")

existing = collection.count()
if existing > 0:
    collection.delete(ids=collection.get()["ids"])

splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)

# 5 документов
documents = [
    {"file": "shop_faq.txt",        "source": "FAQ",         "category": "general"},
    {"file": "delivery_details.txt","source": "Доставка",    "category": "delivery"},
    {"file": "returns_policy.txt",  "source": "Возврат",     "category": "returns"},
    {"file": "wholesale.txt",       "source": "Опт",         "category": "wholesale"},
    {"file": "contacts.txt",        "source": "Контакты",    "category": "contacts"},
]

all_chunks = []
all_embeddings = []
all_ids = []
all_metadatas = []
counter = 0

print("Загружаем документы...")
for doc in documents:
    with open(doc["file"], "r", encoding="utf-8") as f:
        text = f.read()
    chunks = splitter.split_text(text)
    print(f"  {doc['file']} → {len(chunks)} chunks")

    for chunk in chunks:
        all_chunks.append(chunk)
        all_embeddings.append(get_embedding(chunk))
        all_ids.append(f"chunk_{counter}")
        all_metadatas.append({
            "source": doc["source"],
            "category": doc["category"],
            "file": doc["file"],
        })
        counter += 1

collection.add(
    documents=all_chunks,
    embeddings=all_embeddings,
    ids=all_ids,
    metadatas=all_metadatas
)
print(f"\nВсего в базе: {collection.count()} chunks\n")

# RAG функция
def ask(question):
    question_vector = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=6,
        include=["documents", "distances", "metadatas"]
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Reranking
    reranked_chunks, scores = rerank(question, chunks)

    # Берём топ-3 после reranking
    top_chunks = reranked_chunks[:3]
    context = "\n\n".join(top_chunks)

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """Ты ассистент магазина одежды BrandUA.
Отвечай ТОЛЬКО на основе предоставленного контекста.
Отвечай на том языке на котором задан вопрос.
Если ответа нет в контексте — пиши 'уточню у менеджера'."""},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"}
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content

    # Находим источники для топ чанков
    sources = []
    for top_chunk in top_chunks:
        for i, chunk in enumerate(chunks):
            if chunk == top_chunk:
                sources.append(metadatas[i]["source"])
                break

    return answer, sources, scores[:3]

# Тест
test_questions = [
    "сколько стоит доставка?",
    "как вернуть бракованный товар?",
    "какой email для оптовых заказов?",
    "есть ли доставка в день заказа?",
    "какая скидка для оптовых покупателей?",
    "как связаться с магазином?",
]

print("=" * 50)
for question in test_questions:
    answer, sources, scores = ask(question)
    print(f"\nВопрос: {question}")
    print(f"Ответ: {answer}")
    print(f"Источники: {', '.join(set(sources))}")
    print(f"Scores: {[f'{s:.2f}' for s in scores]}")
    print("-" * 50)
