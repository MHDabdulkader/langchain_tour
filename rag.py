import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
import time
import hashlib
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core.utils.pydantic")

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

print("[INIT] Importing libraries...")
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openrouter import ChatOpenRouter

# 1> Embedder
print("[INIT] Loading embedding model...")
t = time.time()
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
print(f"[INIT] Embedding model ready ({time.time() - t:.1f}s)")

# 2> Generator
print("[INIT] Initializing OpenRouter model...")
model = ChatOpenRouter(
    model="nvidia/nemotron-3.5-super-vl:free",
    max_tokens=2048,
    temperature=0.3,
)

# 3> Load + chunk
print("[LOAD] Reading notes.txt...")
text = open("notes.txt", encoding="utf-8").read()
print(f"[LOAD] {len(text)} characters")

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_text(text)
print(f"[CHUNK] {len(chunks)} chunks")
for i, c in enumerate(chunks):
    print(f"[CHUNK] #{i}: {len(c)} chars | {c[:60]!r}...")

# 4> Index
print("[INDEX] Opening Chroma store at ./chroma_db...")
store = Chroma(
    collection_name="notes",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
before = store._collection.count()
print(f"[INDEX] Chunks already stored: {before}")

# Same chunk -> same id -> upserted, no duplicates
ids = [hashlib.md5(c.encode("utf-8")).hexdigest() for c in chunks]
t = time.time()
store.add_texts(chunks, ids=ids)
after = store._collection.count()
print(f"[INDEX] Stored now: {after} (+{after - before} new) in {time.time() - t:.1f}s")


# 5> Ask
def ask(question: str) -> str:
    print("\n" + "=" * 50)
    print(f"[ASK] {question}")
    print("=" * 50)

    # Chroma score = distance (lower = closer)
    t = time.time()
    results = store.similarity_search_with_score(question, k=3)
    print(f"[RETRIEVE] {len(results)} chunks in {time.time() - t:.2f}s")
    for rank, (doc, score) in enumerate(results, 1):
        print(f"[RETRIEVE] #{rank} score={score:.4f} | {doc.page_content[:80]!r}...")

    context = "\n\n".join(doc.page_content for doc, _ in results)
    print(f"[PROMPT] Context size: {len(context)} chars")

    prompt = (
        "Answer using only the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    print("[LLM] Sending to model...")
    t = time.time()
    res = model.invoke(prompt)
    print(f"[LLM] Response received in {time.time() - t:.1f}s")

    answer = res.content or res.additional_kwargs.get("reasoning_content", "")
    if not answer:
        print("[LLM] Empty response")
    return answer


if __name__ == "__main__":
    print("\n[READY] RAG assistant running.")
    while True:
        q = input("\nQuestion ('exit' to quit): ").strip()

        if q.lower() in ("exit", "quit"):
            print("[EXIT] Goodbye!")
            break
        if not q:
            print("[WARNING] Please enter a question.")
            continue

        try:
            print("\n[ANSWER]\n" + ask(q))
        except Exception as error:
            print(f"[ERROR] {error}")