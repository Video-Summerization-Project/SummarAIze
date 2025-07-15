from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
import json
import chromadb
from chromadb.config import Settings

def read_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    segments = data.get("segments", [])
    for s in segments:
        s["start_time"] = s.pop("start", None)
        s["end_time"] = s.pop("end", None)

    return segments


def split_text_into_chunks(segments):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n", "۔", "؟", "!", " ", ""]
    )

    chunks = []
    for segment in segments:
        start_time = segment.get("start_time") or segment.get("start")
        end_time = segment.get("end_time") or segment.get("end")

        if not segment.get("text", "").strip():
            continue

        text_chunks = splitter.split_text(segment["text"])
        for chunk in text_chunks:
            chunks.append({
                "text": chunk,
                "start_time": start_time,
                "end_time": end_time
            })

    return chunks

def initialize_db(embedding_model, chunks: list[dict]):

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{
        "start_time": str(chunk.get("start_time", "unknown")),
        "end_time": str(chunk.get("end_time", "unknown"))
    } for chunk in chunks]
    
    embeddings = embedding_model.encode(texts).tolist()

    client = chromadb.PersistentClient(path="tmp/search/vector_db")
    try:
        client.delete_collection("tts_collection1")
    except:
        pass

    collection = client.get_or_create_collection(name="tts_collection1")

    collection.add(
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[{
            "index": i,
            "start_time": str(chunk.get("start_time", "unknown")),
            "end_time": str(chunk.get("end_time", "unknown"))
        } for i, chunk in enumerate(chunks)],
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings
    )

    return collection


def retrieve_with_surrounding_chunks(chunks, query, embedding_model, before=10, after=20):

    client = chromadb.PersistentClient(path="tmp/search/vector_db")
    collection = client.get_or_create_collection(name="tts_collection1")

    query_embedding = embedding_model.encode([query]).tolist()

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=1,
        include=["metadatas"]
    )

    anchor_idx = int(result["metadatas"][0][0]["index"])

    start = max(anchor_idx - before, 0)
    end = min(anchor_idx + after + 1, len(chunks))

    return chunks[start:end]
