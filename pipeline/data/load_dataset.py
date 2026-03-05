import json
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

def chunk_texts(tokenizer, texts, chunk_len):
    chunks = []
    for text in tqdm(texts, desc="Chunking corpus"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        for i in range(0, len(ids), chunk_len):
            sub = ids[i:i+chunk_len]
            if not sub:
                continue
            chunk_text = tokenizer.decode(sub, skip_special_tokens=True)
            chunks.append(chunk_text)
    return chunks


def embed_texts(model, texts, batch_size=256):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding corpus"):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embs.append(embs)
    return np.vstack(all_embs)


def build_faiss_index(embs):
    d = embs.shape[1]
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    return index


def retrieve_for_query(model, index, chunks, question, top_k=1000):
    q_emb = model.encode([question], show_progress_bar=False, convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    I = I[0]
    return [chunks[i] for i in I]

def load_hotpot_train():
    ds = load_dataset("hotpot_qa", "fullwiki", split="train")
    return ds


def build_corpus_texts_hotpot(ds, max_corpus_docs):
    texts = []
    limit = min(max_corpus_docs, len(ds))
    for ex in tqdm(ds.select(range(limit)), desc="Collecting Hotpot corpus texts"):
        for sent_list in ex["context"]["sentences"]:
            paragraph = " ".join(sent_list).strip()
            if paragraph:
                texts.append(paragraph)
    return texts


def load_musique_train():
    ds = load_dataset("dgslibisey/MuSiQue", split="train")
    return ds


def build_corpus_texts_musique(ds, max_corpus_docs):
    texts = []
    limit = min(max_corpus_docs, len(ds))
    for ex in tqdm(ds.select(range(limit)), desc="Collecting MuSiQue corpus texts"):
        for p in ex["paragraphs"]:
            texts.append(p["paragraph_text"])
    return texts

def load_2wiki_train():
    ds = load_dataset("xanhho/2WikiMultihopQA", split="train", trust_remote_code=True)
    return ds


def build_corpus_texts_2wiki(ds, max_corpus_docs):
    texts = []
    limit = min(max_corpus_docs, len(ds))
    for ex in tqdm(ds.select(range(limit)), desc="Collecting 2Wiki corpus texts"):
        for sent_list in ex["context"]["content"]:
            paragraph = " ".join(sent_list).strip()
            if paragraph:
                texts.append(paragraph)
    return texts

def build_dataset(args):
    """
    dataset_name: "hotpot", "musique", or "2wiki"
    """
    dataset_name = args.dataset
    print(f"Building dataset for: {dataset_name}")

    # load dataset and corpus texts
    if dataset_name == "hotpot":
        ds = load_hotpot_train()
        corpus_texts = build_corpus_texts_hotpot(ds, args.max_corpus_docs)
    elif dataset_name == "musique":
        ds = load_musique_train()
        corpus_texts = build_corpus_texts_musique(ds, args.max_corpus_docs)
    elif dataset_name == "2wiki":
        ds = load_2wiki_train()
        corpus_texts = build_corpus_texts_2wiki(ds, args.max_corpus_docs)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    output_json = args.output_json

    print("Total train examples:", len(ds))
    print("Corpus docs:", len(corpus_texts))

    # tokenizer and chunks
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    print(f"Chunking corpus into {args.chunk_token_len}-token chunks...")
    corpus_chunks = chunk_texts(tokenizer, corpus_texts, args.chunk_token_len)
    print("Total chunks before dedup:", len(corpus_chunks))
    corpus_chunks = list(dict.fromkeys(corpus_chunks))
    print("Total chunks after dedup:", len(corpus_chunks))

    # embeddings and index
    print("Loading embedding model...")
    emb_model = SentenceTransformer(args.embed_model)

    print("Embedding corpus chunks...")
    corpus_embs = embed_texts(emb_model, corpus_chunks)
    print("Embeddings shape:", corpus_embs.shape)

    print("Building FAISS index...")
    index = build_faiss_index(corpus_embs)

    # per-query retrieval
    results = []
    num_built = 0
    print("Processing queries...")

    for ex in tqdm(ds, desc="Per-query retrieval"):
        if num_built >= args.max_queries:
            break

        question = ex["question"]
        answer = ex.get("answer", "")

        top_k = min(len(corpus_chunks), max(1000, args.max_chunks_per_query * 10))
        retrieved = retrieve_for_query(
            emb_model, index, corpus_chunks, question, top_k=top_k
        )
        retrieved = list(dict.fromkeys(retrieved))

        if len(retrieved) < args.min_chunks_per_query:
            continue

        retrieved = retrieved[:args.max_chunks_per_query]

        qid = ex.get("id") or ex.get("question_id")

        results.append({
            "id": qid,
            "dataset": dataset_name,
            "question": question,
            "answer": answer,
            "chunks": retrieved,
        })
        num_built += 1

    print("Built queries:", num_built)
    print("Saving to", output_json)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Build RAG dataset from a source dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["hotpot", "musique", "2wiki"],
        help="Dataset to build.",
    )
    parser.add_argument("--model-name", type=str, required=True, help="Tokenizer model name.")
    parser.add_argument("--embed-model", type=str, required=True, help="Embedding model name.")
    parser.add_argument("--chunk-token-len", type=int, required=True, help="Chunk length in tokens.")
    parser.add_argument("--max-corpus-docs", type=int, required=True, help="Max corpus docs.")
    parser.add_argument("--max-queries", type=int, required=True, help="Max queries to build.")
    parser.add_argument(
        "--min-chunks-per-query",
        type=int,
        required=True,
        help="Minimum retrieved chunks required per query.",
    )
    parser.add_argument(
        "--max-chunks-per-query",
        type=int,
        required=True,
        help="Maximum chunks to keep per query.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Output JSON path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_dataset(args)

if __name__ == "__main__":
    main()
