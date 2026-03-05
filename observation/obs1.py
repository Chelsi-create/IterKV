import json
import argparse
import re
import hashlib
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoConfig
from vllm import LLM, SamplingParams
from sklearn.feature_extraction.text import TfidfVectorizer


def run_iterative_rag_round(
    llm,
    question,
    chunks_this_round,
    prev_reasoning_text,
    max_gen_tokens,
    round_idx,
    num_rounds,
):
    """
    One RAG round:
      - Build prompt from question + all chunks_this_round + prev_reasoning_text.
      - Ask the model for the next reasoning step (not a full repeated answer).
      - Return: (reasoning_text_r, prompt_token_ids, generated_token_ids, prompt_text)
    """
    chunk_block = "\n\n".join(
        f"[Chunk {i + 1}]\n{c}" for i, c in enumerate(chunks_this_round)
    )

    prev_reasoning_block = prev_reasoning_text.strip() if prev_reasoning_text else "None"

    if round_idx < num_rounds:
        instruction = (
            "Instructions:\n"
            "1. Carefully read the question and the retrieved evidence.\n"
            "2. Consider the previous reasoning, but do NOT repeat it.\n"
            "3. Write ONLY the next short reasoning step in 1-3 sentences.\n"
            "4. Do NOT repeat the question.\n"
            "5. Do NOT give the final answer yet.\n\n"
            "Next reasoning step:\n"
        )
    else:
        instruction = (
            "Instructions:\n"
            "1. Use the retrieved evidence and the previous reasoning.\n"
            "2. Write ONLY the final answer in one short sentence.\n"
            "3. Do NOT repeat the question.\n"
            "4. Do NOT show your reasoning.\n"
            "5. If the answer cannot be determined from the evidence, say exactly:\n"
            "   \"Unknown based on the given evidence.\"\n\n"
            "Final answer:\n"
        )

    prompt = (
        "You are an expert reasoning assistant helping to answer multi-hop questions "
        "using retrieved evidence.\n\n"
        f"Question:\n{question}\n\n"
        "Retrieved evidence (each item is a separate chunk of text):\n"
        f"{chunk_block}\n\n"
        "Previous reasoning (may be empty):\n"
        f"{prev_reasoning_block}\n\n"
        f"{instruction}"
    )

    sampling_params = SamplingParams(
        max_tokens=max_gen_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    def _looks_placeholder(text):
        t = (text or "").strip()
        if not t:
            return True
        if "please fill in" in t.lower():
            return True
        if re.fullmatch(r"[-_=.\s|]+", t):
            return True
        if len(t) >= 10 and sum(ch.isalnum() for ch in t) / max(1, len(t)) < 0.15:
            return True
        return False

    def _extract_prefill_decode(req_out, n_tokens, wall_s):
        metrics = getattr(req_out, "metrics", None)
        if metrics is not None:
            arrival_time = getattr(metrics, "arrival_time", None)
            first_token_time = getattr(metrics, "first_token_time", None)
            finished_time = getattr(metrics, "finished_time", None)
            if (
                isinstance(arrival_time, (int, float))
                and isinstance(first_token_time, (int, float))
                and isinstance(finished_time, (int, float))
                and finished_time >= first_token_time >= arrival_time
            ):
                return (
                    float(max(0.0, first_token_time - arrival_time)),
                    float(max(0.0, finished_time - first_token_time)),
                )
        # Fallback split if low-level timing is unavailable.
        if n_tokens <= 1:
            return float(wall_s), 0.0
        est_tpot = wall_s / float(max(1, n_tokens))
        decode_s = est_tpot * float(max(0, n_tokens - 1))
        prefill_s = max(0.0, wall_s - decode_s)
        return float(prefill_s), float(decode_s)

    def _generate_once(prompt_text):
        t0 = time.perf_counter()
        outputs = llm.generate([prompt_text], sampling_params=sampling_params, use_tqdm=False)
        wall_s = float(max(0.0, time.perf_counter() - t0))
        req_out = outputs[0]
        out = req_out.outputs[0]
        text = out.text.strip()
        token_ids = list(out.token_ids)
        prefill_s, decode_s = _extract_prefill_decode(req_out, len(token_ids), wall_s)
        prompt_token_ids = getattr(req_out, "prompt_token_ids", None)
        if prompt_token_ids is None:
            tokenizer = llm.get_tokenizer()
            prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        return text, list(prompt_token_ids), token_ids, prefill_s, decode_s, wall_s

    reasoning_text_r, prompt_token_ids, generated_token_ids, prefill_s, decode_s, wall_s = _generate_once(prompt)
    total_prefill_s = prefill_s
    total_decode_s = decode_s
    total_generation_s = wall_s
    if _looks_placeholder(reasoning_text_r):
        retry_prompt = (
            prompt
            + "\nIMPORTANT: Your last response looked like a placeholder/template.\n"
              "Respond with concrete content only.\n"
              "Do not use dashes, underscores, or placeholder text.\n"
        )
        (
            retry_text,
            retry_prompt_token_ids,
            retry_token_ids,
            retry_prefill_s,
            retry_decode_s,
            retry_wall_s,
        ) = _generate_once(retry_prompt)
        total_prefill_s += retry_prefill_s
        total_decode_s += retry_decode_s
        total_generation_s += retry_wall_s
        if not _looks_placeholder(retry_text):
            reasoning_text_r = retry_text
            prompt_token_ids = retry_prompt_token_ids
            generated_token_ids = retry_token_ids
            prompt = retry_prompt

    timing = {
        "prefill_s": float(total_prefill_s),
        "decode_s": float(total_decode_s),
        "generation_total_s": float(total_generation_s),
    }
    return reasoning_text_r, prompt_token_ids, generated_token_ids, prompt, timing


def compute_kv_memory_bytes(num_tokens, num_layers, num_kv_heads, head_dim, bytes_per_elem=2):
    if num_tokens <= 0:
        return 0.0
    if num_layers <= 0 or num_kv_heads <= 0 or head_dim <= 0 or bytes_per_elem <= 0:
        raise ValueError("All model/config values must be positive.")

    kv_bytes = (2 * num_layers * num_kv_heads * head_dim * bytes_per_elem * num_tokens)
    return kv_bytes


def build_tfidf_index(chunks):
    if not chunks:
        return None, None
    vectorizer = TfidfVectorizer(stop_words="english")
    chunk_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, chunk_matrix


def retrieve_new_chunks_tfidf(
    question,
    prev_reasoning_text,
    chunks,
    vectorizer,
    chunk_matrix,
    top_k,
):
    if not chunks or vectorizer is None or chunk_matrix is None or top_k <= 0:
        return []

    query_text = question if not prev_reasoning_text else f"{question}\n{prev_reasoning_text}"
    q = vectorizer.transform([query_text])  # shape: [1, vocab]
    scores = (chunk_matrix @ q.T).toarray().ravel()  # cosine-like for tf-idf

    ranked = np.argsort(-scores)
    picked = []
    for idx in ranked:
        idx = int(idx)
        picked.append(chunks[idx])
        if len(picked) >= top_k:
            break
    return picked


def truncate_text(text, max_chars):
    if text is None:
        return ""
    if max_chars is None or max_chars < 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated to {max_chars} chars]"


def run_experiment1_one_query(
    llm,
    tokenizer,
    question,
    all_chunks_per_query,
    num_rounds,
    model_config,
    chunks_per_round,
    max_gen_tokens,
    trace_max_chars,
):
    num_layers = model_config["num_layers"]
    num_kv_heads = model_config["num_kv_heads"]
    head_dim = model_config["head_dim"]
    bytes_per_elem = model_config.get("bytes_per_elem", 2)

    # Question tokens
    question_token_ids = tokenizer.encode(question, add_special_tokens=False)
    len_question_tokens = len(question_token_ids)

    # KV and GPU arrays
    kv_q = np.zeros(num_rounds + 1, dtype=float)
    kv_chunks = np.zeros(num_rounds + 1, dtype=float)
    kv_reason = np.zeros(num_rounds + 1, dtype=float)
    gpu_total = np.zeros(num_rounds + 1, dtype=float)
    retrieval_s = np.zeros(num_rounds + 1, dtype=float)
    prefill_s = np.zeros(num_rounds + 1, dtype=float)
    decode_s = np.zeros(num_rounds + 1, dtype=float)
    overlap_prev = np.zeros(num_rounds + 1, dtype=float)
    overlap_round1 = np.zeros(num_rounds + 1, dtype=float)

    # Round 0: only question
    kv_q[0] = compute_kv_memory_bytes(
        len_question_tokens, num_layers, num_kv_heads, head_dim, bytes_per_elem
    )
    kv_chunks[0] = 0.0
    kv_reason[0] = 0.0
    gpu_total[0] = get_gpu_used_bytes()

    # State across rounds
    total_reasoning_tokens_accum = 0
    prev_reasoning_text = ""          # we keep only the latest reasoning step
    accumulated_chunks = []           # all chunks used so far
    vectorizer, chunk_matrix = build_tfidf_index(all_chunks_per_query)
    prev_round_chunk_ids = set()
    round1_chunk_ids = set()

    per_query_trace = {
        "question": truncate_text(question, trace_max_chars),
        "rounds": [],
    }
    reasoning_text_r = ""

    for r in range(1, num_rounds + 1):
        # 1) Select new chunks (no repeats)
        retrieval_t0 = time.perf_counter()
        new_chunks = retrieve_new_chunks_tfidf(
            question=question,
            prev_reasoning_text=prev_reasoning_text,
            chunks=all_chunks_per_query,
            vectorizer=vectorizer,
            chunk_matrix=chunk_matrix,
            top_k=chunks_per_round,
        )
        retrieval_s[r] = float(max(0.0, time.perf_counter() - retrieval_t0))

        current_round_chunk_ids = {
            hashlib.sha1(chunk.encode("utf-8")).hexdigest() for chunk in new_chunks
        }
        if r == 1:
            round1_chunk_ids = set(current_round_chunk_ids)
        else:
            overlap_prev[r] = (
                len(current_round_chunk_ids & prev_round_chunk_ids) / max(1, len(current_round_chunk_ids))
            )
            overlap_round1[r] = (
                len(current_round_chunk_ids & round1_chunk_ids) / max(1, len(current_round_chunk_ids))
            )
        prev_round_chunk_ids = set(current_round_chunk_ids)

        accumulated_chunks.extend(new_chunks)

        # Count total chunk tokens so far
        total_chunk_tokens_accum = sum(
            len(tokenizer.encode(chunk, add_special_tokens=False))
            for chunk in accumulated_chunks
        )

        # 2) One RAG round
        reasoning_text_r, _prompt_token_ids, gen_token_ids, prompt_text, round_timing = run_iterative_rag_round(
            llm=llm,
            question=question,
            chunks_this_round=accumulated_chunks,
            prev_reasoning_text=prev_reasoning_text,
            max_gen_tokens=max_gen_tokens,
            round_idx=r,
            num_rounds=num_rounds,
        )
        prefill_s[r] = round_timing["prefill_s"]
        decode_s[r] = round_timing["decode_s"]

        per_query_trace["rounds"].append(
            {
                "round": r,
                "new_chunks_this_round": len(new_chunks),
                "accumulated_chunks": len(accumulated_chunks),
                "prompt_text": truncate_text(prompt_text, trace_max_chars),
                "output_text": truncate_text(reasoning_text_r, trace_max_chars),
                "generated_tokens": int(len(gen_token_ids)),
                "retrieval_s": float(retrieval_s[r]),
                "prefill_s": float(prefill_s[r]),
                "decode_s": float(decode_s[r]),
                "generation_total_s": float(round_timing["generation_total_s"]),
                "overlap_with_prev_round_pct": float(overlap_prev[r] * 100.0),
                "overlap_with_round1_pct": float(overlap_round1[r] * 100.0),
            }
        )

        # 3) Update reasoning tokens / text
        total_reasoning_tokens_accum += len(gen_token_ids)
        prev_reasoning_text = reasoning_text_r.strip()  # do NOT append old text again

        # 4) KV per bucket
        kv_q[r] = compute_kv_memory_bytes(
            len_question_tokens, num_layers, num_kv_heads, head_dim, bytes_per_elem
        )
        kv_chunks[r] = compute_kv_memory_bytes(
            total_chunk_tokens_accum, num_layers, num_kv_heads, head_dim, bytes_per_elem
        )
        kv_reason[r] = compute_kv_memory_bytes(
            total_reasoning_tokens_accum,
            num_layers,
            num_kv_heads,
            head_dim,
            bytes_per_elem,
        )
        gpu_total[r] = get_gpu_used_bytes()

    per_query_trace["final_output"] = truncate_text(reasoning_text_r, trace_max_chars)
    return (
        kv_q,
        kv_chunks,
        kv_reason,
        gpu_total,
        retrieval_s,
        prefill_s,
        decode_s,
        overlap_prev,
        overlap_round1,
        per_query_trace,
    )



def aggregate_experiment1_over_dataset(llm, tokenizer, dataset, model_config,
                                       num_rounds, chunks_per_round, max_gen_tokens,
                                       trace_max_chars):
    all_kv_q = []
    all_kv_chunks = []
    all_kv_reason = []
    all_gpu_total = []
    all_retrieval_s = []
    all_prefill_s = []
    all_decode_s = []
    all_overlap_prev = []
    all_overlap_round1 = []
    traces = []

    for ex in dataset:
        question = ex["question"]
        chunks = ex["chunks"]

        (
            kv_q,
            kv_chunks,
            kv_reason,
            gpu_total,
            retrieval_s,
            prefill_s,
            decode_s,
            overlap_prev,
            overlap_round1,
            query_trace,
        ) = run_experiment1_one_query(
            llm=llm,
            tokenizer=tokenizer,
            question=question,
            all_chunks_per_query=chunks,
            num_rounds=num_rounds,
            model_config=model_config,
            chunks_per_round=chunks_per_round,
            max_gen_tokens=max_gen_tokens,
            trace_max_chars=trace_max_chars,
        )

        all_kv_q.append(kv_q)
        all_kv_chunks.append(kv_chunks)
        all_kv_reason.append(kv_reason)
        all_gpu_total.append(gpu_total)
        all_retrieval_s.append(retrieval_s)
        all_prefill_s.append(prefill_s)
        all_decode_s.append(decode_s)
        all_overlap_prev.append(overlap_prev)
        all_overlap_round1.append(overlap_round1)
        query_trace["id"] = ex.get("id")
        traces.append(query_trace)

    all_kv_q = np.stack(all_kv_q, axis=0)
    all_kv_chunks = np.stack(all_kv_chunks, axis=0)
    all_kv_reason = np.stack(all_kv_reason, axis=0)
    all_gpu_total = np.stack(all_gpu_total, axis=0)
    all_retrieval_s = np.stack(all_retrieval_s, axis=0)
    all_prefill_s = np.stack(all_prefill_s, axis=0)
    all_decode_s = np.stack(all_decode_s, axis=0)
    all_overlap_prev = np.stack(all_overlap_prev, axis=0)
    all_overlap_round1 = np.stack(all_overlap_round1, axis=0)

    mean_kv_q = np.mean(all_kv_q, axis=0)
    mean_kv_chunks = np.mean(all_kv_chunks, axis=0)
    mean_kv_reason = np.mean(all_kv_reason, axis=0)
    mean_gpu_total = np.mean(all_gpu_total, axis=0)
    mean_retrieval_s = np.mean(all_retrieval_s, axis=0)
    mean_prefill_s = np.mean(all_prefill_s, axis=0)
    mean_decode_s = np.mean(all_decode_s, axis=0)
    mean_overlap_prev = np.mean(all_overlap_prev, axis=0)
    mean_overlap_round1 = np.mean(all_overlap_round1, axis=0)

    return (
        mean_kv_q,
        mean_kv_chunks,
        mean_kv_reason,
        mean_gpu_total,
        mean_retrieval_s,
        mean_prefill_s,
        mean_decode_s,
        mean_overlap_prev,
        mean_overlap_round1,
        traces,
    )


def plot_experiment2_overlap(round_indices, dataset_to_overlap_prev, dataset_to_overlap_round1, output_path):
    rounds = np.asarray(round_indices)
    # Overlap is meaningful from round 2 onward.
    valid_mask = rounds >= 2
    x = rounds[valid_mask]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    ax_prev, ax_r1 = axes

    for label, arr in dataset_to_overlap_prev.items():
        y = np.asarray(arr, dtype=float)[valid_mask] * 100.0
        ax_prev.plot(x, y, marker="o", linewidth=1.6, markersize=4, label=label)

    for label, arr in dataset_to_overlap_round1.items():
        y = np.asarray(arr, dtype=float)[valid_mask] * 100.0
        ax_r1.plot(x, y, marker="o", linewidth=1.6, markersize=4, label=label)

    ax_prev.set_title("Overlap vs Previous Round", fontsize=9, fontweight="bold")
    ax_r1.set_title("Overlap vs Round 1", fontsize=9, fontweight="bold")
    for ax in axes:
        ax.set_xticks(x, [f"R{i}" for i in x])
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_ylim(0, 100)
    ax_prev.set_ylabel("Chunk Overlap (%)", fontsize=9, fontweight="bold")

    handles, labels = ax_prev.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    plt.tight_layout(rect=[0.02, 0.02, 1.0, 0.90])
    plt.savefig(output_path, dpi=350)
    plt.close()


def plot_experiment3_time_breakdown(round_indices, dataset_to_times, output_path):
    """
    Exp3: per-round time breakdown for retrieval / prefill / decode.
    """
    datasets = list(dataset_to_times.keys())
    if not datasets:
        return
    rounds = np.asarray(round_indices)
    x = np.arange(len(rounds))

    fig, axes = plt.subplots(1, len(datasets), figsize=(3.0 * len(datasets), 2.9), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, label in zip(axes, datasets):
        retrieval_s, prefill_s, decode_s = dataset_to_times[label]
        retrieval_s = np.asarray(retrieval_s, dtype=float)
        prefill_s = np.asarray(prefill_s, dtype=float)
        decode_s = np.asarray(decode_s, dtype=float)

        ax.bar(x, retrieval_s, width=0.65, color="#303030", edgecolor="black", linewidth=0.4, label="Retrieval")
        ax.bar(
            x,
            prefill_s,
            width=0.65,
            bottom=retrieval_s,
            color="#c8102e",
            edgecolor="black",
            linewidth=0.4,
            hatch="....",
            label="Prefill (TTFT)",
        )
        ax.bar(
            x,
            decode_s,
            width=0.65,
            bottom=retrieval_s + prefill_s,
            color="#94061d",
            edgecolor="black",
            linewidth=0.4,
            hatch="xxxx",
            label="Decode (TPOT)",
        )
        ax.set_title(label, fontsize=9, pad=4, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"R{i}" for i in rounds], fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Time per Round (s)", fontsize=9, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.03))
    plt.tight_layout(rect=[0.04, 0.02, 1.0, 0.96])
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()


# -------------------------------------------------
# 2. Plotting function (stacked bars in your style)
# -------------------------------------------------

def plot_experiment1_kv_growth(round_indices,
                               kv_q, kv_chunks, kv_reason,
                               title, output_path):
    kv_q = np.asarray(kv_q, dtype=float)
    kv_chunks = np.asarray(kv_chunks, dtype=float)
    kv_reason = np.asarray(kv_reason, dtype=float)
    x = np.asarray(round_indices)

    gb = 1024 ** 3
    kv_q_g = kv_q / gb
    kv_chunks_g = kv_chunks / gb
    kv_reason_g = kv_reason / gb

    width = 0.65

    plt.figure(figsize=(6, 3))
    plt.bar(
        x, kv_q_g, width=width, color="black", edgecolor="black",
        linewidth=0.4, hatch="////", label="Question"
    )
    plt.bar(x, kv_chunks_g, width=width, bottom=kv_q_g,
            color="#c8102e", edgecolor="black", linewidth=0.4, hatch="....",
            label="Retrieved Chunks")
    plt.bar(x, kv_reason_g, width=width, bottom=kv_q_g + kv_chunks_g,
            color="#94061d", edgecolor="black", linewidth=0.4, hatch="xxxx",
            label="Generated Reasoning")

    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel("KV Memory (GB)")
    plt.xticks(x, [f"R{i}" for i in round_indices])
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_experiment1_multi_dataset(round_indices, dataset_to_kv, title, output_path):
    """
    Absolute KV (GB) stacked bars for multiple datasets in a single horizontal row.
    Question: black with '//' hatch.
    Retrieved chunks: dark red with dot hatch.
    Generated reasoning: light red with 'x' hatch.
    """
    datasets = list(dataset_to_kv.keys())
    if not datasets:
        raise ValueError("No dataset results provided for plotting.")

    rounds = list(round_indices)
    x = np.arange(len(rounds))

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(datasets),
        figsize=(3.0 * len(datasets), 3.0),
        sharey=True,
    )
    if len(datasets) == 1:
        axes = [axes]

    gb = 1024 ** 3
    bar_width = 0.65

    for ax, dname in zip(axes, datasets):
        kv_q, kv_chunks, kv_reason = dataset_to_kv[dname]
        kv_q = np.asarray(kv_q, dtype=float) / gb
        kv_chunks = np.asarray(kv_chunks, dtype=float) / gb
        kv_reason = np.asarray(kv_reason, dtype=float) / gb

        ax.bar(
            x,
            kv_q,
            width=bar_width,
            color="black",
            edgecolor="black",
            linewidth=0.4,
            hatch="////",
            label="Question",
        )
        ax.bar(
            x,
            kv_chunks,
            width=bar_width,
            bottom=kv_q,
            color="#c8102e",
            edgecolor="black",
            linewidth=0.4,
            hatch="....",
            label="Retrieved Chunks",
        )
        ax.bar(
            x,
            kv_reason,
            width=bar_width,
            bottom=kv_q + kv_chunks,
            color="#94061d",
            edgecolor="black",
            linewidth=0.4,
            hatch="xxxx",
            label="Generated Reasoning",
        )

        ax.set_title(dname, fontsize=9, pad=4, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"R{i}" for i in rounds], fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("KV Memory (GB)", fontsize=9, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1.4,
        bbox_to_anchor=(0.5, 1.03),
    )

    plt.tight_layout(rect=[0.04, 0.02, 1.0, 0.96])
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close()



def get_gpu_used_bytes(device_idx=0):
    if not torch.cuda.is_available():
        return 0.0
    free_b, total_b = torch.cuda.mem_get_info(device=device_idx)
    return float(total_b - free_b)


def infer_model_config(model_name, bytes_per_elem=2):
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    num_attention_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    num_kv_heads = getattr(cfg, "num_key_value_heads", None) or num_attention_heads
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
        if hidden_size is not None and num_attention_heads is not None:
            head_dim = hidden_size // num_attention_heads
    if num_layers is None or num_kv_heads is None or head_dim is None:
        raise ValueError("Cannot infer model config. Please hardcode num_layers/num_kv_heads/head_dim.")
    return {
        "num_layers": int(num_layers),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
        "bytes_per_elem": int(bytes_per_elem),
    }


def infer_max_model_len(model_name):
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    candidates = []
    for key in ("max_position_embeddings", "model_max_length", "n_positions"):
        val = getattr(cfg, key, None)
        if isinstance(val, (int, float)) and val > 0:
            candidates.append(int(val))
    if not candidates:
        return None
    return min(candidates)


def load_dataset_json(path, max_samples):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of examples.")
    data = [ex for ex in data if isinstance(ex, dict) and "question" in ex and "chunks" in ex]
    if max_samples > 0:
        data = data[:max_samples]
    if not data:
        raise ValueError("No valid examples found in dataset.")
    return data


def fastest_bucket(kv_q, kv_chunks, kv_reason):
    dq = np.mean(np.diff(kv_q)) if len(kv_q) > 1 else 0.0
    dc = np.mean(np.diff(kv_chunks)) if len(kv_chunks) > 1 else 0.0
    dr = np.mean(np.diff(kv_reason)) if len(kv_reason) > 1 else 0.0
    growth = {"question": dq, "chunks": dc, "reasoning": dr}
    return max(growth, key=growth.get)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 1: Memory growth across iterative RAG rounds")
    parser.add_argument("--input-json", type=str, default=None, help="Single dataset JSON path (question/chunks format)")
    parser.add_argument(
        "--input-jsons",
        type=str,
        nargs="+",
        default=None,
        help="Multiple dataset JSON paths for one combined figure.",
    )
    parser.add_argument(
        "--dataset-labels",
        type=str,
        nargs="+",
        default=None,
        help="Labels for --input-jsons (same order and length).",
    )
    parser.add_argument("--model-name", type=str, required=True, help="Model name/path for vLLM")
    parser.add_argument("--num-rounds", type=int, default=4, help="Number of iterative rounds (3-5 recommended)")
    parser.add_argument("--max-samples", type=int, default=50, help="Max examples to run")
    parser.add_argument("--plot-path", type=str, default="observation/obs1_plot.png", help="Output plot path")
    parser.add_argument(
        "--exp2-plot-path",
        type=str,
        default="observation/obs2_overlap_plot.png",
        help="Output overlap plot path for Experiment 2.",
    )
    parser.add_argument(
        "--exp3-plot-path",
        type=str,
        default="observation/obs3_time_breakdown_plot.png",
        help="Output latency breakdown plot path for Experiment 3.",
    )
    parser.add_argument("--output-json", type=str, default="observation/obs1_results.json", help="Output summary json")
    parser.add_argument("--max-gen-tokens", type=int, default=1000, help="Max generated tokens per round")
    parser.add_argument("--chunks-per-round", type=int, default=8, help="How many new chunks to retrieve each round")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="vLLM gpu_memory_utilization")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor_parallel_size")
    parser.add_argument("--max-model-len", type=int, default=32768, help="vLLM max_model_len")
    parser.add_argument("--max-num-seqs", type=int, default=1, help="vLLM max_num_seqs")
    parser.add_argument("--bytes-per-elem", type=int, default=2, help="2 for fp16/bf16 KV cache")
    parser.add_argument(
        "--trace-max-chars",
        type=int,
        default=20000,
        help="Max characters stored per prompt/output trace. Use -1 for no truncation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_jsons = list(args.input_jsons) if args.input_jsons else []
    if args.input_json:
        input_jsons.append(args.input_json)
    if args.dataset_labels is not None and len(args.dataset_labels) != len(input_jsons):
        raise ValueError("--dataset-labels must have same length as --input-jsons/--input-json.")

    if args.dataset_labels is None:
        dataset_labels = [p.split("/")[-1].replace(".json", "") for p in input_jsons]
    else:
        dataset_labels = args.dataset_labels

    derived_max_len = infer_max_model_len(args.model_name)
    effective_max_len = args.max_model_len
    if derived_max_len is not None and args.max_model_len > derived_max_len:
        print(
            f"[obs1] Requested max_model_len={args.max_model_len} exceeds "
            f"model limit={derived_max_len}. Using {derived_max_len}."
        )
        effective_max_len = derived_max_len

    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=effective_max_len,
        max_num_seqs=args.max_num_seqs,
    )
    tokenizer = llm.get_tokenizer()

    model_config = infer_model_config(args.model_name, args.bytes_per_elem)

    round_indices = list(range(args.num_rounds + 1))
    all_results = {}
    all_exp3_times = {}
    all_overlap_prev_results = {}
    all_overlap_round1_results = {}
    summary_datasets = []
    r4 = min(4, args.num_rounds)

    for label, path in zip(dataset_labels, input_jsons):
        dataset = load_dataset_json(path, args.max_samples)
        (
            mean_kv_q,
            mean_kv_chunks,
            mean_kv_reason,
            mean_gpu_total,
            mean_retrieval_s,
            mean_prefill_s,
            mean_decode_s,
            mean_overlap_prev,
            mean_overlap_round1,
            traces,
        ) = aggregate_experiment1_over_dataset(
            llm=llm,
            tokenizer=tokenizer,
            dataset=dataset,
            model_config=model_config,
            num_rounds=args.num_rounds,
            chunks_per_round=args.chunks_per_round,
            max_gen_tokens=args.max_gen_tokens,
            trace_max_chars=args.trace_max_chars,
        )
        all_results[label] = (mean_kv_q, mean_kv_chunks, mean_kv_reason)
        all_exp3_times[label] = (mean_retrieval_s, mean_prefill_s, mean_decode_s)
        all_overlap_prev_results[label] = mean_overlap_prev
        all_overlap_round1_results[label] = mean_overlap_round1
        dominant = fastest_bucket(mean_kv_q, mean_kv_chunks, mean_kv_reason)
        exp3_means = {
            "retrieval": float(np.mean(mean_retrieval_s[1:])) if len(mean_retrieval_s) > 1 else 0.0,
            "prefill": float(np.mean(mean_prefill_s[1:])) if len(mean_prefill_s) > 1 else 0.0,
            "decode": float(np.mean(mean_decode_s[1:])) if len(mean_decode_s) > 1 else 0.0,
        }
        exp3_bottleneck = max(exp3_means, key=exp3_means.get)
        r4_dominant = "reasoning" if mean_kv_reason[r4] > mean_kv_chunks[r4] else "chunks"
        summary_datasets.append(
            {
                "label": label,
                "input_json": path,
                "num_examples": len(dataset),
                "mean_kv_bytes": {
                    "question": np.asarray(mean_kv_q).tolist(),
                    "chunks": np.asarray(mean_kv_chunks).tolist(),
                    "reasoning": np.asarray(mean_kv_reason).tolist(),
                },
                "mean_gpu_used_bytes": np.asarray(mean_gpu_total).tolist(),
                "fastest_growing_bucket": dominant,
                "round4_chunks_vs_reasoning": r4_dominant,
                "exp3_time_s": {
                    "retrieval": np.asarray(mean_retrieval_s).tolist(),
                    "prefill": np.asarray(mean_prefill_s).tolist(),
                    "decode": np.asarray(mean_decode_s).tolist(),
                },
                "exp3_bottleneck_phase": exp3_bottleneck,
                "exp2_overlap_prev_round_pct": (np.asarray(mean_overlap_prev) * 100.0).tolist(),
                "exp2_overlap_round1_pct": (np.asarray(mean_overlap_round1) * 100.0).tolist(),
                "example_traces": traces,
            }
        )

    if len(all_results) == 1:
        only_label = list(all_results.keys())[0]
        kv_q, kv_chunks, kv_reason = all_results[only_label]
        plot_experiment1_kv_growth(
            round_indices=round_indices,
            kv_q=kv_q,
            kv_chunks=kv_chunks,
            kv_reason=kv_reason,
            title=f"Experiment 1 (dataset={only_label}, rounds={args.num_rounds})",
            output_path=args.plot_path,
        )
    else:
        plot_experiment1_multi_dataset(
            round_indices=round_indices,
            dataset_to_kv=all_results,
            title=f"Experiment 1: Memory Breakdown Across Rounds ({args.model_name})",
            output_path=args.plot_path,
        )
    plot_experiment2_overlap(
        round_indices=round_indices,
        dataset_to_overlap_prev=all_overlap_prev_results,
        dataset_to_overlap_round1=all_overlap_round1_results,
        output_path=args.exp2_plot_path,
    )
    plot_experiment3_time_breakdown(
        round_indices=round_indices,
        dataset_to_times=all_exp3_times,
        output_path=args.exp3_plot_path,
    )

    summary = {
        "num_rounds": args.num_rounds,
        "model_name": args.model_name,
        "model_config": model_config,
        "datasets": summary_datasets,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved plot:", args.plot_path)
    print("Saved Exp2 overlap plot:", args.exp2_plot_path)
    print("Saved Exp3 time breakdown plot:", args.exp3_plot_path)
    print("Saved summary:", args.output_json)


if __name__ == "__main__":
    main()
