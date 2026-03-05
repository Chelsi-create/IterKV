#!/usr/bin/env bash

PYTHON_BIN="python"
SCRIPT_PATH="pipeline/data/load_dataset.py"
DATASET="2wiki" # options: hotpot, musique, 2wiki
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
CHUNK_TOKEN_LEN=1024
MAX_CORPUS_DOCS=50000
MAX_QUERIES=1000
MIN_CHUNKS_PER_QUERY=300
MAX_CHUNKS_PER_QUERY=400
OUTPUT_JSON="dataset/2wiki_rag512_q1000.json"

# Avoid thread-creation failures on constrained systems during HF dataset generation.
export ARROW_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Output file: $OUTPUT_JSON"

$PYTHON_BIN "$SCRIPT_PATH" \
  --dataset "$DATASET" \
  --model-name "$MODEL_NAME" \
  --embed-model "$EMBED_MODEL" \
  --chunk-token-len "$CHUNK_TOKEN_LEN" \
  --max-corpus-docs "$MAX_CORPUS_DOCS" \
  --max-queries "$MAX_QUERIES" \
  --min-chunks-per-query "$MIN_CHUNKS_PER_QUERY" \
  --max-chunks-per-query "$MAX_CHUNKS_PER_QUERY" \
  --output-json "$OUTPUT_JSON"
