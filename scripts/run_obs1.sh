#!/usr/bin/env bash

PYTHON_BIN="python"
SCRIPT_PATH="observation/obs1.py"

MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
NUM_ROUNDS=5
MAX_SAMPLES=50
MAX_GEN_TOKENS=128
TENSOR_PARALLEL_SIZE=1
BYTES_PER_ELEM=2
CHUNKS_PER_ROUND=20
GPU_MEMORY_UTILIZATION=0.60
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=1
OUTPUT_JSON="results/obs1/obs1_output.json"
PLOT_PATH="results/obs1/obs1_plot.png"
EXP2_PLOT_PATH="results/obs2/obs2_plot.png"
EXP3_PLOT_PATH="results/obs3/obs3_plot.png"

INPUT_JSONS=(
  "dataset/musique_rag512_q1000.json"
  "dataset/hotpot_rag512_q1000.json"
  "dataset/2wiki_rag512_q1000.json"
)

DATASET_LABELS=(
  "MuSiQue"
  "HotpotQA"
  "2WikiMQA"
)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"

$PYTHON_BIN "$SCRIPT_PATH" \
  --input-jsons "${INPUT_JSONS[@]}" \
  --dataset-labels "${DATASET_LABELS[@]}" \
  --model-name "$MODEL_NAME" \
  --num-rounds "$NUM_ROUNDS" \
  --max-samples "$MAX_SAMPLES" \
  --max-gen-tokens "$MAX_GEN_TOKENS" \
  --chunks-per-round "$CHUNKS_PER_ROUND" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --bytes-per-elem "$BYTES_PER_ELEM" \
  --output-json "$OUTPUT_JSON" \
  --plot-path "$PLOT_PATH" \
  --exp2-plot-path "$EXP2_PLOT_PATH" \
  --exp3-plot-path "$EXP3_PLOT_PATH"
