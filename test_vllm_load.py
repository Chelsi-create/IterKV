# from vllm import LLM

# llm = LLM(
#     model="meta-llama/Meta-Llama-3-8B-Instruct",
#     tensor_parallel_size=1,
#     gpu_memory_utilization=0.60,
#     max_model_len=4096,
#     max_num_seqs=1,
#     trust_remote_code=True,
# )

# print("Loaded OK")


# from vllm import LLM

# llm = LLM(
#     model="mistralai/Mistral-7B-Instruct-v0.2",
#     tensor_parallel_size=1,
#     gpu_memory_utilization=0.60,
#     max_model_len=4096,
#     max_num_seqs=1,
# )
# print("Loaded small model OK")from vllm import LLM, SamplingParams

from vllm import LLM, SamplingParams

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", tensor_parallel_size=1)
out = llm.generate(["hello"], SamplingParams(max_tokens=1), use_tqdm=False)
print(out[0].outputs[0].text)




