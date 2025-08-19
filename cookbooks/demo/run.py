# This will auto-download the file from HF (cached), then load it.
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF",
    filename="mistralai_Voxtral-Mini-3B-2507-Q4_K_S.gguf",
    # Optional performance knobs:
    n_ctx=8192,
    n_gpu_layers=-1,       # set >0 or -1 for full offload if you have CUDA/Metal build
    chat_format="mistral-instruct",      # use Mistral Instruct chat template
    verbose=False,
)

# Use like normal:
out = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Give me two sentences about Voxtral Mini."}],
    max_tokens=200,
    temperature=0.2,
    top_p=0.95,
)
print(out["choices"][0]["message"]["content"])
