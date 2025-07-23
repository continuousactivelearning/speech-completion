from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
BASE_MODEL = "google/gemma-2b"  
ADAPTER_PATH = "gemma-2b-finetuned"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32  
)

gemma_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
gemma_model.to(device)
gemma_model.eval()

def generate_from_base(instruction: str, max_new_tokens: int = 1024, input_text: str = ""):
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text:
        prompt += f"\n### Input:\n{input_text}"
    prompt += "\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        output = gemma_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)
