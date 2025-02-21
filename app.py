import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI()

# Load Fine-Tuned Model
model_name = "meta-llama/Meta-Llama-3-8B"
lora_path = "./lora_llama3"

print("🔄 Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
model = PeftModel.from_pretrained(base_model, lora_path)
model = model.to("cuda")

print("✅ Model Loaded Successfully!")

# Request Schema
class RequestInput(BaseModel):
    prompt: str
    max_length: int = 100

# API Endpoint
@app.post("/generate/")
def generate_text(request: RequestInput):
    """Generate text from fine-tuned LoRA LLaMA model."""
    inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=request.max_length)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": response}

# Health Check
@app.get("/")
def health_check():
    return {"status": "API is running"}
