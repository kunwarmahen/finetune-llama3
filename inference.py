import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_finetuned_model(model_name="meta-llama/Meta-Llama-3-8B", lora_path="./lora_llama3"):
    """Loads fine-tuned LLaMA 3.1 model with LoRA."""
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, load_in_8bit=True, device_map="auto"
    )

    # Load fine-tuned LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.to("cuda")

    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100):
    """Generates text using the fine-tuned model."""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=max_length)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    model, tokenizer = load_finetuned_model()
    prompt = "How does AI impact sales strategy?"
    
    response = generate_text(model, tokenizer, prompt)
    print("\nGenerated Response:\n", response)
