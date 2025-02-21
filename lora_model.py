import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Use 8-bit quantization to save memory
    llm_int8_threshold=6.0
)

def load_lora_model(model_name="meta-llama/Meta-Llama-3-8B", r=8, alpha=16, dropout=0.05):
    """Loads LLaMA 3.0 with LoRA adapters."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load base model with 8-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config, 
        device_map="auto"
    )

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    print("\nTrainable Parameters After LoRA:")
    model.print_trainable_parameters()
    
    return model, tokenizer
