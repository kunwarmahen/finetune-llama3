import torch
from transformers import TrainingArguments, Trainer
from lora_model import load_lora_model
from data_loader import load_training_data

# Load model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
model, tokenizer = load_lora_model(model_name)

# Load dataset
train_data = load_training_data(tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./lora_llama3",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="adamw_bnb_8bit",
    save_total_limit=2,
    eval_strategy="no",
    save_strategy="steps",
    save_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    max_steps=1000,
    fp16=True,
    report_to="none"
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("./lora_llama3")
