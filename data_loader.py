from datasets import load_dataset
from transformers import PreTrainedTokenizer

def load_training_data(tokenizer: PreTrainedTokenizer, dataset_name="tatsu-lab/alpaca", sample_size=1000):
    """Loads and tokenizes dataset for training."""
    
  # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Assign EOS as padding token if not set
        tokenizer.padding_side = "right"

    dataset = load_dataset(dataset_name)
    train_data = dataset["train"].shuffle().select(range(sample_size))  # Take subset for efficiency

    def tokenize_function(examples):
            tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
            tokenized["labels"] = tokenized["input_ids"].copy()  # Labels should be the same as input_ids
            tokenized["labels"] = [
                [-100 if token == tokenizer.pad_token_id else token for token in label]
                for label in tokenized["labels"]
            ]
            
            return tokenized

    
    tokenized_data = train_data.map(tokenize_function, batched=True)
    return tokenized_data
