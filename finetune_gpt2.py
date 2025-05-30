from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import json
import torch
import os

# Check if we have a pre-trained model and continue training from there
def get_model_path():
    """
    Determine which model to use: existing fine-tuned or base GPT-2
    """
    finetuned_path = "./gpt2_finetuned"
    if os.path.exists(finetuned_path) and os.path.exists(os.path.join(finetuned_path, "config.json")):
        print(f"Found existing fine-tuned model at {finetuned_path}")
        return finetuned_path
    else:
        print("No existing fine-tuned model found, starting from base GPT-2")
        return "gpt2"

# This function is kept for backward compatibility but is no longer used
# The training now uses the comprehensive data from synthetic_dataset.jsonl
def generate_synthetic_dataset():
    """
    Legacy function - now we use the comprehensive JSONL file directly
    """
    print("Warning: This function is deprecated. Using synthetic_dataset.jsonl instead.")
    return []

# Save dataset to JSONL
def save_dataset(dataset, filename="synthetic_dataset.jsonl"):
    """
    Saves the dataset to a JSONL file.
    """
    with open(filename, "w") as f:
        for item in dataset:  # Iterate over dataset, not file
            f.write(json.dumps(item) + "\n")

# Load dataset for fine-tuning
def load_dataset(filename="synthetic_dataset.jsonl"):
    """
    Loads the dataset from a JSONL file.
    """
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_dict({"text": [item["text"] for item in data]})

# Fine-tune GPT-2
def finetune_model(epochs=3, batch_size=1, learning_rate=5e-5, use_existing_data=True):
    """
    Fine-tune GPT-2 model using existing JSON data or generated synthetic data
    
    Args:
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        learning_rate (float): Learning rate for training
        use_existing_data (bool): Whether to use existing JSON file or generate new data
    """
    # Determine model path (existing fine-tuned or base GPT-2)
    model_path = get_model_path()
    
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    if use_existing_data:
        print("Loading existing dataset from synthetic_dataset.jsonl")
        dataset = load_dataset()
    else:
        print("Generating new synthetic dataset")
        # Generate and save dataset
        synthetic_data = generate_synthetic_dataset()
        save_dataset(synthetic_data)
        dataset = load_dataset()
    
    print(f"Dataset size: {len(dataset)}")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,  # Increased for longer examples
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training arguments with better configuration
    training_args = TrainingArguments(
        output_dir="./gpt2_finetuned",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=max(50, len(dataset) // 4),  # Save more frequently for smaller datasets
        save_total_limit=2,  # Keep last 2 checkpoints
        logging_steps=10,
        learning_rate=learning_rate,
        warmup_steps=min(100, len(dataset) // 10),  # Warmup for better convergence
        logging_dir='./logs',
        eval_steps=None,  # No evaluation for now
        load_best_model_at_end=False,
        report_to=None,  # Disable wandb/tensorboard logging
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Check if we're continuing from a checkpoint
    checkpoint_dir = None
    if model_path == "./gpt2_finetuned":
        # Look for the latest checkpoint
        checkpoint_dirs = [d for d in os.listdir("./gpt2_finetuned") if d.startswith("checkpoint-")]
        if checkpoint_dirs:
            latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))
            checkpoint_dir = os.path.join("./gpt2_finetuned", latest_checkpoint)
            print(f"Resuming training from checkpoint: {checkpoint_dir}")
    
    # Fine-tune
    print(f"Starting training for {epochs} epochs...")
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        trainer.train()
    
    # Save final model
    print("Saving final model...")
    model.save_pretrained("./gpt2_finetuned")
    tokenizer.save_pretrained("./gpt2_finetuned")
    
    print("Training completed successfully!")
    return model, tokenizer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune GPT-2 for code debugging')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--generate_new', action='store_true', help='Generate new synthetic data instead of using existing')
    
    args = parser.parse_args()
    
    # Run fine-tuning
    model, tokenizer = finetune_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_existing_data=not args.generate_new
    )
    
    print(f"\nModel fine-tuned with {args.epochs} epochs")
    print("You can now use the fine-tuned model in recommendation_engine.py")