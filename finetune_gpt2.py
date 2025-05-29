from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import json
import torch

# Generate synthetic dataset
def generate_synthetic_dataset():
    """
    Creates a synthetic dataset of error-code-fix pairs for Python, Java, and C++.
    """
    dataset = []
    
    # Python: IndexError
    dataset.append({
        "text": "Prompt: Fix a Python error.\nError: IndexError at line 3\nMessage: list index out of range\nCode line: print(items[i])\nVariables: items\nIntent: loop, function\nCompletion: Corrected Code:\n```python\nfor i in range(len(items)):\n    print(items[i])\n```\nExplanation: The IndexError occurs because the index 'i' exceeds the list length. Using len(items) ensures valid indices.\nBest Practice: Check list bounds or use try-except."
    })
    dataset.append({
        "text": "Prompt: Fix a Python error.\nError: IndexError at line 5\nMessage: list index out of range\nCode line: value = array[i]\nVariables: array\nIntent: loop\nCompletion: Corrected Code:\n```python\ntry:\n    value = array[i]\nexcept IndexError:\n    value = None\n```\nExplanation: The IndexError is due to an invalid index. A try-except block handles the error gracefully.\nBest Practice: Use try-except for error-prone operations."
    })
    dataset.append({
        "text": "Prompt: Fix a Python error.\nError: IndexError at line 4\nMessage: list index out of range\nCode line: result = data[i]\nVariables: data\nIntent: function\nCompletion: Corrected Code:\n```python\nif i < len(data):\n    result = data[i]\n```\nExplanation: The IndexError occurs due to an out-of-bounds index. Checking the length prevents the error.\nBest Practice: Validate indices before accessing lists."
    })
    
    # Java: NullPointerException
    dataset.append({
        "text": "Prompt: Fix a Java error.\nError: NullPointerException at line 4\nMessage: Cannot invoke method on null object\nCode line: System.out.println(s.length());\nVariables: s\nIntent: function, class\nCompletion: Corrected Code:\n```java\nif (s != null) {\n    System.out.println(s.length());\n}\n```\nExplanation: The NullPointerException occurs because 's' is null. Checking for null prevents the error.\nBest Practice: Use null checks or Optional."
    })
    dataset.append({
        "text": "Prompt: Fix a Java error.\nError: NullPointerException at line 6\nMessage: null object\nCode line: obj.method();\nVariables: obj\nIntent: function\nCompletion: Corrected Code:\n```java\nobj = new Object(); // Initialize\nobj.method();\n```\nExplanation: The NullPointerException occurs due to an uninitialized object. Initializing 'obj' fixes it.\nBest Practice: Initialize objects at declaration."
    })
    dataset.append({
        "text": "Prompt: Fix a Java error.\nError: NullPointerException at line 5\nMessage: null reference\nCode line: str.toUpperCase();\nVariables: str\nIntent: method\nCompletion: Corrected Code:\n```java\nstr = str != null ? str.toUpperCase() : \"\";\n```\nExplanation: The NullPointerException occurs because 'str' is null. Using a ternary operator ensures safety.\nBest Practice: Use null-safe operations."
    })
    
    # C++: out_of_range
    dataset.append({
        "text": "Prompt: Fix a C++ error.\nError: out_of_range at line 4\nMessage: vector index exceeds size\nCode line: int x = vec[10];\nVariables: vec\nIntent: function\nCompletion: Corrected Code:\n```cpp\nif (10 < vec.size()) {\n    int x = vec[10];\n}\n```\nExplanation: The out_of_range error occurs because the index exceeds the vector size. Checking the size prevents the error.\nBest Practice: Use at() for bounds checking."
    })
    dataset.append({
        "text": "Prompt: Fix a C++ error.\nError: out_of_range at line 5\nMessage: invalid vector index\nCode line: value = vector[5];\nVariables: vector\nIntent: function\nCompletion: Corrected Code:\n```cpp\nvalue = vector.at(5);\n```\nExplanation: The out_of_range error is due to an invalid index. Using at() provides bounds checking.\nBest Practice: Prefer at() over [] for safety."
    })
    dataset.append({
        "text": "Prompt: Fix a C++ error.\nError: out_of_range at line 6\nMessage: vector access out of bounds\nCode line: result = arr[8];\nVariables: arr\nIntent: loop\nCompletion: Corrected Code:\n```cpp\nresult = arr.at(8);\n```\nExplanation: The out_of_range error occurs due to an invalid index. The at() method checks bounds automatically.\nBest Practice: Use at() for safe vector access."
    })
    
    return dataset

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
def finetune_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Generate and save dataset
    dataset = generate_synthetic_dataset()
    save_dataset(dataset)
    
    # Load dataset
    dataset = load_dataset()
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=384,  # Combined prompt + completion
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./gpt2_finetuned",
        num_train_epochs=5,
        per_device_train_batch_size=1,
        save_steps=500,
        save_total_limit=1,
        logging_steps=50,
        learning_rate=5e-5,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Fine-tune
    trainer.train()
    
    # Save model
    model.save_pretrained("./gpt2_finetuned")
    tokenizer.save_pretrained("./gpt2_finetuned")

if __name__ == "__main__":
    finetune_model()