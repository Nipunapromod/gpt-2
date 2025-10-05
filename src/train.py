import os
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# === Settings ===
DATA_PATH = "./data/data.txt"   # Path to your data.txt
MODEL_NAME = "gpt2"             # 124M GPT-2
OUTPUT_DIR = "./gpt2-finetuned"

# === Load tokenizer and model ===
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = TFGPT2LMHeadModel.from_pretrained(MODEL_NAME)

# === Prepare dataset ===
def load_dataset(file_path, tokenizer, block_size=128):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read()
    encodings = tokenizer(lines, return_tensors="tf", truncation=True, padding=True, max_length=block_size)
    return tf.data.Dataset.from_tensor_slices((encodings["input_ids"], encodings["input_ids"]))

dataset = load_dataset(DATA_PATH, tokenizer)
dataset = dataset.shuffle(1000).batch(2)

# === Compile model ===
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)

# === Train ===
model.fit(dataset, epochs=1)

# === Save model ===
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training finished! Model saved to:", OUTPUT_DIR)
