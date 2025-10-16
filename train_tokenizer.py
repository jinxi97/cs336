import os
import random
import json
import base64
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path to allow importing from tests
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from tests.adapters import get_tokenizer, run_train_bpe

# --- Constants ---

# Assumes you have trained tokenizers and saved them in these files.
# The vocab should be a JSON file mapping token ID (int) to a list of byte values (int).
# The merges should be a text file, with each line containing two space-separated,
# base64-encoded strings representing the byte pairs.
TS_VOCAB_PATH = "ts_vocab.json"
TS_MERGES_PATH = "ts_merges.txt"

TS_DATA_PATH = "data/TinyStoriesV2-GPT4-train.txt"
TS_VALIDATION_PATH = "data/TinyStoriesV2-GPT4-valid.txt"

TS_VOCAB_SIZE = 10_000

NUM_SAMPLES = 10
SPECIAL_TOKENS = ["<|endoftext|>"]
DOC_SEPARATOR = "<|endoftext|>"


def save_tokenizer(
    vocab: Dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
    vocab_path: str,
    merges_path: str,
):
    """Saves the tokenizer vocab and merges to the specified paths."""
    print(f"Saving vocabulary to {vocab_path}...")
    # Convert bytes to a list of ints for JSON serialization
    serializable_vocab = {str(k): list(v) for k, v in vocab.items()}
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_vocab, f, indent=2)

    print(f"Saving merges to {merges_path}...")
    with open(merges_path, 'w', encoding='utf-8') as f:
        for p1, p2 in merges:
            p1_b64 = base64.b64encode(p1).decode('utf-8')
            p2_b64 = base64.b64encode(p2).decode('utf-8')
            f.write(f"{p1_b64} {p2_b64}\n")
    print("Save complete.")


def train_tokenizer(
    input_path: str,
    vocab_size: int,
    vocab_path: str,
    merges_path: str,
):
    """Trains a BPE tokenizer and saves the vocab and merges."""
    print(f"--- Training Tokenizer on {input_path} ---")
    print(f"Vocab size: {vocab_size}")

    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        print("Tokenizer files already exist. Skipping training.")
        return

    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS
    )
    
    save_tokenizer(vocab, merges, vocab_path, merges_path)


def load_bpe_tokenizer(vocab_path: str, merges_path: str) -> Any:
    """
    Loads a BPE tokenizer from vocab and merges files, assuming a specific format.
    """
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        # JSON keys are strings, convert them back to integers.
        # Assumes byte values were stored as a list of integers.
        vocab: Dict[int, bytes] = {int(k): bytes(v) for k, v in vocab_data.items()}

        merges: List[Tuple[bytes, bytes]] = []
        with open(merges_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                # Assumes merges are saved as space-separated base64 strings
                p1_b64, p2_b64 = line.strip().split()
                merges.append(
                    (base64.b64decode(p1_b64), base64.b64decode(p2_b64))
                )
    except (IOError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading tokenizer files: {e}")
        print("Please ensure the vocab/merges files exist and are in the correct format.")
        sys.exit(1)

    return get_tokenizer(vocab, merges, special_tokens=SPECIAL_TOKENS)


def sample_documents(file_path: str, n: int, separator: str) -> List[str]:
    """Reads a file, splits it by a separator, and returns n random documents."""
    print(f"Sampling {n} documents from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        sys.exit(1)

    documents = content.split(separator)
    # Filter out any empty strings that may result from the split
    documents = [doc for doc in documents if doc.strip()]
    
    if len(documents) < n:
        print(f"Warning: Found only {len(documents)} documents, returning all of them.")
        return documents
        
    return random.sample(documents, n)


def calculate_compression(documents: List[str], tokenizer: Any) -> float:
    """Encodes documents and calculates the compression ratio (bytes/token)."""
    total_bytes = 0
    total_tokens = 0
    for doc in documents:
        encoded_bytes = doc.encode('utf-8')
        tokens = tokenizer.encode(doc)
        total_bytes += len(encoded_bytes)
        total_tokens += len(tokens)
    
    if total_tokens == 0:
        return 0.0
        
    return total_bytes / total_tokens


def run_experiment_a():
    """
    Runs experiment (a): sampling documents, encoding them with corresponding
    tokenizers, and calculating the compression ratio.
    """
    print("--- Running Tokenizer Experiment (a) ---")

    # Check for tokenizer files first
    if not all(os.path.exists(p) for p in [TS_VOCAB_PATH, TS_MERGES_PATH]):
        print("\nError: One or more tokenizer files are missing.")
        print("This script expects previously-trained tokenizers. Please run training to generate:")
        print(f"- {TS_VOCAB_PATH} (10K vocab from TinyStories)")
        print(f"- {TS_MERGES_PATH}")
        return

    # Sample documents
    ts_docs = sample_documents(TS_DATA_PATH, NUM_SAMPLES, DOC_SEPARATOR)
    print("Sampling complete.")

    # Load tokenizers
    print("Loading tokenizers...")
    ts_tokenizer = load_bpe_tokenizer(TS_VOCAB_PATH, TS_MERGES_PATH)
    print("Tokenizers loaded.")

    # Calculate compression ratio
    print("Calculating compression ratios...")
    ts_compression = calculate_compression(ts_docs, ts_tokenizer)

    # Print response
    print("\n--- Results for (a) ---")
    print(f"TinyStories Tokenizer Compression Ratio: {ts_compression:.2f} bytes/token")


def main():
    parser = argparse.ArgumentParser(description="Run tokenizer experiments and training.")
    parser.add_argument(
        "command",
        choices=["train-tinystories", "run-experiment-a"],
        help="The command to execute."
    )
    args = parser.parse_args()

    if args.command == "train-tinystories":
        train_tokenizer(
            input_path=TS_DATA_PATH,
            vocab_size=TS_VOCAB_SIZE,
            vocab_path=TS_VOCAB_PATH,
            merges_path=TS_MERGES_PATH,
        )
    elif args.command == "run-experiment-a":
        run_experiment_a()


if __name__ == "__main__":
    main()
