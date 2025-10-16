import sys
import os
import time
from pathlib import Path
import numpy as np
import torch

# Add project root to path to allow importing from helper scripts
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from train_tokenizer import load_bpe_tokenizer, TS_DATA_PATH, TS_VALIDATION_PATH
from tests.adapters import (
    run_get_batch,
    get_adamw_cls,
    run_get_lr_cosine_schedule,
    run_cross_entropy,
    run_gradient_clipping,
    run_transformer_lm,
)

# --- Hyperparameters ---
# Training
BATCH_SIZE = 64
CONTEXT_LENGTH = 256
TOTAL_STEP_COUNT = 5000
VALIDATION_INTERVAL = 1
VALIDATION_BATCHES = 20

# Model
D_MODEL = 256
NUM_LAYERS = 6
NUM_HEADS = 4
D_FF = D_MODEL * 4
ROPE_THETA = 10000.0

# Optimizer
MAX_LEARNING_RATE = 6e-4
MIN_LEARNING_RATE = 6e-5
WARMUP_ITERS = 2000
COSINE_CYCLE_ITERS = 5000
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0


def initialize_weights(vocab_size, num_layers, d_model, d_ff, device):
    """Initializes a dictionary of transformer weights."""
    print("Initializing weights")
    weights = {
        'token_embeddings.weight': torch.randn(vocab_size, d_model),
        'ln_final.weight': torch.randn(d_model),
    }
    # Tie output projection to token embeddings
    weights['lm_head.weight'] = weights['token_embeddings.weight']

    for i in range(num_layers):
        weights[f'layers.{i}.attn.q_proj.weight'] = torch.randn(d_model, d_model)
        weights[f'layers.{i}.attn.k_proj.weight'] = torch.randn(d_model, d_model)
        weights[f'layers.{i}.attn.v_proj.weight'] = torch.randn(d_model, d_model)
        weights[f'layers.{i}.attn.output_proj.weight'] = torch.randn(d_model, d_model)
        weights[f'layers.{i}.ln1.weight'] = torch.randn(d_model)
        weights[f'layers.{i}.ffn.w1.weight'] = torch.randn(d_ff, d_model)
        weights[f'layers.{i}.ffn.w2.weight'] = torch.randn(d_model, d_ff)
        weights[f'layers.{i}.ffn.w3.weight'] = torch.randn(d_ff, d_model)
        weights[f'layers.{i}.ln2.weight'] = torch.randn(d_model)

    # Move to device and set requires_grad
    for k, v in weights.items():
        weights[k] = v.to(device).requires_grad_()
        
    return weights


@torch.no_grad()
def estimate_loss(weights, vocab_size, tokenized_val_data, device):
    """Estimates the validation loss."""
    losses = torch.zeros(VALIDATION_BATCHES)
    for k in range(VALIDATION_BATCHES):
        x, y = run_get_batch(
            dataset=tokenized_val_data,
            batch_size=BATCH_SIZE,
            context_length=CONTEXT_LENGTH,
            device=device
        )
        logits = run_transformer_lm(
            vocab_size, CONTEXT_LENGTH, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, ROPE_THETA, weights, x
        )
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = y.view(-1)
        loss = run_cross_entropy(logits_flat, targets_flat)
        losses[k] = loss.item()
    return losses.mean()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Tokenizer and Data Loading ---
    tokenizer = load_bpe_tokenizer("ts_vocab.json", "ts_merges.txt")
    vocab_size = len(tokenizer.vocab)
    print(f"Tokenizer loaded with vocab size: {vocab_size}")

    with open(TS_DATA_PATH, 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(TS_VALIDATION_PATH, 'r', encoding='utf-8') as f:
        val_text = f.read()
    
    tokenized_train_data = np.array(tokenizer.encode(train_text), dtype=np.int64)
    tokenized_val_data = np.array(tokenizer.encode(val_text), dtype=np.int64)

    # --- Model and Optimizer Initialization ---
    torch.manual_seed(1337)
    weights = initialize_weights(vocab_size, NUM_LAYERS, D_MODEL, D_FF, device)
    
    AdamW = get_adamw_cls()
    optimizer = AdamW(weights.values(), lr=MAX_LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(BETA1, BETA2))

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    start_time = time.time()
    for step in range(TOTAL_STEP_COUNT):
        lr = run_get_lr_cosine_schedule(step, MAX_LEARNING_RATE, MIN_LEARNING_RATE, WARMUP_ITERS, COSINE_CYCLE_ITERS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        x, y = run_get_batch(tokenized_train_data, BATCH_SIZE, CONTEXT_LENGTH, device)

        # Forward pass
        logits = run_transformer_lm(
            vocab_size, CONTEXT_LENGTH, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, ROPE_THETA, weights, x
        )
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = y.view(-1)
        loss = run_cross_entropy(logits_flat, targets_flat)

        # Backward pass and update
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        run_gradient_clipping(weights.values(), max_l2_norm=GRAD_CLIP)
        optimizer.step()

        if step % VALIDATION_INTERVAL == 0 or step == TOTAL_STEP_COUNT - 1:
            val_loss = estimate_loss(weights, vocab_size, tokenized_val_data, device)
            end_time = time.time()
            print(f"Step {step:4d}/{TOTAL_STEP_COUNT}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}, Time: {(end_time-start_time)*1000:.2f}ms")
            start_time = time.time()

if __name__ == "__main__":
    main()
