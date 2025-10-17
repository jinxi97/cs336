import sys
import os
import time
from pathlib import Path
import numpy as np
import torch
from multiprocessing import Pool

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
    run_save_checkpoint,
    run_load_checkpoint,
)

# --- Hyperparameters ---
# Training
BATCH_SIZE = 64
CONTEXT_LENGTH = 256
TOTAL_STEP_COUNT = 5000
VALIDATION_INTERVAL = 1
VALIDATION_BATCHES = 20

# Model
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS = 16
D_FF = 1344
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
        'lm_head.weight': torch.randn(vocab_size, d_model),
    }

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


class WeightsModule(torch.nn.Module):
    def __init__(self, weights_dict: dict[str, torch.Tensor]):
        super().__init__()
        # Sanitize parameter names (PyTorch forbids '.' in parameter names)
        self._orig_to_sanitized: dict[str, str] = {}
        self._sanitized_to_orig: dict[str, str] = {}

        def _sanitize(name: str) -> str:
            return name.replace('.', '__DOT__')

        sanitized_pairs = []
        for orig_key, tensor in weights_dict.items():
            sanitized_key = _sanitize(orig_key)
            self._orig_to_sanitized[orig_key] = sanitized_key
            self._sanitized_to_orig[sanitized_key] = orig_key
            sanitized_pairs.append((sanitized_key, tensor))

        # Store all weights inside a ParameterDict for easy checkpointing, using sanitized keys
        self.params = torch.nn.ParameterDict({
            k: torch.nn.Parameter(v.detach().clone()) for k, v in sanitized_pairs
        })

    def state_dict(self, *args, **kwargs):
        # Expose parameters under their original keys for serialization
        base = super().state_dict(*args, **kwargs)
        remapped: dict[str, torch.Tensor] = {}
        for k, v in base.items():
            if k.startswith('params.'):
                sanitized = k.split('params.', 1)[1]
                orig = self._sanitized_to_orig.get(sanitized, sanitized)
                remapped[f'params.{orig}'] = v
            else:
                remapped[k] = v
        return remapped

    def load_state_dict(self, state_dict, strict: bool = True):
        # Accept state dicts saved with original keys by remapping to sanitized
        mapped: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if isinstance(k, str) and k.startswith('params.'):
                name = k.split('params.', 1)[1]
                # Prefer explicit mapping if present
                if name in self._orig_to_sanitized:
                    mapped[f'params.{self._orig_to_sanitized[name]}'] = v
                else:
                    # Fallback: if already sanitized or unknown, sanitize deterministically
                    sanitized = name.replace('.', '__DOT__')
                    mapped[f'params.{sanitized}'] = v
            else:
                mapped[k] = v
        return super().load_state_dict(mapped, strict=strict)

    def by_original_keys(self) -> dict[str, torch.Tensor]:
        # Convenience view: original key -> Parameter tensor
        return {orig: self.params[sanitized] for orig, sanitized in self._orig_to_sanitized.items()}

_POOL_TOKENIZER = None

def _init_tokenizer_pool(vocab_path: str, merges_path: str):
    global _POOL_TOKENIZER
    # Each worker loads its own tokenizer to avoid cross-process pickling issues
    _POOL_TOKENIZER = load_bpe_tokenizer(vocab_path, merges_path)


def _encode_lines_batch(lines: list[str]) -> tuple[list[int], int]:
    # Returns (encoded_token_ids, line_count)
    tokens: list[int] = []
    for line in lines:
        tokens.extend(_POOL_TOKENIZER.encode(line))
    return tokens, len(lines)


def encode_file_concurrently(
    file_path: str,
    vocab_path: str,
    merges_path: str,
    batch_lines: int = 1000,
    num_workers: int | None = None,
) -> list[int]:
    """Encode a text file concurrently by batching lines across processes."""
    # Build batches of lines
    batches: list[list[str]] = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        current: list[str] = []
        for line in f:
            current.append(line)
            if len(current) >= batch_lines:
                batches.append(current)
                current = []
        if current:
            batches.append(current)

    if not batches:
        return []

    tokens_all: list[int] = []
    processed_lines = 0
    next_report = 1000

    with Pool(processes=num_workers, initializer=_init_tokenizer_pool, initargs=(vocab_path, merges_path)) as pool:
        for encoded_tokens, line_count in pool.imap(_encode_lines_batch, batches):
            tokens_all.extend(encoded_tokens)
            processed_lines += line_count
            if processed_lines >= next_report:
                print(f"Encoded {processed_lines} lines from {file_path}")
                next_report += 1000

    return tokens_all


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

    print("Start to encode")
    train_tokens_path = Path(TS_DATA_PATH).with_suffix('.tokens.npy')
    val_tokens_path = Path(TS_VALIDATION_PATH).with_suffix('.tokens.npy')

    if train_tokens_path.exists():
        print(f"Loading cached train tokens from {train_tokens_path}")
        tokenized_train_data = np.load(train_tokens_path).astype(np.int64, copy=False)
    else:
        tokens_list = encode_file_concurrently(
            TS_DATA_PATH,
            "ts_vocab.json",
            "ts_merges.txt",
            batch_lines=8000,
            num_workers=8,
        )
        tokenized_train_data = np.array(tokens_list, dtype=np.int64)
        np.save(train_tokens_path, tokenized_train_data)
        print(f"Saved train tokens to {train_tokens_path}")

    if val_tokens_path.exists():
        print(f"Loading cached validation tokens from {val_tokens_path}")
        tokenized_val_data = np.load(val_tokens_path).astype(np.int64, copy=False)
    else:
        tokens_list = encode_file_concurrently(
            TS_VALIDATION_PATH,
            "ts_vocab.json",
            "ts_merges.txt",
            batch_lines=8000,
            num_workers=8,
        )
        tokenized_val_data = np.array(tokens_list, dtype=np.int64)
        np.save(val_tokens_path, tokenized_val_data)
        print(f"Saved validation tokens to {val_tokens_path}")
    print("Finished encoding")

    # --- Model and Optimizer Initialization ---
    torch.manual_seed(1337)
    weights = initialize_weights(vocab_size, NUM_LAYERS, D_MODEL, D_FF, device)
    # Wrap weights for checkpointing
    weights_module = WeightsModule(weights)
    
    AdamW = get_adamw_cls()
    # Deduplicate params: weight tying means lm_head.weight and token_embeddings.weight are identical
    unique_params = list({id(p): p for p in weights_module.parameters()}.values())
    optimizer = AdamW(unique_params, lr=MAX_LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(BETA1, BETA2))

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    # Attempt resume
    start_iteration = 0
    checkpoint_path = Path("checkpoint.pt")
    if checkpoint_path.exists():
        try:
            start_iteration = run_load_checkpoint(str(checkpoint_path), weights_module, optimizer)
            print(f"Resumed from checkpoint at iteration {start_iteration}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint: {e}")

    start_time = time.time()
    for step in range(start_iteration, TOTAL_STEP_COUNT):
        print(f"Step: {step}")
        lr = run_get_lr_cosine_schedule(step, MAX_LEARNING_RATE, MIN_LEARNING_RATE, WARMUP_ITERS, COSINE_CYCLE_ITERS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        x, y = run_get_batch(tokenized_train_data, BATCH_SIZE, CONTEXT_LENGTH, device)

        # Forward pass
        logits = run_transformer_lm(
            vocab_size, CONTEXT_LENGTH, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, ROPE_THETA,
            weights_module.by_original_keys(), x
        )
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = y.view(-1)
        loss = run_cross_entropy(logits_flat, targets_flat)

        # Backward pass and update
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        run_gradient_clipping(unique_params, max_l2_norm=GRAD_CLIP)
        optimizer.step()

        if step % VALIDATION_INTERVAL == 0 or step == TOTAL_STEP_COUNT - 1:
            val_loss = estimate_loss(weights_module.by_original_keys(), vocab_size, tokenized_val_data, device)
            end_time = time.time()
            print(f"Step {step:4d}/{TOTAL_STEP_COUNT}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}, Time: {(end_time-start_time)*1000:.2f}ms")
            start_time = time.time()

        # Periodic checkpoint
        if (step + 1) % 1000 == 0 or step == TOTAL_STEP_COUNT - 1:
            run_save_checkpoint(weights_module, optimizer, step + 1, str(checkpoint_path))
            print(f"Checkpoint saved at iteration {step + 1}")

if __name__ == "__main__":
    main()
