import argparse
from pathlib import Path
import sys
import torch

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from train_tokenizer import load_bpe_tokenizer  # noqa: E402
from tests.adapters import run_transformer_lm  # noqa: E402


def load_weights_from_checkpoint(checkpoint_path: Path, device: str) -> dict[str, torch.Tensor] | None:
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model_state = checkpoint.get('model_state_dict', {})
    weights: dict[str, torch.Tensor] = {}
    for key, tensor in model_state.items():
        if isinstance(key, str) and key.startswith('params.'):
            orig_key = key.split('params.', 1)[1]
            weights[orig_key] = tensor.to(device)
    return weights if weights else None


@torch.no_grad()
def generate(
    tokenizer,
    weights: dict[str, torch.Tensor],
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    temperature: float,
    top_k: int | None,
    device: str,
) -> str:
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = []

    for _ in range(max_new_tokens):
        # Prepare input tensor (batch_size=1)
        context_ids = input_ids[-context_length:] if len(input_ids) > context_length else input_ids
        x = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)

        # Forward
        logits = run_transformer_lm(
            vocab_size=len(tokenizer.vocab),
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            weights=weights,
            in_indices=x,
        )
        logits_last = logits[:, -1, :].squeeze(0)  # [vocab]

        # Sampling
        if temperature <= 0:
            next_id = int(torch.argmax(logits_last).item())
        else:
            logits_adj = logits_last / temperature
            if top_k is not None and top_k > 0:
                values, indices = torch.topk(logits_adj, k=min(top_k, logits_adj.size(-1)))
                probs = torch.softmax(values, dim=-1)
                idx = torch.multinomial(probs, num_samples=1).item()
                next_id = int(indices[idx].item())
            else:
                probs = torch.softmax(logits_adj, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

        input_ids.append(next_id)

    return tokenizer.decode(input_ids)


def main():
    parser = argparse.ArgumentParser(description="Interactive CLI to generate next tokens from a checkpointed model")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt", help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default=None, help="Force device: cpu or cuda")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (<=0 to disable)")
    parser.add_argument("--context-length", type=int, default=256, help="Context length used during training")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=1344, help="Feedforward hidden dimension")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta parameter")
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.set_default_device(device)

    # Tokenizer
    tokenizer = load_bpe_tokenizer("ts_vocab.json", "ts_merges.txt")
    vocab_size = len(tokenizer.vocab)
    print(f"Tokenizer loaded (vocab size {vocab_size})")

    # Weights
    checkpoint_path = Path(args.checkpoint)
    weights = load_weights_from_checkpoint(checkpoint_path, device)
    if weights is None:
        print(f"Warning: checkpoint not found at {checkpoint_path}. Can't initialize random weights for serving without shapes.")
        print("Please train first (to create checkpoint.pt) or provide a valid checkpoint with trained weights.")
        return

    print("Loaded weights from checkpoint.")

    # Interactive loop
    print("Enter a prompt (or blank line to exit):")
    try:
        while True:
            prompt = input("> ").rstrip("\n")
            if prompt == "":
                break
            output = generate(
                tokenizer=tokenizer,
                weights=weights,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                context_length=args.context_length,
                d_model=args.d_model,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                d_ff=args.d_ff,
                rope_theta=args.rope_theta,
                temperature=args.temperature,
                top_k=(args.top_k if args.top_k > 0 else None),
                device=device,
            )
            print(output)
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()


