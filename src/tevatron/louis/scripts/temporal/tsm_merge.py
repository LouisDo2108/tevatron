import os
import argparse
import torch
from safetensors.torch import load_file, save_file
from typing import List, Dict


def average_safetensors(
    file_paths: List[str], output_path: str, use_fp32_for_accumulation: bool = True
) -> None:
    """
    Average weights from multiple .safetensors checkpoint files and save the result.

    Args:
        file_paths: List of paths to input .safetensors files
        output_path: Path to save the averaged .safetensors file
        use_fp32_for_accumulation: If True, cast to FP32 before summing to prevent
                                   precision loss in half-precision models.
    """
    if not file_paths:
        raise ValueError("No input files provided.")

    n_files = len(file_paths)
    avg_weights: Dict[str, torch.Tensor] = {}
    original_dtypes: Dict[str, torch.dtype] = {}

    with torch.no_grad():
        for i, path in enumerate(file_paths):
            print(f"[{i+1}/{n_files}] Loading {os.path.basename(path)}...")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint not found: {path}")

            weights = load_file(path)

            if i == 0:
                # Store original dtypes for final casting
                original_dtypes = {k: v.dtype for k, v in weights.items()}

                # Initialize accumulator
                if use_fp32_for_accumulation:
                    avg_weights = {k: v.float().clone() for k, v in weights.items()}
                else:
                    avg_weights = {k: v.clone() for k, v in weights.items()}
            else:
                # Validate that all checkpoints have identical keys
                if set(weights.keys()) != set(avg_weights.keys()):
                    missing = set(avg_weights.keys()) - set(weights.keys())
                    extra = set(weights.keys()) - set(avg_weights.keys())
                    raise ValueError(
                        f"Key mismatch in {path}.\n"
                        f"Missing keys: {missing}\n"
                        f"Extra keys: {extra}"
                    )

                # Accumulate weights in-place
                for k, v in weights.items():
                    tensor_to_add = v.float() if use_fp32_for_accumulation else v
                    avg_weights[k].add_(tensor_to_add)

        # Compute average
        print("Computing average...")
        for k in avg_weights:
            avg_weights[k].div_(n_files)

            # Cast back to original dtype if FP32 accumulation was used
            if use_fp32_for_accumulation:
                avg_weights[k] = avg_weights[k].to(original_dtypes[k])

    # Save result
    print(f"Saving averaged model to {output_path}...")
    save_file(avg_weights, output_path)
    print("✅ Successfully saved averaged checkpoint.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Average multiple .safetensors model checkpoints"
    )
    parser.add_argument("inputs", nargs="+", help="Paths to input .safetensors files")
    parser.add_argument(
        "-o", "--output", required=True, help="Path for the output .safetensors file"
    )
    parser.add_argument(
        "--no-fp32",
        action="store_true",
        help="Disable FP32 accumulation (use original dtype)",
    )
    args = parser.parse_args()

    average_safetensors(
        file_paths=args.inputs,
        output_path=args.output,
        use_fp32_for_accumulation=not args.no_fp32,
    )
