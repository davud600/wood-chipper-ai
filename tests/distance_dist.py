import random
import numpy as np
import matplotlib.pyplot as plt


def sample_shifted_erlang(
    shape: int, scale: float, offset: float, integer: bool = True
):
    """
    Draw from Erlang/Gamma(shape, scale) and then add `offset`.
    The result is always >= offset.
    """
    val = random.gammavariate(shape, scale) + offset
    return int(round(val)) if integer else val


# Example single draw
dist = sample_shifted_erlang(shape=3, scale=50, offset=20)
print(f"Shifted sample (>=20): {dist}")

# Plot many to see the shape
if __name__ == "__main__":
    shape, scale, offset = 3, 50, 20
    n_samples = 10_000
    samples = [sample_shifted_erlang(shape, scale, offset) for _ in range(n_samples)]

    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=50, density=True, edgecolor="black", alpha=0.7)
    plt.title(f"Erlang(shape={shape}, scale={scale}) + offset={offset}")
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("shifted_erlang.png")
    print("Saved shifted_erlang.png")
