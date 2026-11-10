import pandas as pd
import matplotlib.pyplot as plt

# Load the actual experiment results
df = pd.read_csv("outputs/nanolm/sweeps/context_length/summary.csv")

# Extract context length from names such as lstm_128 and gpt_256
df["context_length"] = df["name"].str.extract(r"_(\d+)$").astype(int)

# Separate and sort architectures
lstm = df[df["name"].str.startswith("lstm")].sort_values("context_length")
gpt = df[df["name"].str.startswith("gpt")].sort_values("context_length")

# Create plot
plt.figure(figsize=(8, 5))

plt.plot(
    lstm["context_length"],
    lstm["val_bpb"],
    marker="o",
    linewidth=2,
    label="LSTM"
)

plt.plot(
    gpt["context_length"],
    gpt["val_bpb"],
    marker="o",
    linewidth=2,
    label="Modern Transformer"
)

plt.xlabel("Context Length (tokens)")
plt.ylabel("Validation Bits per Byte (BPB)")
plt.title("Validation BPB vs. Context Length")
plt.xticks([64, 128, 256, 512])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save figure
plt.savefig(
    "outputs/nanolm/sweeps/context_length/context_length_bpb.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
