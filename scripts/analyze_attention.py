
import torch
import numpy as np
from nanolm.viz import load_run, attention_maps

RUN = "outputs/nanolm/sweeps/char_models/gpt_modern_s1337"

texts = {
    "Shakespeare": "To be, or not to be: that is the question.",
    "Prose": "First, we observe the pattern; next, we compare each result.",
    "Sentence": "The quick brown fox jumps over the lazy dog."
}

# Load the trained model and character tokenizer
cfg, model, tok, dinfo, _ = load_run(RUN)

# Collect metrics for each head across all examples
results = {}

for name, text in texts.items():
    idx = torch.tensor([tok.encode(text)])
    maps = attention_maps(model, idx)

    for layer, layer_maps in enumerate(maps):
        for head, weights in enumerate(layer_maps):

            T = weights.shape[0]

            # Average attention to the immediately previous position
            previous = np.mean([
                weights[t, t - 1]
                for t in range(1, T)
            ])

            # Identify word-start positions
            word_starts = [
                i for i, char in enumerate(text)
                if not char.isspace()
                and (i == 0 or text[i - 1].isspace())
            ]

            # Average attention mass assigned to word starts
            word_start_mass = []

            for t in range(T):
                valid_starts = [i for i in word_starts if i <= t]
                mass = sum(weights[t, i] for i in valid_starts)
                word_start_mass.append(mass)

            word_start_avg = np.mean(word_start_mass)

            key = (layer, head)
            results.setdefault(key, {"previous": [], "word_start": []})

            results[key]["previous"].append(previous)
            results[key]["word_start"].append(word_start_avg)

# Print averages across all three examples
print("\nATTENTION HEAD ANALYSIS")
print("-" * 65)
print(f"{'Head':<15} {'Previous-token':<20} {'Word-start mass':<20}")
print("-" * 65)

for (layer, head), values in results.items():
    previous = np.mean(values["previous"])
    word_start = np.mean(values["word_start"])

    print(
        f"L{layer} H{head:<11} "
        f"{previous:<20.4f} "
        f"{word_start:<20.4f}"
    )
