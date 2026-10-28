import sys
import os
import json
import time
import numpy as np

# ==========================================
# 1. CHECK & LOAD MODEL
# ==========================================
MODEL_FILE = "model.json"

if not os.path.exists(MODEL_FILE):
    print(f"Error: Could not find {MODEL_FILE} in the current directory.")
    sys.exit(1)

print(f"Loading {MODEL_FILE}...")
with open(MODEL_FILE, "r") as f:
    data = json.load(f)

vocab = data["vocab"]
context_len = data["contextLen"]

# Parse layers into NumPy arrays
layers = []
for L in data["layers"]:
    W = np.array(L["W"], dtype=np.float32).reshape((L["out"], L["in"]))
    b = np.array(L["b"], dtype=np.float32)
    layers.append({
        "W": W, 
        "b": b, 
        "act": L["act"]
    })

# OPTIMIZATION: Reshape Layer 0 for sparse matrix multiplication.
# Instead of multiplying zeros, we just index the active characters.
# Shape becomes: (Hidden Size, Context Length, Vocab Size)
L0_W = layers[0]["W"].reshape(layers[0]["W"].shape[0], context_len, len(vocab))

# ==========================================
# 2. FAST NUMPY FORWARD PASS
# ==========================================
def forward(ids):
    # Layer 0: Sparse lookup (extremely fast)
    # ids is a numpy array of shape (context_len,)
    # We grab only the weights corresponding to the active character IDs and sum them.
    a = L0_W[:, np.arange(context_len), ids].sum(axis=1) + layers[0]["b"]
    a = np.maximum(a, 0)  # ReLU

    # Layers 1 to N
    for i in range(1, len(layers)):
        L = layers[i]
        z = L["W"] @ a + L["b"]
        if L["act"] == "relu":
            a = np.maximum(z, 0)
        else:
            a = z  # Linear output
    return a

# ==========================================
# 3. SAMPLING LOGIC (Temp & Top-P)
# ==========================================
def sample_next(ids, temp=0.8, top_p=0.92):
    logits = forward(ids) / max(temp, 0.01)
    
    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    # Top-P (Nucleus) filtering
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    
    # Keep only the characters within the top_p threshold
    cutoff = np.searchsorted(cumulative_probs, top_p) + 1
    sorted_indices = sorted_indices[:cutoff]
    sorted_probs = sorted_probs[:cutoff]
    
    # Renormalize
    sorted_probs /= np.sum(sorted_probs)
    
    # Draw randomly based on the probabilities
    choice = np.random.choice(sorted_indices, p=sorted_probs)
    return int(choice)

# ==========================================
# 4. LIVE STREAMING TERMINAL UI
# ==========================================
def main():
    # Default arguments
    length = 300
    temp = 0.8
    top_p = 0.92

    # Parse command line arguments
    if len(sys.argv) == 1:
        try:
            prompt = input(">>> ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)
    else:
        prompt = sys.argv[1]
        if len(sys.argv) > 2: length = int(sys.argv[2])
        if len(sys.argv) > 3: temp = float(sys.argv[3])
        if len(sys.argv) > 4: top_p = float(sys.argv[4])

    # Encode prompt
    ids = []
    for ch in prompt:
        if ch in vocab:
            ids.append(vocab.index(ch))

    # Pad or truncate to exactly context_len
    pad_id = vocab.index(' ') if ' ' in vocab else 0
    while len(ids) < context_len:
        ids.insert(0, pad_id)
    if len(ids) > context_len:
        ids = ids[-context_len:]

    context = np.array(ids, dtype=np.int32)

    # Print the seed prompt (gray)
    sys.stdout.write(f"\033[90m{prompt}\033[0m")
    sys.stdout.flush()

    # Generate loop
    try:
        for _ in range(length):
            next_id = sample_next(context, temp, top_p)
            char = vocab[next_id]

            # Print character (green) and flush to force terminal update
            sys.stdout.write(f"\033[92m{char}\033[0m")
            sys.stdout.flush()

            # Slide context window left
            context = np.roll(context, -1)
            context[-1] = next_id

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        pass

    print("\n\n\033[90m[Generation stopped]\033[0m\n")

if __name__ == "__main__":
    main()