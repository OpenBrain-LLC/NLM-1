# NLM-1 (Neural Language Model - 1)

NLM-1 is a character-level neural language model that was trained (and ran) entirely in your web browser.

It proves that you don't need a billion parameters and a Transformer architecture to capture the phonetic, structural, and grammatical patterns of English text.

### Model Specs
*   **Architecture:** Feed-Forward Neural Network
*   **Parameters:** ~148,00
*   **Context Window:** 32 characters
*   **File Size:** ~1.3 MB (in raw JSON)

### What can it do?
At this scale, the model operates as a **lossy compression algorithm** for the training text. 
*   **Memorization:** If given a 20+ character prompt from the training corpus, it will generate the exact subsequent text with zero errors.
*   **Generalization:** Because it is forced into a 64-neuron bottleneck, it learns the *taste* of English. When it doesn't know the exact word, it hallucinates brilliant pseudo-English (e.g., `waller`, `bork`, `shere breid`).

### How to Run
You need to download `app.py` and `model.json` (use git clone if avalable) and run `python app.py`

### The "Drunk English" Phenomenon
Exspect the AI to have a "Drunk-like" english due to it being a very small model. Exspect ansers like
> *"the sun rose over the quiet ats meat beat . she frica s were nobt the alock"*