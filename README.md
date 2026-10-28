# NLM v1 (Neural Language Model v1)

NLM v1 is a character-level neural language model that was trained (and ran) entirely in your web browser.

It proves that you don't need a billion parameters and a Transformer architecture to capture the phonetic, structural, and grammatical patterns of English text.

### Model Specs!
*   **Architecture:** Feed-Forward Neural Network
*   **Parameters:** ~148,000
*   **Context Window:** 32 characters
*   **File Size:** ~1.3 MB (in raw JSON)

### What can it do?
At this scale, the model operates as a **lossy compression algorithm** for the training text. 
*   **Memorization:** If given a 20+ character prompt from the training corpus
*   **Generalization:** Because it is forced into a 64-neuron bottleneck, it learns the *taste* of English. When it doesn't know the exact word, it hallucinates brilliant pseudo-English (e.g., `waller`, `bork`, `shere breid`).

### How do you run it?
You need to download `app.sh`, `__main__.py` and `model.json` (use git clone if available) and run `python __main__.py` or `bash app.sh`

### Do you like "Drunk English"?
Exspect the AI to have a "Drunk-like" english due to it being a very small model. Exspect ansers like
> *"the sun rose over the quiet ats meat beat . she frica s were nobt the alock"*
