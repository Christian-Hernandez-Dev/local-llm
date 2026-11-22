# Things I learned

- **Slice 1:** An LLM "writing" is just a loop: predict probabilities for the next token, roll weighted dice, append, repeat. Every model from this bigram to Claude generates text this way; what changes is how good the prediction is.
- **Slice 2:** Self-attention is "each token asks a question, earlier tokens answer, good matches get listened to more". A causal mask hides future tokens so the model can't read the answer during training. Stacking blocks of attend → think (feed-forward), with residual "add, don't replace" connections, is the whole transformer.
