# Things I learned

- **Slice 1:** An LLM "writing" is just a loop: predict probabilities for the next token, roll weighted dice, append, repeat. Every model from this bigram to Claude generates text this way; what changes is how good the prediction is.
