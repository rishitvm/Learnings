# Gen AI Learnings

This repository contains a series of Jupyter Notebooks implementing foundational and advanced Gen AI models from scratch using NumPy and PyTorch.

---

## 📁 Contents

### `CBOW.ipynb`
Implements the Continuous Bag of Words (CBOW) model from scratch using NumPy. It demonstrates word embedding used as input for LLMs.

- Techniques: One-hot encoding, Embedding lookup, Dot products, Softmax
- Usage: Learning word representations in a simple context window setting

---

### `Transformer_Scratch.ipynb`
A full NumPy-based implementation of the original Transformer architecture.

- Modules: Multi-head attention, Positional encoding, LayerNorm, FeedForward
- Framework: Pure NumPy (no PyTorch/TensorFlow)

---

### `Llama_Scratch.ipynb`
Builds a LLaMA3 2B decoder-only transformer model from scratch according to it's architecture.

- Includes: Grouped Query Attention, Rotary Positional Embeddings (RoPE), RMSNorm, and SwiGLU activation and FFNs
- Used for text generation, text summarization

---

### `LoRA_Fine_Tuning.ipynb`
Fine-tunes a small GPT-style model using Low-Rank Adaptation (LoRA) from scratch (manual implementation without libraries like PEFT).
Normal text generation is fine tuned to text generation in jokes style.

- Components:
  - LoRA modules injected into attention layers
  - Frozen base model weights
  - Training loop using a small text dataset
- Artifact: `lora_adapters.pt` (saved LoRA weights)

---

### `Multi_Latent_Attention.ipynb`
Implements the Multi-Latent Attention (MLA) mechanism, a recent transformer improvement using grouped latent queries.
Used in DeepSeek models.

- Benefit: Reduces computation while maintaining global context
- Similar in spirit to Perceiver or GQA

---

### `Sliding_Window_Attention.ipynb`
Demonstrates local attention using a sliding window over tokens.

- Use case: Long-sequence modeling with reduced complexity
- Similar to: Longformer-style models
