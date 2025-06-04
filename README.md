# Gen AI Learnings

This repository contains a series of Notebooks implementing foundational and advanced Concepts of Gen AI models from scratch using NumPy and PyTorch.

---

## 📁 Contents

### `CBOW.ipynb`
Implements the Continuous Bag of Words (CBOW) model from scratch using NumPy. It demonstrates word embedding used as input for LLMs.

- Techniques: One-hot encoding, Embedding lookup, Dot products, Softmax
- Usage: Learning word representations in a simple context window setting

---

### `Image_Captioning_with_Attention.ipynb`
Implementation of Attention mechanism and using it in Image Captioning Task. Attention in large language models (LLMs) is the core mechanism that allows the model to focus on different parts of the input when producing an output. Image Captioning with Attention is a technique in deep learning where a model generates natural language descriptions for images while selectively focusing on specific parts of the image at each step of the caption generation. Rather than looking at the entire image equally when predicting each word, the model attends to different regions of the image dynamically. Usage of CNN Encoder (VGG-19) + Bahdanau Attention + LSTM/GRU (Decoder).

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

---

### `LoRA_Peft_Finetuning.ipynb`
Implements LoRA Finetuning using inbuilt modules form PEFT Library. Once LoRA is applied, GPTQ Quanitsation is performed and the quantised model is tested again for inference.

- Use case: Fine tuning a model with freezing the pre-trained weights.

---

### `VLLM_Inference.ipynb`
This notebook makes use of VLLM interface to make an inference on Facebook - opt & Falcon-rw model
- Use Case: Memory Optimisation through Paged Attention which is highly useful for Inference.

---

### `Quantisation_Techniques.ipynb`
This notebook consists od various Quantisation Techniques like Post Training Quantisations like QLoRA (Quantised LoRA), GPTQ (Generalized Post-Training Quantization), AWQ (Activation-Aware Weight Quantization) which are highly helpful duirng model infernce which reduces the memory footprint for storing the model weights during model inference or any other tasks done locally.

---

### `BERT_NER_Fine_Tuning.ipynb`
This notebook consists of making use of a Base BERT Model and Fine tuning the model with a NER Dataset consists of various NER Tags. LoRA fine tuning is performed in this task. BERT is a Encoder Only model used for various tasks like Q&A, MLM, NSP, NER, Sentence Classification and many more. It is trained using left and right contexts. NER (Named Entity Recognition) is a token based task where each token is given a particulat NER Tag depending on it's context like Person / Organisation / Location or any other.

---
