# 🤖 Gen AI Learnings

This repository contains a series of Notebooks and zip files implementing foundational and advanced Concepts of 🧠 Gen AI models and 🤖 Agentic AI workflows along with 🚀 FastAPI (as Backend API), 🖥️ Streamlit (as Frontend UI) and 📦 Docker (for deployment). Few concepts are done from scratch (using PyTorch and NumPy) and few are mini projects for practice.

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

### `RLHF_Scratch.ipynb`
This repository demonstrates a complete Reinforcement Learning from Human Feedback (RLHF) pipeline applied to a language model, showcasing the training and alignment of a base language model through several stages. We begin with a pre-trained causal language model (DistilGPT2) and first fine-tune it using Supervised Fine-Tuning (SFT) on the OpenAssistant/oasst1 dataset, where prompt-response pairs are extracted to teach the model to produce helpful assistant-like replies. The next phase involves reward modeling, where a reward model (OpenAssistant's DeBERTa-v3) is used to evaluate the quality of responses based on human-preference-aligned judgment. Using this reward model, we apply Proximal Policy Optimization (PPO), a policy-gradient reinforcement learning algorithm, to further optimize the SFT model so that it generates outputs that receive higher reward scores. We also explore Direct Preference Optimization (DPO), a simpler yet effective method that aligns the model directly to human preferences using paired comparisons between “chosen” and “rejected” responses, bypassing the need for an explicit reward model. RLAF (Reinforcement Learning with Attribute Feedback), which goes beyond scalar feedback by optimizing across multiple attributes (such as helpfulness, factuality, and toxicity). This phase integrates multiple reward signals to train the model using a multi-objective version of PPO, enabling more nuanced alignment. We train (N) reward models to get the (N) sized Reward vector.

---

### `BERT_NER_Fine_Tuning.ipynb`
This notebook consists of making use of a Base BERT Model and Fine tuning the model with a NER Dataset consists of various NER Tags. LoRA fine tuning is performed in this task. BERT is a Encoder Only model used for various tasks like Q&A, MLM, NSP, NER, Sentence Classification and many more. It is trained using left and right contexts. NER (Named Entity Recognition) is a token based task where each token is given a particulat NER Tag depending on it's context like Person / Organisation / Location or any other.

---

### `rag-scienceqa.ipynb`
This project implements a complete Retrieval-Augmented Generation (RAG) pipeline for the ScienceQA dataset, designed to answer science-related questions with high accuracy. It uses a fine-tuned MiniLM model to embed scientific documents, which are stored in ChromaDB for efficient retrieval. Given a user query, the system retrieves top-k relevant contexts and reranks them using a cross-encoder to ensure semantic relevance. A T5 model is then used to generate the final answer based on the reranked context and the original question. This end-to-end system showcases how semantic retrieval and generative models can be integrated for domain-specific question answering, making it ideal for educational tools and factual information systems.

---

### `GPT2_Finetuned_Math_Q&A_Dockered.txt`

#### GPT-2 Fine-Tuned on GSM8K for Math Question Answering

This project demonstrates how to fine-tune a GPT-2 model on the GSM8K dataset for solving grade-school level math problems. It includes:

- ✅ A fine-tuned GPT-2 model using OpenAI's GSM8K dataset.
- 🚀 A FastAPI backend to handle inference requests from clients.
- 🎯 A Streamlit frontend that provides an interactive UI to test the model's predictions.
- 🐳 Full Dockerization for portability and ease of deployment.
- 🔗 Available on Docker Hub: `rishit89/gpt2_mathqa`

#### Project Structure

- `gpt2-lora-gsm8k-merged`: Saved fine-tuned GPT-2 model.
- `app.py`: FastAPI server to serve the model via REST API.
- `frontend.py`: Streamlit interface to interact with the model.
- `Dockerfile`: Instructions to build the Docker image.
- `supervisord.conf`: Used to run both FastAPI and Streamlit simultaneously inside Docker.

---

### `Weather_Agent.zip`
This basic project deals with the use of Agent to fetech the weather from a specific location through API Call as a tool. This project is built till end i.e. deployment. It uses FastAPI for API integration with Streamlit as a UI (client). This is also dockerized but not yet pushed to the Docker Hub.

Key Learnings:
- Usage of Langchain Agents
- Usage of external APIs as tools which agent uses
- FastAPI Integeration with a UI PLatform

---

### `Agent_using_MCP.zip`
This agent is built to get familiar with the usage of MCP Tools by using already built respective MCP Servers. It uses "qwen-qwq-32b" as an LLM to reason and complete it's goal. The MCP Tools used are "playwright" for webpage interaction, "arxiv" for fetching research papers, "airbnb" to interact with hotels, "weather" to get the weather. All these are tested in Cursor IDE (a very good IDE to interact with MCP Agents).

---

### `chatbot_langgraph.ipynb`
This Chatbot is built completely using Langgraph (a framework for building stateful, multi-step reasoning workflows with Large Language Models (LLMs)). Tools used in this project are Arxiv, Wikipedia and Tavily Search. This chatbot makes use of 2 LLMs for reasoning. The aim of this chatbot is to get the Research Papers, Wikipedia content and general web searching using Tavily.

Flow of the ChatBot:
- LLM 1: "Qwen-qwq-32b" from GROQ attached to those 3 tools which is used for feteching the content by interacting with tool APIs.
- LLM 2: "gemma2-9b-it" from GROQ which is used as a summarizer to summarize the large content generated by LLM 1.
- Finally the StateGraph is invoked with a HumanMessage.

---

### `Coding_Agent.zip`
This project deals with creation of a coding agent by using a base model "llama-3.1-8b-instant" using GROQ API. This agent takes in the input like a Coding Question and generates a code for that and also saves that code locally inside the server. Then the LLM takes inputs and uses the saved code to run on those inputs using subprocess and gives out the outputs for those respective inputs. The server along with FastAPI runs in a docker and the client code i.e. the streamlit one runs in another docker. Now both these dockers are composed together with docker 2 depending on docker 1 using a .yml file. 

---

### `Coding_Agent_with_MCP.zip`
This project also deals with creation of a coding agent by using a base model "llama-3.1-8b-instant" using GROQ API. This agent takes in the input like a Coding Question and generates a code for that and also saves that code locally inside the server but these all are done using MCP (Model Context Protocol). Here the MCP defines 2 tools, one for code generation and one for running the code. Then the LLM takes inputs and uses the saved code to run on those inputs using subprocess and gives out the outputs for those respective inputs. The server along with FastAPI runs in a docker and the client code i.e. the streamlit one runs in another docker. Now both these dockers are composed together with docker 2 depending on docker 1 using a .yml file.

---

### `Medical_Agent_CrewAI.zip`
This mini-project demonstrates how to read PDF medical documents, analyze lab reports, symptoms and diagnosis using a powerful LLM, and scrape medical websites to recommend accurate prescriptions. It uses CrewAI — a collaborative multi-agent AI framework — and the llama-3.1-8b-instant model from Groq for reasoning and summarization.

What It Does
- Reads and parses medical reports (PDF) using CrewAI Tool
- Scrapes a trusted medical website for prescription recommendations using CrewAI Tool
- Summarizes each case using LLM
- Uses multiple CrewAI agents (e.g., summarizer, recommender)
- Writes the generated summary and prescription

---

### `LLM_with_MongoDB.zip`
This zip consists of an LLM interacting with MongoDB for data access and retrieval. Patient's data including (name, age, symptoms, diagnosis) are stored in MongoDB (a Document based NoSQL Database) in the form of JSON. The LLM tries to fetect the data from the database and understance the symptoms and diagnosis with a suitable prompt and generates a summary and prescription (medicine names) used to treat the patient. These details (summary, prescription) get updated in the MongoDB once these are generated by LLM.

---

### `Speculative_Decoding.ipynb`
This notebook demonstrates speculative decoding, an efficient inference technique to accelerate large language models (LLMs) by using a smaller "draft" model to propose tokens, which are then selectively validated by a larger "target" model.

What I did:
- Used GPT-2 XL as the large model (big_model)
- Used DistilGPT-2 as the small assistant model (small_model)
- Implemented two approaches: From scratch and inbuilt Speculative Decoding
- Inference (Inbuilt): Over 38% Speedup is seen.

---
