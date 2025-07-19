| Member | Assigned Model | Task Responsibility |
|--------|----------------|---------------------|
| **M1 - Anirudhra** | Model 1: Causal Transformer (Rel Pos + Word Emb) | - Build tokenizer and embedding matrix using pretrained word vectors  <br> - Design and implement Transformer encoder with relative positional encoding |
| **M2 - Sumith** | Model 1: Causal Transformer (Rel Pos + Word Emb) | - Train the model on biomedical Q&A pairs <br> - Evaluate using ROUGE-L and BERT-F1 <br> - Visualize attention and analyze interpretability |
| **M3 - Aparna** | Model 2: RAG + FAISS + LangChain/DSPy | - Chunk the biomedical passages <br> - Build FAISS vector store <br> - Integrate with LangChain/DSPy and configure LLM backend (e.g., LLaMA/Gemma) |
| **M4 - Devanshi** | Model 2: RAG + FAISS + LangChain/DSPy | - Implement RAG pipeline with top-10 retrieval <br> - Filter OOD (out-of-domain) questions <br> - Evaluate using MAP, MRR, ROUGE-L, BERT-F1 |
| **M5 - Shabda** | Model 3: Transfer learning on GPT-2/TinyLLaMA + PEFT | - Set up PEFT (LoRA/QLoRA) <br> - Fine-tune on biomedical support dialogues <br> - Handle GPU and checkpoint optimization |
| **M6 - Jeffin** | Model 3: Transfer learning on GPT-2/TinyLLaMA + PEFT | - Evaluate model: ROUGE-L, BERT-F1 <br> - Conduct error analysis & compare with baseline <br> - Provide improvement recommendations |
| **M7 - Stephen** | Model 4: RAG + FAISS + Query Rewriting (Gemma) | - Integrate Gemma-based query rewriting module <br> - Set up FAISS + Retriever <br> - Connect to LLM (Llama/Groq) for final response |
| **M8 - Neha** | Model 4: RAG + FAISS + Query Rewriting (Gemma) | - Ensure model ignores out-of-context queries <br> - Evaluate using MAP, MRR, ROUGE-L, BERT-F1 <br> - Conduct stress testing with OOD prompts |
| **M9 - Arjun** | Model 5: Prompt Engineering with CoT/Few-shot/DSP | - Design few-shot and CoT prompts <br> - Run multiple prompt styles on selected LLM <br> - Compare zero-shot vs few-shot vs DSP |
| **M1–M9 (shared)** | Common Tasks | - Preprocessing: cleaning, splitting passages <br> - Export to .parquet as needed <br> - Final deliverable: notebook → PDF conversion |
