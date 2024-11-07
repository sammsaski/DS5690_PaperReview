# DS5690_PaperReview
A review of [Extending Content Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) by <br/>
Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian.

**Citation.**
S. Chen, S. Wong, L. Chen, and Y. Tian, “Extending context window of
large language models via positional interpolation,” 2023.

# Rubric
- [ ] [2] Technical
- [ ] [5] Presentation (15 minutes or less)
- [ ] [10] Presentation Materials (README format + screen clear enough)
- [ ] [15] Overview (5-minute overview providing context, stating problem, characterizing approach, brief account of how problem was addressed)
- [ ] [8] Question 1 (prepare and deliver a question to the audience)
- [ ] [5] Question 2 (prepare and deliver a question to the audience)
- [ ] [15] Architecture overview (pseudocode description of model or approach)
- [ ] [15] Critical Analysis (what was overlooked? what could have been developed further? were there any errors?)
- [ ] [15] Impacts (what was the impact of the work?)\
- [ ] [5] Resource links
- [ ] [10] Code demonstration
- [ ] [5] Repo
- [ ] [2] Citation

Maximum possible points: 112

---

# Overview

The proposed method, Position Interpolation (PI), extends the context window sizes of RoPE-bsed pretrained LLMs such as LLaMA models to up to 32768 with minimal fine-tuning (within 1000 steps), while demonstrating strong empirical results on various tasks that require long context, including passkey retrieval, language modeling, and long document summarization from LLaMA 7B to 65B. The method also adequately preserves model performance on the original context laength (2048). 

Position Interpolation linearly down-scales the input position indices to match the original context window size, rather than extrapolating beyond the trained context length, which may lead to catastrophically high attention scores that completely ruin the self-attention mechanism.

The problem: can we extend the context window of an existing pre-trained LLM?

Naively, we could try fine-tuning an existing pre-trained Transformer with a lnoger context window. Empirically, however, the authors found that models trained this way adapt to long context windows very slowly. Table 4 shows that training this way for long periods of time (> 10000 batches) resulted in only an effective context window length increase from 2048 -> 2560, whereas PI could do it up to 32768 in less than 1000 batches.

Position Interpolation: instead of extrapolation, directly down-scale the position indices so that the maximum position index matches the previous context window limit in the pre-training stage.

# Architecture

# Critical Analysis

# Impacts

# Resources to directly help understand Position Interpolation better
1. https://github.com/ggerganov/llama.cpp/discussions/1965 - official discussion by Meta engineers/researchers.
2. https://kaiokendev.github.io/til#extending-context-to-8k - initial proof of concept
3. https://kaiokendev.github.io/context - a longer write up of (2)
4. https://www.youtube.com/watch?v=oyXdmtHgZFw&ab_channel=GabrielMongaras - a helpful paper review

# Resources to help understand background, motivation, and related works of Position Interpolation better
1. https://arxiv.org/abs/2104.09864 - RoPE paper
2. https://www.llama.com/ - LLaMA
3. https://arxiv.org/abs/2108.12409 - ALiBi
4. https://arxiv.org/abs/2212.10554 - LeX
