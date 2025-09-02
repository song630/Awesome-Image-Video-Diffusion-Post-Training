# Awesome-Image-Video-Diffusion-Post-Training [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This awesome repository tracks the fast-moving space of post-training for image & video generation models—from alignment with human preferences to practical deployment on constrained hardware. It emphasizes methods that start from a pretrained generator (DiT/UNet/rectified-flow/AR hybrids) and then improve quality, alignment, controllability, or efficiency without retraining from scratch. You’ll also find surveys/benchmarks that define tasks and metrics for reproducible comparison.

Categories at a glance:

- Benchmarks & Surveys — Datasets, leaderboards, and survey papers that standardize evaluation for diffusion post-training: prompt fidelity, aesthetics, safety, temporal coherence, and editing/personalization.
- Efficiency / Quantization / Distillation — Practical recipes to shrink memory/latency or reduce NFEs: PTQ/QAT (INT/FP low-bit), activation/weight transforms and rotations, vector quantization, caching/token-merging, and student–teacher distillation/consistency for few-step or one-step generation.
- Foundation Model — Base generative models (image/video or unified multimodal) that introduce architectures, training pipelines, or decoding schemes later adapted via post-training.
- LoRA — Parameter-efficient adaptation for customization and editing.
- Preference Optimization — Directly align models to human or proxy preferences without full RL loops: DPO and multi-sample/segment-level variants for finer supervision (e.g., motion segments in video). Typically improves aesthetics, compositionality, and prompt following.
- Reinforcement Learning — Online/offline RL for diffusion/flow models with KL regularization, credit assignment over timesteps, and exploration tricks.
- Reward Model — Learning or leveraging differentiable/non-differentiable reward functions and using them via reward gradients, reward-weighted likelihood, or gradient-based fine-tuning.

## Table of Contents

- [Awesome-Image-Video-Diffusion-Post-Training ](#awesome-image-video-diffusion-post-training-)
  - [Table of Contents](#table-of-contents)
  - [Benchmarks \& Surveys](#benchmarks--surveys)
  - [Efficiency / Quantization / Distillation](#efficiency--quantization--distillation)
  - [Foundation Model](#foundation-model)
  - [Lora](#lora)
  - [Preference Optimization](#preference-optimization)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Reward Model](#reward-model)
  - [Other](#other)
  - [Star History](#star-history)
  - [Citation](#citation)

## Benchmarks & Surveys

| Title | Paper | Code | Venue | Date |
|---|---|---|---|---|
| T2I-ConBench: Text-to-Image Benchmark for Continual Post-training | [Link](https://arxiv.org/abs/2505.16875) |  |  | 2025-05-22 |
| Generative Diffusion Modeling: A Practical Handbook | [Link](https://arxiv.org/abs/2412.17162) |  |  | 2024-12-22 |

## Efficiency / Quantization / Distillation

| Title | Paper | Code | Venue | Date |
|---|---|---|---|---|
| Fewer Denoising Steps or Cheaper Per-Step Inference: Towards Compute-Optimal Diffusion Model Deployment | [Link](https://arxiv.org/abs/2508.06160) |  |  | 2025-08-08 |
| S$^2$Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation | [Link](https://arxiv.org/abs/2508.04016) |  |  | 2025-08-06 |
| LRQ-DiT: Log-Rotation Post-Training Quantization of Diffusion Transformers for Text-to-Image Generation | [Link](https://arxiv.org/abs/2508.03485) |  |  | 2025-08-05 |
| Enhancing Generalization in Data-free Quantization via Mixup-class Prompting | [Link](https://arxiv.org/abs/2507.21947) |  |  | 2025-07-29 |
| SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models | [Link](https://arxiv.org/abs/2507.14811) |  |  | 2025-07-20 |
| DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization | [Link](https://arxiv.org/abs/2507.12933) |  |  | 2025-07-17 |
| Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization | [Link](https://arxiv.org/abs/2506.22463) |  |  | 2025-06-18 |
| HadaNorm: Diffusion Transformer Quantization through Mean-Centered Transformations | [Link](https://arxiv.org/abs/2506.09932) |  |  | 2025-06-11 |
| Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation | [Link](https://arxiv.org/abs/2506.09350) |  |  | 2025-06-11 |
| SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training | [Link](https://arxiv.org/abs/2506.05301) |  |  | 2025-06-05 |
| QuantFace: Low-Bit Post-Training Quantization for One-Step Diffusion Face Restoration | [Link](https://arxiv.org/abs/2506.00820) |  |  | 2025-06-01 |
| Pioneering 4-Bit FP Quantization for Diffusion Models: Mixup-Sign Quantization and Timestep-Aware Fine-Tuning | [Link](https://arxiv.org/abs/2505.21591) |  |  | 2025-05-27 |
| DVD-Quant: Data-free Video Diffusion Transformers Quantization | [Link](https://arxiv.org/abs/2505.18663) |  |  | 2025-05-24 |
| FPQVAR: Floating Point Quantization for Visual Autoregressive Model with FPGA Hardware Co-design | [Link](https://arxiv.org/abs/2505.16335) |  |  | 2025-05-22 |
| Attend to Not Attended: Structure-then-Detail Token Merging for Post-training DiT Acceleration | [Link](https://arxiv.org/abs/2505.11707) |  |  | 2025-05-16 |
| DiTFastAttnV2: Head-wise Attention Compression for Multi-Modality Diffusion Transformers | [Link](https://arxiv.org/abs/2503.22796) |  |  | 2025-03-28 |
| FP4DiT: Towards Effective Floating Point Quantization for Diffusion Transformers | [Link](https://arxiv.org/abs/2503.15465) |  |  | 2025-03-19 |
| Post-Training Quantization for Diffusion Transformer via Hierarchical Timestep Grouping | [Link](https://arxiv.org/abs/2503.06930) |  |  | 2025-03-10 |
| Q&C: When Quantization Meets Cache in Efficient Image Generation | [Link](https://arxiv.org/abs/2503.02508) |  |  | 2025-03-04 |
| Hardware-Friendly Static Quantization Method for Video Diffusion Transformers | [Link](https://arxiv.org/abs/2502.15077) |  |  | 2025-02-20 |
| D$^2$-DPM: Dual Denoising for Quantized Diffusion Probabilistic Models | [Link](https://arxiv.org/abs/2501.08180) |  |  | 2025-01-14 |
| Diffusion Adversarial Post-Training for One-Step Video Generation | [Link](https://arxiv.org/abs/2501.08316) |  |  | 2025-01-14 |
| PQD: Post-training Quantization for Efficient Diffusion Models | [Link](https://arxiv.org/abs/2501.00124) |  |  | 2024-12-30 |
| Data-Free Group-Wise Fully Quantized Winograd Convolution via Learnable Scales | [Link](https://arxiv.org/abs/2412.19867) |  |  | 2024-12-27 |
| TCAQ-DM: Timestep-Channel Adaptive Quantization for Diffusion Models | [Link](https://arxiv.org/abs/2412.16700) |  |  | 2024-12-21 |
| Qua$^2$SeDiMo: Quantifiable Quantization Sensitivity of Diffusion Models | [Link](https://arxiv.org/abs/2412.14628) |  |  | 2024-12-19 |
| Efficiency Meets Fidelity: A Novel Quantization Framework for Stable Diffusion | [Link](https://arxiv.org/abs/2412.06661) |  |  | 2024-12-09 |
| PassionSR: Post-Training Quantization with Adaptive Scale in One-Step Diffusion based Image Super-Resolution | [Link](https://arxiv.org/abs/2411.17106) |  |  | 2024-11-26 |
| Efficient Pruning of Text-to-Image Models: Insights from Pruning Stable Diffusion | [Link](https://arxiv.org/abs/2411.15113) |  |  | 2024-11-22 |
| SVDQunat: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models | [Link](https://arxiv.org/abs/2411.05007) |  |  | 2024-11-07 |
| Error Diffusion: Post Training Quantization with Block-Scaled Number Formats for Neural Networks | [Link](https://arxiv.org/abs/2410.11203) |  |  | 2024-10-15 |
| Token Caching for Diffusion Transformer Acceleration | [Link](https://arxiv.org/abs/2409.18523) |  |  | 2024-09-27 |
| DiTAS: Quantizing Diffusion Transformers via Enhanced Activation Smoothing | [Link](https://arxiv.org/abs/2409.07756) |  |  | 2024-09-12 |
| Accurate Compression of Text-to-Image Diffusion Models via Vector Quantization | [Link](https://arxiv.org/abs/2409.00492) |  |  | 2024-08-31 |
| VQ4DiT: Efficient Post-Training Vector Quantization for Diffusion Transformers | [Link](https://arxiv.org/abs/2408.17131) |  |  | 2024-08-30 |
| Low-Bitwidth Floating Point Quantization for Efficient High-Quality Diffusion Models | [Link](https://arxiv.org/abs/2408.06995) |  |  | 2024-08-13 |
| Vector Quantized Diffusion Model for Text-to-Image Synthesis | [Link](https://arxiv.org/abs/2111.14822) |  |  | 2021-11-29 |

## Foundation Model

| Title | Paper | Code | Venue | Date |
|---|---|---|---|---|
| OmniGen2: Exploration to Advanced Multimodal Generation | [Link](https://arxiv.org/abs/2506.18871) |  |  | 2025-06-23 |
| Seedance 1.0: Exploring the Boundaries of Video Generation Models | [Link](https://arxiv.org/abs/2506.09113) |  |  | 2025-06-10 |
| BLIP3-o: A Family of Fully Open Unified Multimodal Models-Architecture, Training and Dataset | [Link](https://arxiv.org/abs/2505.09568) |  |  | 2025-05-14 |
| Mogao: An Omni Foundation Model for Interleaved Multi-Modal Generation | [Link](https://arxiv.org/abs/2505.05472) |  |  | 2025-05-08 |
| Seedream 3.0 Technical Report | [Link](https://arxiv.org/abs/2504.11346) |  |  | 2025-04-15 |
| ILLUME+: Illuminating Unified MLLM with Dual Visual Tokenization and Diffusion Refinement | [Link](https://arxiv.org/abs/2504.01934) |  |  | 2025-04-02 |
| Seedream 2.0: A Native Chinese-English Bilingual Image Generation Foundation Model | [Link](https://arxiv.org/abs/2503.07703) |  |  | 2025-03-10 |
| Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling | [Link](https://arxiv.org/abs/2501.17811) |  |  | 2025-01-29 |
| ACE++: Instruction-Based Image Creation and Editing via Context-Aware Content Filling | [Link](https://arxiv.org/abs/2501.02487) |  |  | 2025-01-05 |
| Orthus: Autoregressive Interleaved Image-Text Generation with Modality-Specific Heads | [Link](https://arxiv.org/abs/2412.00127) |  |  | 2024-11-28 |
| FiTv2: Scalable and Improved Flexible Vision Transformer for Diffusion Model | [Link](https://arxiv.org/abs/2410.13925) |  |  | 2024-10-17 |
| Emu3: Next-Token Prediction is All You Need | [Link](https://arxiv.org/abs/2409.18869) |  |  | 2024-09-27 |
| VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation | [Link](https://arxiv.org/abs/2409.04429) |  |  | 2024-09-06 |
| Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model | [Link](https://arxiv.org/abs/2408.11039) |  |  | 2024-08-20 |
| Chameleon: Mixed-Modal Early-Fusion Foundation Models | [Link](https://arxiv.org/abs/2405.09818) |  |  | 2024-05-16 |
| Scaling Rectified Flow Transformers for High-Resolution Image Synthesis | [Link](https://arxiv.org/abs/2403.03206) |  |  | 2024-03-05 |
| Generative Multimodal Models are In-Context Learners | [Link](https://arxiv.org/abs/2312.13286) |  |  | 2023-12-20 |
| DreamLLM: Synergistic Multimodal Comprehension and Creation | [Link](https://arxiv.org/abs/2309.11499) |  |  | 2023-09-20 |
| SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis | [Link](https://arxiv.org/abs/2307.01952) |  |  | 2023-07-04 |

## Lora

| Title | Paper | Code | Venue | Date |
|---|---|---|---|---|
| Modular Customization of Diffusion Models via Blockwise-Parameterized Low-Rank Adaptation | [Link](https://arxiv.org/abs/2503.08575) |  |  | 2025-03-11 |
| AdaptSR: Low-Rank Adaptation for Efficient and Scalable Real-World Super-Resolution | [Link](https://arxiv.org/abs/2503.07748) |  |  | 2025-03-10 |
| IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models | [Link](https://arxiv.org/abs/2410.21759) |  |  | 2024-10-29 |

## Preference Optimization

| Title | Paper | Code | Venue | Date |
|---|---|---|---|---|
| Inversion-DPO: Precise and Efficient Post-Training for Diffusion Models | [Link](https://arxiv.org/abs/2507.11554) |  |  | 2025-07-14 |
| RDPO: Real Data Preference Optimization for Physics Consistency Video Generation | [Link](https://arxiv.org/abs/2506.18655) |  |  | 2025-06-23 |
| AlignHuman: Improving Motion and Fidelity via Timestep-Segment Preference Optimization for Audio-Driven Human Animation | [Link](https://arxiv.org/abs/2506.11144) |  |  | 2025-06-11 |
| DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models | [Link](https://arxiv.org/abs/2506.03517) |  |  | 2025-06-04 |
| Diffusion Distillation With Direct Preference Optimization For Efficient 3D LiDAR Scene Completion | [Link](https://arxiv.org/abs/2504.11447) |  |  | 2025-04-15 |
| IPO: Iterative Preference Optimization for Text-to-Video Generation | [Link](https://arxiv.org/abs/2502.02088) |  |  | 2025-02-04 |
| Preference Optimization with Multi-Sample Comparisons | [Link](https://arxiv.org/abs/2410.12138) |  |  | 2024-10-16 |
| Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization | [Link](https://arxiv.org/abs/2406.04314) |  |  | 2024-06-06 |
| Using Human Feedback to Fine-tune Diffusion Models without Any Reward Model | [Link](https://arxiv.org/abs/2311.13231) |  |  | 2023-11-22 |
| Diffusion Model Alignment Using Direct Preference Optimization | [Link](https://arxiv.org/abs/2311.12908) |  |  | 2023-11-21 |

## Reinforcement Learning

| Title | Paper | Code | Venue | Date |
|---|---|---|---|---|
| X-Omni: Reinforcement Learning Makes Discrete Autoregressive Image Generative Models Great Again | [Link](https://arxiv.org/abs/2507.22058v1) |  |  | 2025-07-29 |
| MMaDA: Multimodal Large Diffusion Language Models | [Link](https://arxiv.org/abs/2505.15809) |  |  | 2025-05-21 |
| Flow-GRPO: Training Flow Matching Models via Online RL | [Link](https://arxiv.org/abs/2505.05470) |  |  | 2025-05-08 |
| SkyReels-V2: Infinite-length Film Generative Model | [Link](https://arxiv.org/abs/2504.13074) |  |  | 2025-04-17 |
| Towards Better Alignment: Training Diffusion Models with Reinforcement Learning Against Sparse Rewards | [Link](https://arxiv.org/abs/2503.11240) |  |  | 2025-03-14 |
| A Simple and Effective Reinforcement Learning Method for Text-to-Image Diffusion Fine-tuning | [Link](https://arxiv.org/abs/2503.00897) |  |  | 2025-03-02 |
| DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models | [Link](https://arxiv.org/abs/2305.16381) |  |  | 2023-05-25 |
| Training Diffusion Models with Reinforcement Learning | [Link](https://arxiv.org/abs/2305.13301) |  |  | 2023-05-22 |

## Reward Model

| Title | Paper | Code | Venue | Date |
|---|---|---|---|---|
| Rewards Are Enough for Fast Photo-Realistic Text-to-image Generation | [Link](https://arxiv.org/abs/2503.13070) |  |  | 2025-03-17 |
| Harness Local Rewards for Global Benefits: Effective Text-to-Video Generation Alignment with Patch-level Reward Models | [Link](https://arxiv.org/abs/2502.06812) |  |  | 2025-02-04 |
| T2V-Turbo-v2: Enhancing Video Generation Model Post-Training through Data, Reward, and Conditional Guidance Design | [Link](https://arxiv.org/abs/2410.05677) |  |  | 2024-10-08 |
| Video Diffusion Alignment via Reward Gradients | [Link](https://arxiv.org/abs/2407.08737) |  |  | 2024-07-11 |
| Directly Fine-Tuning Diffusion Models on Differentiable Rewards | [Link](https://arxiv.org/abs/2309.17400) |  |  | 2023-09-29 |
| RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment | [Link](https://arxiv.org/abs/2304.06767) |  |  | 2023-04-13 |
| Aligning Text-to-Image Models using Human Feedback | [Link](https://arxiv.org/abs/2302.12192) |  |  | 2023-02-23 |

## Other

| Title | Paper | Code | Venue | Date |
|---|---|---|---|---|
| Enhancing Scene Transition Awareness in Video Generation via Post-Training | [Link](https://arxiv.org/abs/2507.18046) |  |  | 2025-07-24 |
| Efficient and Unbiased Sampling from Boltzmann Distributions via Variance-Tuned Diffusion Models | [Link](https://arxiv.org/abs/2505.21005) |  |  | 2025-05-27 |
| PISA Experiments: Exploring Physics Post-Training for Video Diffusion Models by Watching Stuff Drop | [Link](https://arxiv.org/abs/2503.09595) |  |  | 2025-03-12 |
| Spend Wisely: Maximizing Post-Training Gains in Iterative Synthetic Data Boostrapping | [Link](https://arxiv.org/abs/2501.18962) |  |  | 2025-01-31 |
| HelloMeme: Integrating Spatial Knitting Attentions to Embed High-Level and Fidelity-Rich Conditions in Diffusion Models | [Link](https://arxiv.org/abs/2410.22901) |  |  | 2024-10-30 |
| Pixel-Space Post-Training of Latent Diffusion Models | [Link](https://arxiv.org/abs/2409.17565) |  |  | 2024-09-26 |
| ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment | [Link](https://arxiv.org/abs/2403.05135) |  |  | 2024-03-08 |
| Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation | [Link](https://arxiv.org/abs/2402.10210) |  |  | 2024-02-15 |

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=song630/Awesome-Image-Video-Diffusion-Post-Training&type=Date)](https://www.star-history.com/#song630/Awesome-Image-Video-Diffusion-Post-Training&Date)

## Citation

```bibtex
@misc{song2025imagevideogenposttraining,
  title={Awesome-Image-Video-Diffusion-Post-Training},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/song630/Awesome-Image-Video-Diffusion-Post-Training}},
}
```
