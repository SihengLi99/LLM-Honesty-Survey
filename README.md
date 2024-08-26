# A Survey on the Honesty of Large Language Models

> **[A Survey on the Honesty of Large Language Models](todo)**</br>
Siheng Li<sup>1</sup>, Cheng Yang<sup>1</sup>, Taiqiang Wu<sup>2</sup>, 
Chufan Shi<sup>3</sup>, Yuji Zhang<sup>4</sup>, Xinyu Zhu<sup>3</sup>, Zesen Cheng<sup>5</sup>, Deng Cai,
Lemao Liu<sup>6</sup>, Mo Yu<sup>6</sup>, Yujiu Yang<sup>3</sup>, Ngai Wong<sup>2</sup>, Xixin Wu<sup>1</sup>, Wai Lam<sup>1</sup>
</br><sup>1</sup>The Chinese University of Hong Kong,<sup>2</sup>The University of Hong Kong,
<sup>3</sup>Tsinghua University, <sup>4</sup>The Hong Kong Polytechnic University,
<sup>5</sup>Peking University, <sup>6</sup>WeChat AI

## What is This Survey About?



## Table of Content
- [What is This Survey About?](#-what-is-this-survey-about)
- [Honesty in LLMs](#-honesty-in-llms)
  - [What is Honesty in LLMs](#what-is-honesty-in-llms)
    - [Self-knowledge](#self-knowledge)
    - [Self-consistency](#self-consistency)
  - [Relation with Related Problems](#relation-with-related-problems)
    - [Hallucination](#hallucination)
    - [Unfaithful Chain-of-Thought](#unfaithful-chain-of-thought)
    - [Prompt Robustness](#prompt-robustness)
    - [Generator-Validator Inconsistency](#generator-validator-inconsistency)
    - [Over-Confidence](#over-confidence)
    - [Sycophancy](#sycophancy)
  - [Related Applications](#related-applications)
    - [Hallucination Detection and Mitigation](#hallucination-detection-and-mitigation)
    - [Retrieval Augmented Generation](#retrieval-augmented-generation)
    - [Language Agent](#language-agent)
    - [Model Cascading](#model-cascading)
- [Evaluation of LLM Honesty](#evaluation-of-llm-honesty)
  - [Self-knowledge](#self-knowledge-1)
    - [Recognition of Known/Unknown](#recognition-of-knownunknown)
    - [Calibration](#calibration)
    - [Selective Prediction](#selective-prediction)
  - [Self-consistency](#self-consistency-1)
    - [Reference-based Evaluation](#reference-based-evaluation)
    - [Reference-free Evaluation](#reference-free-evaluation)
- [Methods of Self-knowledge](#methods-of-self-knowledge)
  - [Training-free Approaches](#training-free-approaches)
    - [Predictive Probability](#predictive-probability)
    - [Prompting](#prompting)
    - [Sampling and Aggregation](#sampling-and-aggregation)
  - [Training-based Approaches](#training-based-approaches)
    - [Supervised Fine-tuning](#supervised-fine-tuning)
    - [Reinforcement Learning](#reinforcement-learning)
    - [Probing](#probing)
- [Methods of Self-consistency](#methods-of-self-consistency)
  - [Training-free Approaches](#training-free-approaches-1)
    - [Chain-of-thought Prompting](#chain-of-thought-prompting)
    - [Decoding-time Intervention](#decoding-time-intervention)
    - [Sampling and Aggregation](#sampling-and-aggregation-1)
    - [Post-generation Revision](#post-generation-revision)
  - [Training-based Approaches](#training-based-approaches-1)
    - [Self-aware Fine-tuning](#self-aware-fine-tuning)
    - [Self-supervised Fine-tuning](#self-supervised-fine-tuning)
- [Future Work](#future-work)


## Honesty in LLMs
### What is Honesty in LLMs
#### Self-knowledge

#### Self-consistency

### Relation with Related Problems
#### Hallucination

- Survey of hallucination in natural language generation, <ins>ACM Computing Surveys, 2023</ins> [[Paper](https://dl.acm.org/doi/10.1145/3571730)]
- Siren’s song in the AI ocean: a survey on hallucination in large language models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.01219)][[Code](https://github.com/HillZhang1999/llm-hallucination-survey)]
- A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2311.05232)][[Code](https://github.com/LuckyyySTA/Awesome-LLM-hallucination)]
- Knowledge overshadowing causes amalgamated hallucination in large language models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2407.08039v1)]
- LLM internal states reveal hallucination risk faced with a query, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2407.03282v1)]
- Learning to trust your feelings: Leveraging self-awareness in LLMs for hallucination mitigation, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2401.15449)][[Code](https://github.com/liangyuxin42/dreamcatcher)]
- How language model hallucinations can snowball, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.13534)][[Code](https://github.com/Nanami18/Snowballed_Hallucination)]
- Insights into LLM long-context failures: When transformers know but don’t tell, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2406.14673)][[Code](https://github.com/TaiMingLu/know-dont-tell)]
- The internal state of an LLM knows when it's lying, <ins>EMNLP Findings, 2023</ins> [[Paper](https://aclanthology.org/2023.findings-emnlp.68/)]

#### Unfaithful Chain-of-Thought
- Language models don’t always say what they think: unfaithful explanations in chain-of-thought prompting, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2305.04388)][[Code](https://github.com/milesaturpin/cot-unfaithfulness)]
- Measuring faithfulness in chain-of-thought reasoning, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.13702)]
- On the hardness of faithful chain-of-thought reasoning in large language models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2406.10625)]
- Knowledge overshadowing causes amalgamated hallucination in large language models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2407.08039)]


#### Prompt Robustness
- State of what art? A call for multi-prompt LLM evaluation, <ins>TACL, 2024</ins> [[Paper](https://arxiv.org/pdf/2401.00595)][[Code](https://github.com/SLAB-NLP/Multi-Prompt-LLM-Evaluation)]
- Large language models can be easily distracted by irrelevant context, <ins>ICML, 2023</ins> [[Paper](http://arxiv.org/abs/2302.00093)][[Code](https://github.com/google-research-datasets/GSM-IC)]
- On the robustness of ChatGPT: An adversarial and out-of-distribution perspective, <ins>ICLR Workshop, 2023</ins> [[Paper](https://arxiv.org/abs/2302.12095)][[Code](https://github.com/microsoft/robustlearn)]
- Quantifying language models’ sensitivity to spurious features in prompt design or: How I learned to start worrying about prompt formatting, <ins>ICLR, 2024</ins> [[Paper](http://arxiv.org/abs/2310.11324)][[Code](https://github.com/msclar/formatspread)]

#### Generator-Validator Inconsistency
- Benchmarking and improving generator-validator consistency of language models, <ins>ICLR, 2024</ins> [[Paper](http://arxiv.org/abs/2310.01846)][[Code](https://github.com/XiangLi1999/GV-consistency)]
- Active retrieval augmented generation, <ins>EMNLP, 2023</ins> [[Paper](http://arxiv.org/abs/2305.15717)][[Code](https://github.com/jzbjyb/FLARE)]

#### Over-Confidence
- Sayself: Teaching LLMs to express confidence with self-reflective rationales, <ins>arXiv, 2024b</ins> [[Paper](http://arxiv.org/abs/2405.20974)][[Code](https://github.com/xu1868/SaySelf)]
- Reducing conversational agents’ overconfidence through linguistic calibration, <ins>TACL, 2022</ins> [[Paper](http://arxiv.org/abs/2012.14983)]
- Deductive verification of chain-of-thought reasoning, <ins>Advances in Neural Information Processing Systems, 2023</ins> [[Paper](https://arxiv.org/abs/2306.03872)][[Code](https://github.com/lz1oceani/verify_cot)]
- Relying on the unreliable: The impact of language models’ reluctance to express uncertainty, <ins>ACL, 2024</ins> [[Paper](http://arxiv.org/abs/2401.06730)][[Code](https://arxiv.org/pdf/2401.06730)]
- Think twice before assure: Confidence estimation for large language models through reflection on multiple answers, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2403.09972)]


#### Sycophancy
- Discovering language model behaviors with model-written evaluations, <ins>ACL Findings, 2023</ins> [[Paper](https://aclanthology.org/2023.findings-acl.847/)][[Code](https://github.com/anthropics/evals)]
- Question decomposition improves the faithfulness of model-generated reasoning, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.11768)][[Code](https://github.com/anthropics/DecompositionFaithfulnessPaper)]
- Can ChatGPT defend its belief in truth? Evaluating LLM reasoning via debate, <ins>EMNLP Findings, 2023</ins> [[Paper](https://aclanthology.org/2023.findings-emnlp.795/)][[Code](https://github.com/%20OSU-NLP-Group/Auto-Dialectical-Evaluation)]
- Debating with more persuasive LLMs leads to more truthful answers, <ins>ICML, 2024</ins> [[Paper](https://arxiv.org/abs/2402.06782)][[Code](https://github.com/ucl-dark/llm_debate)]
- AI deception: A survey of examples, risks, and potential solutions, <ins>Patterns, 2024</ins> [[Paper](http://arxiv.org/abs/2308.14752)]

### Related Applications
#### Hallucination Detection and Mitigation
- SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models, <ins>EMNLP, 2023</ins> [[Paper](https://arxiv.org/abs/2303.08896)][[Code](https://github.com/potsawee/selfcheckgpt)]
- Fact-checking the output of large language models via token-level uncertainty quantification, <ins>ACL Findings, 2024</ins> [[Paper](https://arxiv.org/abs/2403.04696)][[Code](https://github.com/IINemo/lm-polygraph)]
- Knowledge verification to nip hallucination in the bud, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2401.10768)][[Code](https://github.com/fanqiwan/KCA)]
- Learning to trust your feelings: Leveraging self-awareness in LLMs for hallucination mitigation, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2401.15449)][[Code](https://github.com/liangyuxin42/dreamcatcher)]
- Teaching large language models to express knowledge boundary from their own signals, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2406.10881)]

#### Retrieval Augmented Generation
- Self-knowledge guided retrieval augmentation for large language models, <ins>EMNLP Findings, 2023</ins> [[Paper](https://aclanthology.org/2023.findings-emnlp.691)][[Code](https://github.com/THUNLP-MT/SKR)]
- Unified active retrieval for retrieval augmented generation, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2406.12534)][[Code](https://github.com/xiami2019/UAR)]
- Active retrieval augmented generation, <ins>EMNLP, 2023</ins> [[Paper](https://aclanthology.org/2023.emnlp-main.495/)][[Code](https://github.com/jzbjyb/FLARE)]
- When do LLMs need retrieval augmentation? Mitigating LLMs’ overconfidence helps retrieval augmentation, <ins>ACL Findings, 2024</ins> [[Paper](https://aclanthology.org/2024.findings-acl.675/)][[Code](https://github.com/ShiyuNee/When-to-Retrieve)]
- CTRL-A: Adaptive retrieval-augmented generation via probe-guided control, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2405.18727)][[Code](https://github.com/HSLiu-Initial/CtrlA.git)]
- SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2406.19215)][[Code](https://github.com/THU-KEG/SeaKR)]
- RA-ISF: Learning to answer and understand from retrieval augmentation via iterative self-feedback, <ins>ACL Findings, 2024</ins> [[Paper](https://aclanthology.org/2024.findings-acl.281/)][[Code](https://github.com/OceannTwT/ra-isf)]

#### Language Agent
- Introspective planning: Guiding language-enabled agents to refine their own uncertainty, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.06529v2)]
- Towards robots that know when they need help: Affordance-based uncertainty for large language model planners, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.13198)]
- Towards uncertainty-aware language agent, <ins>ACL Findings, 2024</ins> [[Paper](https://aclanthology.org/2024.findings-acl.398/)][[Code](https://github.com/Jiuzhouh/Uncertainty-Aware-Language-Agent)]

#### Model Cascading
- FrugalGPT: How to use large language models while reducing cost and improving performance, <ins>arXiv, 2023b</ins> [[Paper](https://arxiv.org/abs/2305.05176)][[Code](https://github.com/stanford-futuredata/FrugalGPT)]
- Large language model cascades with mixture of thoughts representations for cost-efficient reasoning, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2310.03094)][[Code](https://github.com/MurongYue/LLM_MoT_cascade)]
- Trust or escalate: LLM judges with provable guarantees for human agreement, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2407.18370)]
- Language model cascades: Token-level uncertainty and beyond, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2404.10136)]
- Cascade-aware training of language models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2406.00060)]
- C<sup>3</sup>: Confidence calibration model cascade for inference-efficient cross-lingual natural language understanding, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.15991)]


### Evaluation of LLM Honesty
#### Self-knowledge
##### Recognition of Known/Unknown
- Do large language models know what they don’t know?, <ins>ACL Findings, 2023</ins> [[Paper](http://arxiv.org/abs/2305.18153)][[Code](https://github.com/yinzhangyue/SelfAware)]
- Knowledge of knowledge: Exploring known-unknowns uncertainty with large language models, <ins>arXiv, 2023</ins> [[Paper](http://arxiv.org/abs/2305.13712)][[Code](https://github.com/amayuelas/knowledge-of-knowledge)]
- Learning to trust your feelings: Leveraging self-awareness in LLMs for hallucination mitigation, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2401.15449)][[Code](https://github.com/liangyuxin42/dreamcatcher)]
- Examining LLMs’ uncertainty expression towards questions outside parametric knowledge, <ins>arXiv, 2024a</ins> [[Paper](http://arxiv.org/abs/2311.09731)][[Code](https://github.com/genglinliu/UnknownBench)]
- The best of both worlds: Toward an honest and helpful large language model, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2406.00380)][[Code](https://github.com/Flossiee/HonestyLLM)]
- Behonest: Benchmarking honesty of large language models, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2406.13261)][[Code](https://github.com/GAIR-NLP/BeHonest)]

##### Calibration
- On calibration of modern neural networks, <ins>ICML, 2017</ins> [[Paper](https://arxiv.org/abs/1706.04599)]
- Measuring calibration in deep learning, <ins>CVPR Workshops, 2019</ins> [[Paper](https://arxiv.org/abs/1904.01685)]
- Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration, <ins>NeurIPS, 2019</ins> [[Paper](https://arxiv.org/abs/1905.11001)][[Code](https://dirichletcal.github.io/)]
- A survey of confidence estimation and calibration in large language models, <ins>NAACL, 2024</ins> [[Paper](https://arxiv.org/abs/2311.08298)]
- Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2306.13063)][[Code](https://github.com/MiaoXiong2320/llm-uncertainty)]
- Calibrating large language models with sample consistency, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.13904)]

##### Selective Prediction
- Uncertainty-based abstention in LLMs improves safety and reduces hallucinations, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2404.10960)]
- Sayself: Teaching LLMs to express confidence with self-reflective rationales, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2405.20974)][[Code](https://github.com/xu1868/SaySelf)]
- Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs, <ins>ICLR, 2024</ins> [[Paper](http://arxiv.org/abs/2306.13063)][[Code](https://github.com/MiaoXiong2320/llm-uncertainty)]
- A survey of confidence estimation and calibration in large language models, <ins>NAACL, 2024</ins> [[Paper](http://arxiv.org/abs/2311.08298)]
- Factual confidence of LLMs: On reliability and robustness of current estimators, <ins>ACL, 2024</ins> [[Paper](https://aclanthology.org/2024.acl-long.250/)][[Code](https://github.com/amazon-science/factual-confidence-of-llms)]
- Self-evaluation improves selective generation in large language models, <ins>NeurIPS Workshop, 2023</ins> [[Paper](http://arxiv.org/abs/2312.09300)]
- Benchmarking uncertainty quantification methods for large language models with LM-Polygraph, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2406.15627)][[Code](https://github.com/IINemo/lm-polygraph)]

#### Self-consistency
##### Reference-based Evaluation
- Do large language models know what they don’t know?, <ins>ACL Findings, 2023</ins> [[Paper](http://arxiv.org/abs/2305.18153)][[Code](https://github.com/yinzhangyue/SelfAware)]
- Knowledge of knowledge: Exploring known-unknowns uncertainty with large language models, <ins>arXiv, 2023</ins> [[Paper](http://arxiv.org/abs/2305.13712)][[Code](https://github.com/amayuelas/knowledge-of-knowledge)]
- Learning to trust your feelings: Leveraging self-awareness in LLMs for hallucination mitigation, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2401.15449)][[Code](https://github.com/liangyuxin42/dreamcatcher)]
- Behonest: Benchmarking honesty of large language models, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2406.13261)][[Code](https://github.com/GAIR-NLP/BeHonest)]
- Does fine-tuning llms on new knowledge encourage hallucinations?, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2405.05904)]

##### Reference-free Evaluation
- Promptbench: Towards evaluating the robustness of large language models on adversarial prompts, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.04528)][[Code](https://github.com/microsoft/promptbench)]
- Behonest: Benchmarking honesty of large language models, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2406.13261)][[Code](https://github.com/GAIR-NLP/BeHonest)]
- Quantifying language models’ sensitivity to spurious features in prompt design or: How I learned to start worrying about prompt formatting, <ins>ICLR, 2024</ins> [[Paper](http://arxiv.org/abs/2310.11324)][[Code](https://github.com/msclar/formatspread)]
- State of what art? A call for multi-prompt LLM evaluation, <ins>TACL, 2024</ins> [[Paper](http://arxiv.org/abs/2407.07840)][[Code](https://github.com/SLAB-NLP/Multi-Prompt-LLM-Evaluation)]
- On the worst prompt performance of large language models, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2406.10248)][[Code](https://github.com/cbwbuaa/On-the-Worst-Prompt-Performance-of-LLMs)]
- AI deception: A survey of examples, risks, and potential solutions, <ins>Patterns, 2024</ins> [[Paper](http://arxiv.org/abs/2308.14752)]
- Benchmarking and improving generator-validator consistency of language models, <ins>ICLR, 2024</ins> [[Paper](http://arxiv.org/abs/2310.01846)][[Code](https://github.com/XiangLi1999/GV-consistency)]

### Methods of Self-knowledge
#### Training-free Approaches
##### Predictive Probability
- Uncertainty quantification with pre-trained language models: A large-scale empirical analysis, <ins>EMNLP Findings, 2022</ins> [[Paper](http://arxiv.org/abs/2210.04714)][[Code](https://github.com/xiaoyuxin1002/UQ-PLM.git)]
- Language models (mostly) know what they know, <ins>arXiv, 2022</ins> [[Paper](http://arxiv.org/abs/2207.05221)]
- Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation, <ins>ICLR, 2023</ins> [[Paper](http://arxiv.org/abs/2302.09664)][[Code](https://github.com/lorenzkuhn/semantic_uncertainty)]
- Self-evaluation improves selective generation in large language models, <ins>NeurIPS Workshop, 2023</ins> [[Paper](http://arxiv.org/abs/2312.09300)]
- Shifting attention to relevance: Towards the uncertainty estimation of large language models, <ins>ACL, 2024</ins> [[Paper](http://arxiv.org/abs/2307.01379)][[Code](https://github.com/jinhaoduan/SAR)]
- GPT-4 technical report, <ins>arXiv, 2023</ins> [[Paper](http://arxiv.org/abs/2303.08774)]

##### Prompting
- Language models (mostly) know what they know, <ins>arXiv, 2022</ins> [[Paper](http://arxiv.org/abs/2207.05221)]
- Fact-and-reflection (FAR) improves confidence calibration of large language models, <ins>ACL Findings, 2024</ins> [[Paper](https://aclanthology.org/2024.findings-acl.515/)]
- Self-[in]correct: LLMs struggle with refining self-generated responses, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2404.04298)]
- Do large language models know what they don’t know? <ins>ACL Findings, 2023</ins> [[Paper](http://arxiv.org/abs/2305.18153)][[Code](https://github.com/yinzhangyue/SelfAware)]
- Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback, <ins>EMNLP, 2023</ins> [[Paper](https://aclanthology.org/2023.emnlp-main.330/)]
- Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2306.13063)][[Code](https://github.com/MiaoXiong2320/llm-uncertainty)]
- Navigating the grey area: How expressions of uncertainty and overconfidence affect language models, <ins>EMNLP, 2023</ins> [[Paper](http://arxiv.org/abs/2302.13439)][[Code](https://github.com/katezhou/navigating_the_grey)]

##### Sampling and Aggregation
- Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2306.13063)][[Code](https://github.com/MiaoXiong2320/llm-uncertainty)]
- Just rephrase it! Uncertainty estimation in closed-source language models via multiple rephrased queries, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2405.13907)]
- Prompt consistency for zero-shot task generalization, <ins>EMNLP Findings, 2022</ins> [[Paper](http://arxiv.org/abs/2205.00049)][[Code](https://github.com/violet-zct/swarm-distillation-zero-shot)]
- Calibrating large language models with sample consistency, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2402.13904)]
- Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation, <ins>ICLR, 2023</ins> [[Paper](http://arxiv.org/abs/2302.09664)][[Code](https://github.com/lorenzkuhn/semantic_uncertainty)]
- Generating with confidence: Uncertainty quantification for black-box large language models, <ins>TMLR, 2024</ins> [[Paper](http://arxiv.org/abs/2305.19187)][[Code](https://github.com/zlin7/UQ-NLG)]
- SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models, <ins>EMNLP, 2023</ins> [[Paper](https://arxiv.org/abs/2303.08896)][[Code](https://github.com/potsawee/selfcheckgpt)]
- Mitigating LLM hallucinations via conformal abstention, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2405.01563)]
- Calibrating long-form generations from large language models, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2402.06544)]
- Detecting hallucinations in large language models using semantic entropy, <ins>Nature, 2024</ins> [[Paper](https://doi.org/10.1038/s41586-024-07421-0)][[Code](https://github.com/jlko/semantic_uncertainty)]
- Fact-checking the output of large language models via token-level uncertainty quantification, <ins>ACL Findings, 2024</ins> [[Paper](http://arxiv.org/abs/2403.04696)][[Code](https://github.com/IINemo/lm-polygraph)]

#### Training-based Approaches
##### Supervised Fine-tuning
- Alignment for honesty, <ins>arXiv, 2023</ins> [[Paper](http://arxiv.org/abs/2312.07000)][[Code](https://github.com/GAIR-NLP/alignment-for-honesty)]
- R-tuning: Instructing large language models to say ‘I don’t know’, <ins>NAACL, 2024</ins> [[Paper](http://arxiv.org/abs/2311.09677)][[Code](https://github.com/shizhediao/R-Tuning)]
- Can AI assistants know what they don’t know?, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2401.13275)][[Code](https://github.com/OpenMOSS/Say-I-Dont-Know)]
- Teaching large language models to express knowledge boundary from their own signals, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2406.10881)]
- Knowledge verification to nip hallucination in the bud, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2401.10768)][[Code](https://github.com/fanqiwan/KCA)]
- Large language models must be taught to know what they don’t know, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2406.08391)][[Code](https://github.com/activatedgeek/calibration-tuning)]
- Teaching models to express their uncertainty in words, <ins>TMLR, 2022</ins> [[Paper](http://arxiv.org/abs/2205.14334)][[Code](https://github.com/sylinrl/CalibratedMath)]
- Calibrating large language models using their generations only, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2403.05973)][[Code](https://github.com/parameterlab/apricot)]
- Enhancing confidence expression in large language models through learning from past experience, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2404.10315)]

##### Reinforcement Learning
- Can AI assistants know what they don’t know?, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2401.13275)][[Code](https://github.com/OpenMOSS/Say-I-Dont-Know)]
- Rejection improves reliability: Training LLMs to refuse unknown questions using RL from knowledge feedback, <ins>COLM, 2024</ins> [[Paper](http://arxiv.org/abs/2403.18349)]
- The best of both worlds: Toward an honest and helpful large language model, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2406.00380)][[Code](https://github.com/Flossiee/HonestyLLM)]
- Sayself: Teaching LLMs to express confidence with self-reflective rationales, <ins>arXiv, 2024b</ins> [[Paper](http://arxiv.org/abs/2405.20974)][[Code](https://github.com/xu1868/SaySelf)]
- LACIE: Listener-aware finetuning for confidence calibration in large language models, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2405.21028)][[Code](https://github.com/esteng/pragmatic_calibration)]
- Linguistic calibration of long-form generations, <ins>ICML, 2024</ins> [[Paper](http://arxiv.org/abs/2404.00474)][[Code](https://github.com/tatsu-lab/linguistic_calibration)]

##### Probing
- Language models (mostly) know what they know, <ins>arXiv, 2022</ins> [[Paper](http://arxiv.org/abs/2207.05221)]
- The internal state of an LLM knows when it’s lying, <ins>EMNLP Findings, 2023</ins> [[Paper](http://arxiv.org/abs/2304.13734)][[Code]()]
- The geometry of truth: Emergent linear structure in large language model representations of true/false datasets, <ins>COLM, 2024</ins> [[Paper](http://arxiv.org/abs/2310.06824)][[Code](https://github.com/saprmarks/geometry-of-truth)]
- On the universal truthfulness hyperplane inside LLMs, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2408.10692)]
- Discovering latent knowledge in language models without supervision, <ins>ICLR, 2023</ins> [[Paper](http://arxiv.org/abs/2212.03827)][[Code](https://www.github.com/collin-burns/discovering_latent_knowledge)]
- Semantic entropy probes: Robust and cheap hallucination detection in LLMs, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2406.15927)]
- LLM internal states reveal hallucination risk faced with a query, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2407.03282v1)]

### Methods of Self-consistency
#### Training-free Approaches
##### Chain-of-thought Prompting
- Chain-of-thought prompting elicits reasoning in large language models, <ins>NeurIPS, 2022</ins> [[Paper](http://arxiv.org/abs/2201.11903)]
- Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks, <ins>TMLR, 2023</ins> [[Paper](http://arxiv.org/abs/2308.04584)][[Code](https://github.com/TIGER-AI-Lab/Program-of-Thoughts)]
- Least-to-most prompting enables complex reasoning in large language models, <ins>ICLR, 2023</ins> [[Paper](http://arxiv.org/abs/2305.01341)]
- Take a step back: Evoking reasoning via abstraction in large language models, <ins>ICLR, 2024</ins> [[Paper](http://arxiv.org/abs/2304.03461)]
- Executable code actions elicit better LLM agents, <ins>ICML, 2024</ins> [[Paper](http://arxiv.org/abs/2312.06681)][[Code](https://github.com/xingyaoww/code-act)]
- Teaching LLMs to abstain across languages via multilingual feedback, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2406.15948)][[Code](https://github.com/BunsenFeng/M-AbstainQA)]
- Phenomenal yet puzzling: Testing inductive reasoning capabilities of language models with hypothesis refinement, <ins>ICLR, 2024</ins> [[Paper](http://arxiv.org/abs/2310.07651)][[Code](https://github.com/linlu-qiu/lm-inductive-reasoning)]
- Language models as inductive reasoners, <ins>EACL, 2024</ins> [[Paper](https://aclanthology.org/2024.eacl-long.13/)][[Code](https://github.com/ZonglinY/Inductive_Reasoning)]
- Navigating the grey area: How expressions of uncertainty and overconfidence affect language models, <ins>EMNLP, 2023</ins> [[Paper](http://arxiv.org/abs/2302.13439)][[Code](https://github.com/katezhou/navigating_the_grey)]
- Hypothesis search: Inductive reasoning with language models, <ins>ICLR, 2024</ins> [[Paper](http://arxiv.org/abs/2310.09199)][[Code](https://github.com/Relento/hypothesis_search)]

##### Decoding-time Intervention
- Inference-time intervention: Eliciting truthful answers from a language model, <ins>NeurIPS, 2024</ins> [[Paper](https://openreview.net/forum?id=aLLuYpn83y)][[Code](https://github.com/likenneth/honest_llama)]
- In-context sharpness as alerts: An inner representation perspective for hallucination mitigation, <ins>ICML, 2024</ins> [[Paper](http://arxiv.org/abs/2404.04298)][[Code](https://github.com/hkust-nlp/Activation_decoding.git)]
- Dola: Decoding by contrasting layers improves factuality in large language models, <ins>ICLR, 2024</ins> [[Paper](http://arxiv.org/abs/2309.03883)][[Code](https://github.com/voidism/DoLa)]
- Unchosen experts can contribute too: Unleashing MoE models’ power by self-contrast, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2407.18698)][[Code](https://github.com/DavidFanzz/SCMoE.git)]
- Alleviating hallucinations of large language models through induced hallucinations, <ins>arXiv, 2023b</ins> [[Paper](http://arxiv.org/abs/2312.15710)][[Code](https://github.com/hillzhang1999/ICD)]
- Trusting your evidence: Hallucinate less with context-aware decoding, <ins>NAACL, 2024</ins> [[Paper](http://arxiv.org/abs/2305.14739)]
- Mitigating object hallucinations in large vision-language models through visual contrastive decoding, <ins>CVPR, 2024</ins> [[Paper](http://arxiv.org/abs/2406.03441)][[Code](https://github.com/DAMO-NLP-SG/VCD)]

##### Sampling and Aggregation
- Self-consistency improves chain of thought reasoning in language models, <ins>ICLR, 2023</ins> [[Paper](http://arxiv.org/abs/2203.11171)]
- Universal self-consistency for large language model generation, <ins>arXiv, 2023</ins> [[Paper](http://arxiv.org/abs/2311.17311)]
- Integrate the essence and eliminate the dross: Fine-grained self-consistency for free-form language generation, <ins>ACL, 2024</ins> [[Paper](http://arxiv.org/abs/2407.02056)][[Code](https://github.com/WangXinglin/FSC)]
- Atomic self-consistency for better long form generations, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2405.13131)][[Code](https://github.com/raghavlite/ASC)]

##### Post-generation Revision
- Chain-of-verification reduces hallucination in large language models, <ins>arXiv, 2023</ins> [[Paper](http://arxiv.org/abs/2309.11495)]
- A stitch in time saves nine: Detecting and mitigating hallucinations of LLMs by validating low-confidence generation, <ins>arXiv, 2023</ins> [[Paper](http://arxiv.org/abs/2307.03987)]
- Verify-and-edit: A knowledge-enhanced chain-of-thought framework, <ins>ACL, 2023</ins> [[Paper](http://arxiv.org/abs/2305.03268)][[Code](https://github.com/RuochenZhao/Verify-and-Edit)]

#### Training-based Approaches
##### Self-aware Fine-tuning
- Alignment for honesty, <ins>arXiv, 2023</ins> [[Paper](http://arxiv.org/abs/2312.07000)][[Code](https://github.com/GAIR-NLP/alignment-for-honesty)]
- R-tuning: Instructing large language models to say ‘I don’t know’, <ins>NAACL, 2024</ins> [[Paper](http://arxiv.org/abs/2311.09677)][[Code](https://github.com/shizhediao/R-Tuning)]
- Can AI assistants know what they don’t know?, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2401.13275)][[Code](https://github.com/OpenMOSS/Say-I-Dont-Know)]
- Knowledge verification to nip hallucination in the bud, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2401.10768)][[Code](https://github.com/fanqiwan/KCA)]
- Unfamiliar fine-tuning examples control how language models hallucinate, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2405.12889)][[Code](https://github.com/katiekang1998/llm_hallucinations)]

##### Self-supervised Fine-tuning
- Fine-tuning language models for factuality, <ins>ICLR, 2024</ins> [[Paper](http://arxiv.org/abs/2312.04045)][[Code](https://github.com/kttian/llm_factuality_tuning)]
- Self-alignment for factuality: Mitigating hallucinations in LLMs via self-evaluation, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.09267)][[Code](https://github.com/zhangxy-2019/Self-Alignment-for-Factuality)]
- FLAME: Factuality-aware alignment for large language models, <ins>arXiv, 2024</ins> [[Paper](http://arxiv.org/abs/2404.02655)]


 ## Community Support

If you have feedback on the taxonomy, notice missing papers, or have updates on accepted arXiv preprints, please email us or submit a pull request in the following markdown format:

```markdown
Paper Title, <ins>Conference/Journal/Preprint, Year</ins>  [[Paper](link)] [[Code](link)].
```

## Citation
```
```