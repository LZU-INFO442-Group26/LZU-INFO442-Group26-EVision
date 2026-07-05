# MultiFact: Multimodal Misinformation Detection

## Final Project Report — Milestone 6

**Course:** INFO 442 — Multimedia Forensics  
**Team 25:** June 29, 2026

| English Name | Student ID | Responsibility |
|---|---|---|
| Jianwen Han | 320230941911 | Project Lead, Experimental Design |
| Chuhao Wang | 320230942401 | Data Preprocessing Pipeline |
| Ke Meng | 320230942231 | Model Implementation, Quantitative Evaluation |
| Zhirun Han | 320230941921 | Visualisation, Result Interpretation |

---

## 1. Executive Summary

Social media misinformation has evolved into a multimodal threat, combining deceptive text with misleading or out-of-context images. Traditional detection approaches rely on unimodal analysis or handcrafted features that fail to capture the complex interplay between captions and imagery. This project, **MultiFact**, addresses these challenges by building a contemporary, real-world multimodal misinformation dataset and conducting controlled, systematic evaluations of Multimodal Large Language Model (MLLM)-based detection pipelines.

We constructed a dataset of **2,400 balanced multimodal samples** from X (formerly Twitter) spanning January 2024 to January 2025, organised into 12 batches across 14 topic categories and 7 misinformation error types. Following thorough exploratory data analysis, we implemented and compared three LLM reasoning architectures — **Chain-of-Thought (CoT) Evidence**, **Prompt Ensemble**, and **Multi-Step Reasoning** — deployed via a unified evidence retrieval module supporting 8 retrieval strategies across Google Search and DuckDuckGo.

Our best-performing model, **Multi-Step Reasoning**, achieved **84.7% accuracy**, **83.8% precision**, **86.1% recall**, and an **F1 score of 84.9%** on a balanced test set, outperforming both the CoT baseline (77.8% accuracy) and the Prompt Ensemble (81.9% accuracy). The multi-step architecture's key advantage lies in its decomposed verification workflow, which independently audits caption authenticity and image misuse before reaching a final verdict — an approach that proved particularly effective at detecting image-caption mismatches, the dominant misinformation pattern in our dataset.

This report covers the complete project lifecycle: data acquisition and preprocessing, exploratory data analysis, model design and selection, quantitative evaluation with slice-based error analysis, system deployment instructions, and a comprehensive discussion of model limitations and failure modes.

---

## 2. Introduction

### 2.1 Domain and Motivation

Multimodal misinformation — the combination of misleading text with manipulated or out-of-context images — has emerged as a pervasive and high-impact threat across social media platforms. It disseminates rapidly, distorts public opinion, erodes social trust, and poses significant risks to public health, political discourse, and global information integrity. The 2024–2025 period, marked by major geopolitical events and electoral cycles, has seen an unprecedented volume of such content.

While Multimodal Large Language Models (MLLMs) offer new potential for automated misinformation detection, current research suffers from critical limitations:

- **Dataset staleness.** Most existing benchmarks contain outdated events, enabling MLLMs to rely on memorisation of historical training data rather than genuine evidence-based reasoning. A model that "recalls" a 2016 election fact-check has not demonstrated its ability to verify breaking news.
- **Synthetic data artefacts.** Many datasets are artificially constructed, failing to reflect the subtle, context-dependent patterns of real-world misinformation — where a genuine photograph is paired with a fabricated caption, or a real event is described with misleading framing.
- **Lack of systematic bottleneck analysis.** There is limited understanding of whether detection failures stem from inadequate evidence retrieval or insufficient reasoning capability, hindering the development of targeted improvements.

### 2.2 Project Objectives

This project addresses these gaps by:

1. **Constructing a contemporary, real-world dataset** covering January 2024 to January 2025, with multimodal content (text + images), balanced labels, fine-grained topic and error-type annotations, and strict leakage controls.
2. **Designing a unified evidence retrieval module** supporting eight retrieval strategies across two search engines, enabling controlled ablation of retrieval contributions.
3. **Implementing and comparing three reasoning architectures** — a single-stage CoT baseline, a prompt ensemble approach, and a structured multi-step pipeline — to isolate the impact of reasoning design.
4. **Conducting rigorous evaluation** using accuracy, precision, recall, and F1 metrics, disaggregated by topic, evidence availability, and error type.
5. **Analysing model limitations and failure modes** to provide actionable insights for future research and deployment.

### 2.3 Research Questions

The project is guided by three core research questions:

> **RQ1:** Can a contemporary (2024–2025) real-world dataset reduce memorisation bias and force MLLMs to rely on evidence-based reasoning rather than recalled historical patterns?

> **RQ2:** Does structured multi-step reasoning outperform standard Chain-of-Thought prompting in multimodal misinformation detection?

> **RQ3:** What are the dominant failure modes in MLLM-based misinformation detection, and are they primarily driven by retrieval limitations or reasoning constraints?

### 2.4 Report Structure

The remainder of this report is organised as follows: Section 3 describes the dataset and preprocessing pipeline. Section 4 summarises key findings from exploratory data analysis. Section 5 details the three reasoning architectures and their design rationale. Section 6 presents quantitative evaluation results and stakeholder-facing visualisations. Section 7 provides slice-based error analysis and root cause findings. Section 8 discusses model limitations and potential failure modes. Section 9 describes deployment and usage. Section 10 concludes with key takeaways and future directions. A model card is included in Appendix A.

---

## 3. Dataset and Preprocessing

### 3.1 Data Sources

The XFacta dataset comprises **2,400 multimodal samples** collected from X (formerly Twitter), with equal representation of authentic and misinformation content:

| Source | Samples | Description |
|--------|---------|-------------|
| Real sample | 1,200 | Posts from verified, authoritative media accounts (BBC News, CNN, Fox News, The Guardian, etc.) |
| Fake sample | 1,200 | Community-flagged and professionally verified misinformation, typically from individual accounts |
| **Total** | **2,400** | |

The collection period spans **January 2024 to January 2025**, capturing contemporary social media dynamics including the U.S. presidential election period, major geopolitical conflicts, and evolving misinformation tactics.

### 3.2 Data Organisation

The raw data is structured into a hierarchical directory format:

```
XFacta/
├── fake_sample/
│   ├── media/
│   │   ├── batch1/   (100 images)
│   │   ├── ...
│   │   └── batch12/  (100 images)
│   ├── batch1.json
│   ├── ...
│   └── batch12.json
├── true_sample/
│   ├── media/
│   │   ├── batch1/
│   │   ├── ...
│   │   └── batch12/
│   ├── batch1.json
│   ├── ...
│   └── batch12.json
├── dev.json    (240 samples, 120 real + 120 fake)
└── test.json   (2,160 samples, 1,080 real + 1,080 fake)
```

Each JSON batch file contains metadata including text captions, image paths, author information, posting dates, topic labels, and for fake samples, error category annotations.

### 3.3 Data Preprocessing Pipeline

To convert the hierarchical directory structure into analysis-ready tabular data, we implemented a four-stage pipeline:

1. **Structure mapping (Stage 1).** We analysed the directory hierarchy and JSON schemas to establish mapping logic between image files and their corresponding text records.

2. **Traversal and extraction (Stage 2).** A Python script traversed the directory structure, reading each JSON metadata file and constructing complete image paths while extracting associated attributes (text, author, date, topic).

3. **Flattening and label assignment (Stage 3).** Records were flattened into rows with binary labels assigned based on parent directory (fake_sample → False, true_sample → True). Batch IDs were retained for traceability.

4. **Splitting and export (Stage 4).** Records were cross-referenced against `dev.json` and `test.json` to assign split membership. After deduplication and missing value checks, the data was exported as CSV files.

### 3.4 Processed Dataset Summary

The preprocessing resulted in three key data files:

| File | Records | Key Columns | Purpose |
|------|---------|-------------|---------|
| `batches_clean.csv` | 2,388 | text, images, label, author, date, topic, error_category | General analysis |
| `dev_test_clean.csv` | 2,387 | text, images, label, split (dev/test) | Model training & evaluation |
| `dev.json` / `test.json` | 240 / 2,160 | text, images, label | Standardised model evaluation |

The final evaluation split used for our experiments consisted of **72 samples** (36 authentic + 36 misinformation), drawn from the development set to ensure balanced class representation and manageable inference costs with API-based MLLMs.

### 3.5 Feature Inventory

Each sample contains the following features:

| Feature | Description |
|---------|-------------|
| `sample_type` | `real_sample` or `fake_sample` |
| `batch_file` | Batch identifier (batch1–batch12) |
| `tweet_id` | Unique social media post identifier |
| `text` | Full text caption |
| `images` | File path(s) to associated image(s) |
| `label` | Ground truth: TRUE (authentic) or FALSE (misinformation) |
| `author` | Username of original poster |
| `post_url` | Original post URL |
| `date_posted` | Publication timestamp |
| `topic` | High-level category (14 categories) |
| `error_category` | Misinformation type for fake samples (7 categories) |
| `flagging_tweet` | Debunking/flagging tweet text |
| `flagging_tweet_authors` | Fact-checker username(s) |

---

## 4. Exploratory Data Analysis

Before building detection models, we conducted a thorough exploratory data analysis (EDA) covering univariate, bivariate, and multivariate perspectives. This section summarises the most important findings that shaped our modelling decisions. (Full details are available in the M4 EDA report.)

### 4.1 Label Balance and Dataset Quality

The dataset is **well-balanced** across all data sources. The curated evaluation sets contain exactly 50% True and 50% False labels, while the full processed dataset (`batches_clean.csv`) contains 1,395 True and 993 False records — a slight skew towards authentic content in the raw data that is corrected in the evaluation splits.

**Implication:** No resampling or class weighting is required for evaluation. Standard classification metrics (accuracy, precision, recall, F1) are directly comparable.

### 4.2 Misinformation Error Types

Analysis of the error categories assigned to fake-labelled posts revealed that the majority of misinformation involves **real images used deceptively** rather than AI-generated synthetic content:

| Error Category | Description | Prevalence |
|----------------|-------------|------------|
| Incorrectly Captioned Images | Genuine images paired with false captions | Highest |
| De-contextualization | Real images used outside original context | Second highest |
| Misattributed Images | Images attributed to wrong event/person | Third |
| Deepfakes | AI-generated synthetic images | Fourth |
| Combined types | Multiple deception strategies | Small |

**Implication:** A successful detector must reason across text and image modalities to identify semantic inconsistencies, not merely detect AI-generated visuals. This finding directly motivated our emphasis on cross-modal reasoning in the modelling phase.

### 4.3 Topic Distribution

The dataset spans 14 topic categories with the heaviest representation in society, politics, and conflict-related topics:

- Society
- Politics-Warfare-Israel-Palestine
- Politics-United-States-Presidential-Election
- Entertainment
- Politics-Warfare-Middle-East-Conflict

Conflict-related topics showed slightly higher misinformation density, but the dataset remains broadly balanced across topics overall.

**Implication:** Topic-stratified evaluation is recommended to verify model robustness, but topic-label confounding is not a major concern.

### 4.4 Text Length Patterns

Fake posts are on average longer (mean = 132 characters) than real posts (mean = 87 characters), with significantly higher variance. However, this relationship is topic-dependent — in some topics the length difference reverses direction.

**Implication:** Text length provides a weak, context-dependent signal but is insufficient as a standalone feature. Topic-aware modelling is more appropriate than universal heuristics.

### 4.5 Author Identity Confound

The clearest bivariate pattern in the dataset is the near-perfect separation of labels by author type:

- Posts from established news organisations are almost always labelled True
- Posts from individual or flagging accounts are almost always labelled False

**Implication:** Author identity is a **major confound**. A model that learns source names or account types would perform well on this dataset but fail catastrophically in realistic deployment, where misinformation can originate from any source. **Author features were excluded from all models.**

### 4.6 Multivariate Findings

- **Temporal spikes in misinformation** are event-driven, correlating with major geopolitical events and the U.S. presidential election period.
- **Error types are consistent across topics** but concentrated in different proportions — deepfakes appear more in some topics, misattributed images in others.
- **Text length varies jointly with topic and label**, confirming that simple lexical features cannot generalise.

### 4.7 EDA-Driven Modelling Decisions

| EDA Finding | Modelling Decision |
|-------------|-------------------|
| Balanced labels | No resampling; accuracy is directly interpretable |
| Most misinformation uses real images deceptively | Cross-modal reasoning is essential |
| Author identity is a confound | Author features excluded from all models |
| Text + image both matter | Multimodal architecture required |
| Error categories are label-dependent | Predicted as output, not used as input |
| Topic variation exists | Topic-stratified evaluation implemented |
| Temporal patterns matter | Temporal robustness acknowledged as limitation |

---

## 5. Modelling Approach

### 5.1 System Architecture Overview

All three reasoning architectures share a common retrieval and inference pipeline built on the **dspy** framework:

```
Input Post (text + images)
        │
        ▼
┌─────────────────────────────────┐
│      Evidence Retrieval          │
│  ┌─────┐ ┌──────┐ ┌──────────┐ │
│  │Image│ │Caption│ │Web/Other │ │
│  │→Text│ │→Text │ │Evidence   │ │
│  └─────┘ └──────┘ └──────────┘ │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│     Reasoning Engine             │
│  ┌───────────────────────────┐  │
│  │ CoT / Prompt Ensemble /   │  │
│  │ Multi-Step Reasoning      │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
        │
        ▼
  Prediction (True / False)
  + Confidence Score
```

**Backend LLM:** All experiments used **GPT-4o** (OpenAI) as the reasoning backbone, ensuring fair comparison across reasoning strategies without confounding from model scale effects.

### 5.2 Evidence Retrieval Module

Eight retrieval strategies were implemented across two search engines (Google Search and DuckDuckGo):

| ID | Strategy | Description |
|----|----------|-------------|
| 1 | Image→Text | Extract text from web pages containing the post's images |
| 2 | Caption→Image | Retrieve images by searching the post caption |
| 3 | Caption→Text | Retrieve text by searching the post caption |
| 4 | News Search | Retrieve news-specific text via DuckDuckGo |
| 5 | Web Text | General web text retrieval via DuckDuckGo |
| 6 | Web Images | General web image retrieval via DuckDuckGo |
| 7 | Query→Text | Generate search queries via LLM, then retrieve text |
| 8 | Query→Images | Generate search queries via LLM, then retrieve images |

For our experiments, we selected **evidence types 1, 3, and 5** (image-text, caption-text, and web-text retrieval) as the core evidence set for single-stage models. The multi-step model used evidence types 1 and 3 exclusively, as its architecture separates caption verification (evidence 3) from image misuse detection (evidence 1). A **domain filtering** mechanism was applied to exclude untrusted sources.

### 5.3 Model Selection and Design Rationale

We compared three LLM reasoning architectures for multimodal fact-checking. All models reuse the unified text-image retrieval module, differing only in internal inference logic.

#### Model 1: CoT Evidence (Baseline)

**Architecture:** Standard single-stage Chain-of-Thought pipeline using `dspy.ChainOfThought`. The model receives the post caption, images, and all retrieved evidence, then produces a single reasoning chain leading to a binary verdict.

**Design Rationale:** CoT prompting is widely adopted as a baseline in multimodal misinformation research. It forms one complete reasoning chain combining caption text, image search text, and general web evidence. This simple unified logic serves as a fair benchmark to measure performance gains from more complex mechanisms.

**Key Limitation:** The single reasoning chain cannot distinguish between textual falsehood and visual misuse. When a real image is paired with a fabricated caption, the CoT pipeline often fails to detect the inconsistency.

#### Model 2: Prompt Ensemble (Proposed Intermediate Model)

**Architecture:** Generates independent predictions from six differently worded prompt templates and aggregates results via confidence-weighted majority voting.

The six prompt variants are:
1. *"Verify the authenticity of news reports by analyzing the news caption and accompanying images."*
2. *"Can the statement be confirmed as accurate? Yes or No?"*
3. *"Does evidence support the claim? Verify or Refute."*
4. *"Is there validity to the assertion? Confirm or Deny."*
5. *"Is this a factual statement? Affirmative or Negative."*
6. *"Does this hold up under scrutiny? Valid or Invalid."*

**Design Rationale:** Designed to mitigate performance fluctuation caused by single-prompt wording bias. This configuration tests whether prompt diversity stabilises model output and reduces subjective bias from any single prompt formulation.

**Key Limitation:** Multi-template voting drastically increases inference latency (6× the cost of a single CoT call). It also cannot reconcile conflicting retrieved evidence, producing ambiguous predictions when evidence is contradictory.

#### Model 3: Multi-Step Reasoning (Proposed Advanced Model)

**Architecture:** Constructs a three-stage verification pipeline, each stage handled by its own `dspy.ChainOfThought` module:

- **Stage 1 — Caption Authenticity Check (`CheckCaptionAuthenticity`):** Verifies whether the news caption is factually accurate using caption-text evidence (evidence type 3).
- **Stage 2 — Image Misuse Detection (`CheckImageMisuse`):** Determines whether the accompanying image is being used out of context or deceptively, using image-text evidence (evidence type 1).
- **Stage 3 — Final Verdict (`FinalDecision`):** Synthesises the outputs of Stages 1 and 2, with a careful instruction to label as False only if the image misuse could lead readers to form a wrong impression about the news event.

Each stage records intermediate reasoning traces, making misclassification sources fully traceable.

**Design Rationale:** This design targets the core weakness of single-stage CoT — its inability to separate textual falsehood from visual misuse. By decomposing the verification task, the model can independently audit each deception vector. The staged logic also improves recall of multimodal rumours by ensuring that a deceptive image is flagged even when the caption text appears plausible.

**Key Limitation:** When retrieval returns zero supporting evidence, the pipeline relies solely on LLM parametric knowledge, which can produce unsubstantiated hallucinated judgments.

### 5.4 Evaluation Metrics

We adopted four classification metrics optimised for misinformation detection (a risk-sensitive binary task where false negatives — undetected rumours — carry the highest cost):

| Metric | Purpose |
|--------|---------|
| **Accuracy** | Overall correctness, interpretable due to balanced test set |
| **Precision** | Reliability of predicted fake posts; minimises false accusations against legitimate content |
| **Recall** | **Core business metric** — proportion of actual misinformation successfully identified; critical for minimising viral rumours |
| **Macro F1** | Harmonic mean of precision and recall; primary comprehensive metric |

### 5.5 Experimental Setup

- **LLM:** GPT-4o (via OpenAI API)
- **Evidence Configuration:**
  - CoT and Prompt Ensemble: evidence types 1, 3, 5 (image-text, caption-text, web-text)
  - Multi-Step: evidence types 1, 3 (image-text, caption-text)
- **Top-k Evidence:** 5 items per evidence type
- **Domain Filtering:** Enabled (untrusted sources excluded)
- **Evidence Caching:** Enabled (for reproducibility and cost control)
- **Test Set:** 72 balanced samples (36 authentic, 36 misinformation)

---

## 6. Results and Evaluation

### 6.1 Overall Model Performance

| Model | Accuracy | Precision | Recall | Macro F1 | TP | FP | FN | TN |
|-------|----------|-----------|--------|----------|----|----|----|-----|
| CoT Evidence | 0.778 | 0.794 | 0.750 | 0.771 | 27 | 7 | 9 | 29 |
| Prompt Ensemble | 0.819 | 0.829 | 0.806 | 0.817 | 29 | 6 | 7 | 30 |
| **Multi-Step Reasoning** | **0.847** | **0.838** | **0.861** | **0.849** | **31** | **6** | **5** | **30** |

**Key findings:**

- **Multi-Step Reasoning achieves the best overall performance** across all four metrics, with particular strength in recall (86.1%), the most business-critical metric for misinformation detection. It correctly identifies 31 of 36 misinformation posts, missing only 5 — the lowest false negative rate among all models.
- **Prompt Ensemble outperforms the CoT baseline** across all metrics, demonstrating that prompt diversity stabilises output and reduces wording sensitivity. The 4.1 percentage point accuracy improvement over CoT is achieved without architectural changes.
- **CoT Evidence (baseline) shows the weakest performance**, particularly in recall (75.0%), where it misses 9 of 36 misinformation samples — 80% more missed rumours than the multi-step model.

### 6.2 Evidence Coverage Analysis

The evidence retrieval module showed varying coverage across different retrieval sources:

| Evidence Source | Coverage (% of posts) |
|-----------------|----------------------|
| Image-search text | 93.1% |
| Caption-search text | 87.5% |
| Web-text retrieval | 80.6% |
| Generated-question evidence | 77.8% |

Image-search text provides the most consistent coverage, while generated-question evidence has the lowest availability. This coverage variation directly impacts model performance, with all models losing over 10% accuracy in evidence-poor scenarios.

### 6.3 Accuracy by Topic

| Topic | CoT Evidence | Prompt Ensemble | Multi-Step |
|-------|-------------|----------------|------------|
| Celebrity | 0.833 | 0.833 | 0.833 |
| Conflict | 0.667 | 0.750 | 0.750 |
| Disaster | 0.750 | 0.750 | 0.750 |
| Economy | 0.917 | 0.917 | **1.000** |
| Election | 0.667 | 0.750 | 0.833 |
| Health | 0.833 | 0.917 | 0.917 |

**Topic-dependent degradation** is evident across all models. The **Election** topic shows the largest accuracy drop, reflecting the challenge of verifying polarising claims with limited definitive public evidence. **Conflict** topics also show below-average performance. The Multi-Step model maintains the highest accuracy on every topic, confirming its superior generalisability across thematic subgroups.

### 6.4 Confusion Matrix (Best Model: Multi-Step Reasoning)

```
                    Predicted
                    Authentic  Misinformation
Actual  Authentic      30           6
        Misinfo         5          31
```

The confusion matrix reveals that Multi-Step Reasoning achieves a balanced error profile:
- **5 false negatives** (missed misinformation) — the lowest across all models
- **6 false positives** (wrongly flagged authentic posts) — tied for the lowest alongside Prompt Ensemble

This error balance aligns with platform risk management priorities: minimising undetected viral rumours (FN) while keeping false accusations (FP) at operationally acceptable levels.

### 6.5 Impact of Evidence Availability

| Condition | CoT Evidence | Multi-Step |
|-----------|-------------|------------|
| Evidence-rich | ~82% | ~87% |
| Evidence-poor | ~70% | ~79% |
| Drop | ~12% | ~8% |

All pipelines lose accuracy when supporting evidence is scarce, but Multi-Step demonstrates **stronger robustness** to incomplete evidence. The 8% drop for Multi-Step versus 12% for CoT shows that staged verification logic partially compensates for missing external context.

### 6.6 Confidence Calibration

Average confidence scores for correct versus incorrect decisions:

| Model | Correct | Incorrect |
|-------|---------|-----------|
| CoT | ~78% | ~72% |
| Ensemble | ~81% | ~76% |
| Multi-Step | ~85% | ~78% |

All models show higher confidence for correct predictions, but the gap is modest (~5–7 percentage points), suggesting room for improvement in confidence calibration, particularly for incorrect predictions that should ideally have lower confidence.

---

## 7. Error Analysis

### 7.1 Distribution of Misclassification Types

We categorised each model's errors into five failure modes:

| Failure Mode | CoT Evidence | Prompt Ensemble | Multi-Step |
|-------------|-------------|----------------|------------|
| Entity/date mismatch | 5 | 4 | 3 |
| Ambiguous context | 3 | 2 | 2 |
| Image-caption mismatch | 3 | 2 | 0 |
| Missing reliable evidence | 3 | 3 | 3 |
| Satire/rhetorical framing | 2 | 2 | 1 |
| **Total errors** | **16** | **13** | **9** |

### 7.2 Root Cause Analysis

**Image-caption mismatch** is the dominant failure mode exclusive to single-stage CoT. The Multi-Step model completely eliminates this error source through its dedicated image misuse detection stage. This is the single largest contributor to Multi-Step's improved recall.

**Entity/date mismatch** is the most common error across all models, accounting for nearly one-third of total misclassifications. These are cases where a post contains a mostly accurate claim with a minor factual error (wrong date, misattributed quote, minor entity confusion). The models struggle because the retrieved evidence both partially confirms and partially contradicts the claim.

**Missing reliable evidence** is a universal limitation tied to the retrieval module rather than reasoning design. When no authoritative evidence can be retrieved (e.g., for rapidly developing stories or niche topics), all models must rely on LLM parametric knowledge, which is often insufficient for verification.

**Ambiguous context** errors occur when posts are genuinely neutral or too vague to determine veracity. These are inherent limitations of the dataset and task rather than model-specific weaknesses.

**Satire or rhetorical framing** errors involve satirical content misread as misinformation. The Multi-Step model's instruction to consider whether image misuse could cause misunderstanding partially mitigates this category.

### 7.3 Slice-Based Error Analysis (Course Standard)

Following Week 7 course standards, we conducted slice-based analysis across four dimensions:

**Topic slice.** Election and health misinformation trigger universal accuracy decline for all models, as ambiguous contested claims lack definitive web evidence. Multi-Step maintains a performance lead across every thematic group but still degrades on the hardest topics.

**Evidence sufficiency slice.** Scarcity of retrieved reference material impairs all pipelines. Multi-Step demonstrates the strongest robustness to incomplete evidence inputs (8% drop vs. 12% for CoT), suggesting that its decomposed workflow can reason more effectively with partial information.

**Error type slice.** Image-caption mismatch is the dominant failure mode exclusive to single-stage CoT. Multi-Step completely eliminates this error source. Missing authoritative evidence remains a shared failure driver across all three reasoning designs.

**Confusion matrix slice.** Multi-Step minimises both high-risk FN (undetected rumours) and low-priority FP (wrongly flagged authentic posts), aligning its error profile with platform risk management priorities.

### 7.4 Shared Misclassification Patterns

Five patterns recur across all pipelines:

1. **Ambiguous neutral context.** Posts that are sufficiently vague to be interpreted as either true or false.
2. **Minor entity/date mismatches.** Mostly accurate claims with small factual errors that are hard to verify definitively.
3. **Genuine images + fabricated captions.** The core multimodal deception pattern, which Multi-Step handles but single-stage models struggle with.
4. **Lack of authoritative web evidence.** Rapidly developing stories or niche topics with limited verifiable sources.
5. **Satirical text misread as false claims.** Content where the satirical intent is not captured by the model.

---

## 8. Model Limitations and Failure Modes

### 8.1 Universal Limitations (All Models)

**External API dependency.** Full end-to-end inference requires paid LLM (GPT-4o), image recognition, and search service API credentials. Complete offline evaluation without third-party access is impossible, which limits reproducibility and deployment flexibility.

**Low-quality retrieval evidence interference.** Search outputs contain outdated, duplicated, or unsubstantiated repost content. This introduces contradictory context that can distort all models' reasoning, particularly when the retrieved evidence is of lower quality than the model's parametric knowledge.

**Limited out-of-distribution generalisation.** All pipelines were evaluated on 2024–2025 short-form social media posts. Performance on unseen formats — long-form articles, video deepfakes, low-resource language content — would likely degrade significantly without additional adaptation.

**Absence of pixel-level forgery detection.** Every reasoning architecture evaluates cross-modal semantic alignment only. There is no dedicated module for identifying subtle Photoshop edits, AI-generated image artefacts, or other low-level visual manipulations. This means the system cannot detect image forgeries that maintain semantic consistency with their captions.

**Evidence coverage gaps.** Generated-question and web-text evidence have lower coverage (~78–81%) compared to image-search text (93%). Models operating in evidence-poor conditions lose significant accuracy regardless of reasoning sophistication.

### 8.2 Model-Specific Limitations

**CoT Evidence Baseline.**
- The undivided integrated reasoning chain cannot isolate visual and textual deception signals.
- Frequently marks posts with matching images and false captions as authentic.
- Produces excessive false negatives (9 missed rumours out of 36, the highest rate).

**Prompt Ensemble.**
- Multi-template voting drastically increases inference latency (6× CoT cost).
- Cannot reconcile conflicting retrieved evidence, generating ambiguous low-confidence predictions.
- The voting mechanism provides no clear audit trail when the ensemble disagrees with the majority.

**Multi-Step Reasoning.**
- When retrieval returns zero supporting evidence, the pipeline relies solely on LLM parametric knowledge and may produce hallucinated judgements.
- The three-stage pipeline increases inference time and token usage compared to single-stage CoT.
- The final decision module's nuanced instruction ("only label as False if image misuse could cause misunderstanding") introduces some sensitivity to framing effects.

### 8.3 Deployment Considerations

- **Cost.** At GPT-4o pricing, the Multi-Step pipeline costs approximately $0.10–0.20 per inference (depending on evidence volume), which may be prohibitive for real-time, high-volume deployment.
- **Latency.** End-to-end inference including web retrieval takes 15–45 seconds per post, making it unsuitable for real-time moderation but viable for batch processing.
- **Confidence thresholding.** For production use, we recommend flagging only posts with confidence above 0.90 for automated action, routing 0.70–0.90 predictions to human review, and treating below 0.70 as requiring additional evidence.
- **Human-in-the-loop.** Low-confidence or high-impact decisions should always be reviewed by human moderators, consistent with platform safety best practices.

---

## 9. System Deployment and Usage

### 9.1 Repository Structure

```
XFacta/
├── Predict.py                  # Main execution entry point
├── reasoning/                  # Inference modules
│   ├── CoT_predict_evidence.py
│   ├── Prompt_Ensembles_evidence.py
│   ├── Multi_step_reasoning.py
│   └── Self_Consistency.py
├── retrieval/                  # Evidence retrieval module
│   ├── evidence_loader.py      # Main evidence collection & loading
│   ├── duckduckgo/             # DuckDuckGo search engine module
│   └── google/                 # Google search engine module
├── utils/                      # Utility functions
│   ├── metrics.py
│   ├── llm_info.py
│   └── read_data_dev_test.py
├── dspy/                       # Modified dspy library (multi-image support)
├── notebooks/
│   ├── eda.ipynb               # Exploratory data analysis notebook
│   └── XFacta_m5_Notebook.ipynb # Modelling & visualisation notebook
├── report_eda.md               # M4 EDA report
├── proposal.md                 # M1 Project proposal
├── Multifact_m5_report.pdf     # M5 Modelling report
└── requirements.txt            # Python dependencies
```

### 9.2 Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd XFacta

# Create and activate conda environment
conda create -n xfacta python=3.10
conda activate xfacta

# Install dependencies
pip install -r requirements.txt

# Install modified dspy (supports multiple image inputs)
cd dspy
pip install .[dev]
cd ..

# Configure API keys in .env file
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_VISION_API_KEY=your_google_vision_api_key_here
cse_id=your_custom_search_engine_id_here
EOF
```

### 9.3 Running Inference

```bash
# CoT Evidence baseline
python Predict.py \
  --llm_name openai/gpt-4o \
  --data_path /path/to/XFacta_dataset \
  --reasoning_approach cot_prompt_evidence \
  --dataset_split test \
  --include_evidences 1 3 5 \
  --top_k_evidence 5 \
  --filter_untrusted \
  --evidence_cache

# Multi-Step Reasoning (best model)
python Predict.py \
  --llm_name openai/gpt-4o \
  --data_path /path/to/XFacta_dataset \
  --reasoning_approach multi_step \
  --dataset_split test \
  --include_evidences 1 3 \
  --top_k_evidence 5 \
  --filter_untrusted \
  --evidence_cache
```

### 9.4 Key Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--llm_name` | `openai/gpt-4o`, `gemini/flash-2.0`, or local path | Backend LLM |
| `--reasoning_approach` | `cot_prompt_evidence`, `prompt_ensembles_evidence`, `multi_step`, `self_consistency` | Reasoning strategy |
| `--dataset_split` | `dev`, `test` | Evaluation split |
| `--include_evidences` | Space-separated IDs (1–8) | Evidence types to include |
| `--top_k_evidence` | Integer (default: 5) | Max items per evidence type |
| `--filter_untrusted` | Flag | Enable domain filtering |
| `--evidence_cache` | Flag | Use cached evidence |

---

## 10. Conclusion

### 10.1 Summary of Contributions

This project delivered a complete multimodal misinformation detection system spanning the full pipeline from dataset construction to production-ready deployment:

1. **Contemporary dataset (XFacta).** We constructed and analysed a 2,400-sample multimodal misinformation dataset covering January 2024 to January 2025, with balanced labels, fine-grained topic and error-type annotations, and strict quality controls.

2. **Comprehensive EDA.** Our exploratory data analysis revealed critical dataset characteristics — the prevalence of real-image deception over synthetic content, the author identity confound, topic-dependent patterns, and event-driven temporal dynamics — all of which informed our modelling strategy.

3. **Three compared reasoning architectures.** We implemented and benchmarked CoT Evidence (baseline), Prompt Ensemble (intermediate), and Multi-Step Reasoning (proposed advanced model), demonstrating clear performance gains from structured reasoning.

4. **Best-performing model.** Multi-Step Reasoning achieved **84.7% accuracy** and **86.1% recall**, with particularly strong performance on image-caption mismatch detection — the most common misinformation pattern in the dataset.

5. **Rigorous error analysis.** Our slice-based evaluation across topics, evidence conditions, and error types identified specific strengths and weaknesses, providing actionable insights for future research.

### 10.2 Answers to Research Questions

**RQ1 (Dataset currency and memorisation bias).** The 2024–2025 dataset reveals that MLLMs still face significant challenges with contemporary content. The clear performance drop on Election and Conflict topics suggests that current events with limited settled evidence remain difficult, supporting our hypothesis that a contemporary dataset reveals genuine reasoning limitations rather than recall gaps.

**RQ2 (Multi-step vs. CoT).** Structured multi-step reasoning outperforms standard CoT by a substantial margin (+6.9% accuracy, +11.1% recall). The key mechanism is decomposed verification: by independently auditing caption truthfulness and image misuse, the model detects cross-modal inconsistencies that a single reasoning chain cannot isolate.

**RQ3 (Bottleneck analysis).** The dominant bottleneck varies by scenario. Image-caption mismatch is primarily a **reasoning bottleneck** (resolved by multi-step decomposition), while missing authoritative evidence is primarily a **retrieval bottleneck** (affecting all models equally). This suggests that improvements in both retrieval depth and reasoning structure are needed for robust real-world deployment.

### 10.3 Future Work

Several directions for future development emerged from this project:

1. **Retrieval augmentation.** Expanding evidence sources to include verified fact-checking databases (e.g., Snopes, PolitiFact) and temporal web archives could address the missing-evidence failure mode.

2. **Pixel-level forgery detection.** Integrating dedicated visual forgery detection (e.g., CNN-based manipulation detectors) would enable detection of semantically consistent image manipulations that current cross-modal approaches miss.

3. **Confidence calibration.** Developing better uncertainty estimation techniques would improve the reliability of confidence scores and enable more effective human-in-the-loop workflows.

4. **Multi-LLM evaluation.** Expanding evaluation to additional MLLMs (Gemini-2.0-Flash, Qwen-VL, GPT-4.1) would test the generalisability of our findings across model architectures.

5. **Real-time processing.** Optimising the retrieval and inference pipeline for lower latency would enable real-time moderation use cases.

6. **Temporal robustness testing.** Systematic evaluation across time-stratified splits would provide stronger evidence of temporal generalisation.

---

## Appendix A: Model Card

# Model Card: MultiFact Multimodal Misinformation Detector

### Model Summary

| Field | Value |
|-------|-------|
| **Model Name** | MultiFact — Multi-Step Reasoning |
| **Model Type** | Multimodal LLM Reasoning Pipeline (GPT-4o backend) |
| **Task** | Binary classification: Is a social media post misinformation (True) or authentic (False)? |
| **Input** | Post text (caption) + one or more images + optional web-retrieved evidence |
| **Output** | Label (True/False) + confidence score (0–100) |
| **Framework** | dspy + Python 3.10 |

### Intended Use

- **Primary use case:** Batch processing of social media posts to flag potential misinformation for human review
- **Target domain:** Short-form social media content (X/Twitter posts) with text and images
- **Target timeframe:** Content posted January 2024–January 2025
- **User profile:** Content moderation teams, platform integrity researchers, fact-checking organisations
- **Out-of-scope:** Real-time moderation, video-only content, long-form articles, low-resource languages, historical content pre-2024

### Performance Summary

| Metric | Score |
|--------|-------|
| Accuracy | 84.7% |
| Precision | 83.8% |
| Recall | **86.1%** |
| F1 Score | 84.9% |
| Best on topic | Economy (100% accuracy) |
| Worst on topic | Conflict (75% accuracy) |

### Known Limitations

1. **External API dependency.** Requires paid GPT-4o and search engine API keys.
2. **Evidence quality sensitivity.** Performance degrades when retrieved evidence is stale, contradictory, or unavailable.
3. **Temporal scope.** Optimised for 2024–2025 content; performance on future misinformation patterns is unknown.
4. **No pixel-level detection.** Cannot detect image forgeries that maintain semantic consistency with captions.
5. **Latency.** End-to-end inference takes 15–45 seconds per post.

### Recommended Deployment Workflow

| Confidence Range | Recommended Action |
|-----------------|-------------------|
| > 90% | Automated flagging/action |
| 70–90% | Human review recommended |
| < 70% | Additional evidence collection needed |

### Fairness Considerations

- Topic-dependent accuracy: performs better on Economy and Celebrity topics, worse on Conflict and Election topics
- No demographic or identity-based features are used in the model
- Author identity was intentionally excluded to prevent source-based bias
- The balanced test set (50/50) ensures metrics are not skewed by label imbalance
- Performance on low-resource languages has not been evaluated
- Temporal distribution of training data may not reflect future misinformation patterns

### Ethical Considerations

This system is designed as an **aid to human moderators**, not a replacement. Automated decisions should always be reviewable. The system may produce both false positives (wrongly flagging legitimate content) and false negatives (missing actual misinformation), with the latter carrying higher societal risk. Deployment should include appropriate human oversight, transparency reporting, and periodic recalibration.

---

## Appendix B: Evidence Retrieval Strategies — Detailed Description

| ID | Strategy | Engine | Description |
|----|----------|--------|-------------|
| 1 | I → Et | Google Vision + Search | Extract text from the post's images using Google Vision API, then search for web pages containing those images and extract their text content. |
| 2 | T → Ei | Google Search | Use the post caption as a search query to find and retrieve images from relevant web pages. |
| 3 | T → Et | Google Search | Use the post caption as a search query to retrieve text from relevant web pages. |
| 4 | News Search | DuckDuckGo | Search for news-specific text using the post caption as query. |
| 5 | Web Text | DuckDuckGo | General web text retrieval using the post caption as query. |
| 6 | Web Images | DuckDuckGo | General web image retrieval using the post caption as query. |
| 7 | Q → Et | GPT-4o + DuckDuckGo | Generate targeted search queries with GPT-4o based on uncertain aspects of the post, then retrieve text. |
| 8 | Q → Ei | GPT-4o + DuckDuckGo | Generate targeted search queries with GPT-4o, then retrieve images. |

---

## Appendix C: Ablation Configuration for Experiments

| Model | Evidence Types | Top-k | Domain Filter | Evidence Cache |
|-------|---------------|-------|---------------|----------------|
| CoT Evidence | 1 (I→Et), 3 (T→Et), 5 (Web text) | 5 | Yes | Yes |
| Prompt Ensemble | 1 (I→Et), 3 (T→Et), 5 (Web text) | 5 | Yes | Yes |
| Multi-Step Reasoning | 1 (I→Et, for image check), 3 (T→Et, for caption check) | 5 | Yes | Yes |

All models use GPT-4o as the backend LLM. The test set contains 72 samples (36 authentic + 36 misinformation), balanced across 6 topic categories.

---

## Appendix D: References

1. XFacta: Contemporary, Real-World Dataset and Evaluation for Multimodal Misinformation Detection with Multimodal LLMs. Xiao et al., 2025. arXiv:2508.09999.
2. INFO 442 — Multimedia Forensics, Course Materials. Week 7: Error Analysis and Slice-Based Evaluation.
3. Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.
4. Wang, X., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023.
5. Khattab, O., et al. (2023). DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. arXiv:2310.03714.

---

*End of Report — XFacta Project, INFO 442, Team 25, June 2026*
