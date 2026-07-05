# Model Card: MultiFact Multimodal Misinformation Detector

**Project:** MultiFact — INFO 442, Team 25  
**Date:** June 29, 2026

---

## Model Summary

| Field | Value |
|-------|-------|
| **Model Name** | MultiFact — Multi-Step Reasoning (Best Performing) |
| **Model Type** | Multimodal LLM Reasoning Pipeline with Staged Verification |
| **Backend LLM** | GPT-4o (OpenAI) |
| **Task** | Binary classification: Is a social media post **misinformation (False)** or **authentic (True)**? |
| **Input** | Post caption (text) + one or more images + optional web-retrieved evidence |
| **Output** | Label (True/False) + confidence score (0–100) |
| **Framework** | dspy + Python 3.10 |

## Intended Use

- **Primary:** Batch processing social media posts to flag potential misinformation for human moderator review
- **Domain:** Short-form social media content (X/Twitter) with text + images, posted January 2024–January 2025
- **Users:** Content moderation teams, platform integrity researchers, fact-checking organisations
- **Out of scope:** Real-time moderation, video-only content, long-form articles, non-English content, pre-2024 historical content

## Performance

| Metric | CoT Baseline | Prompt Ensemble | **Multi-Step (Best)** |
|--------|-------------|----------------|----------------------|
| Accuracy | 77.8% | 81.9% | **84.7%** |
| Precision | 79.4% | 82.9% | **83.8%** |
| Recall | 75.0% | 80.6% | **86.1%** |
| F1 Score | 77.1% | 81.7% | **84.9%** |
| False Negatives (of 36 fake) | 9 | 7 | **5** |
| False Positives (of 36 real) | 7 | 6 | **6** |

**Accuracy by topic (Multi-Step):** Economy 100%, Celebrity 83%, Health 92%, Election 83%, Conflict 75%, Disaster 75%

## Architecture

The Multi-Step pipeline decomposes verification into three stages:
1. **Caption authenticity check** — is the text factually accurate?
2. **Image misuse detection** — is the image used deceptively?
3. **Final verdict** — synthesises both checks, labels as False only if image misuse could mislead readers

Each stage records intermediate reasoning traces for auditability.

## Known Limitations

- **External API dependency:** Requires paid GPT-4o and search engine API credentials; cannot run fully offline
- **Evidence sensitivity:** Performance drops ~8% when retrieved evidence is scarce or low-quality
- **Temporal scope:** Optimised for 2024–2025 content; unknown generalisation to future patterns
- **No pixel-level detection:** Cannot detect image forgeries that maintain semantic consistency with captions
- **Latency:** 15–45 seconds per post (including web retrieval)
- **Cost:** ~$0.10–0.20 per inference at GPT-4o pricing
- **Topic bias:** Worse on Conflict and Election topics (contested claims with limited evidence)

## Deployment Recommendations

| Confidence | Action |
|------------|--------|
| > 90% | Automated flagging |
| 70–90% | Human review recommended |
| < 70% | Additional evidence needed |

This system is a decision **aid for human moderators**, not a replacement. Low-confidence or high-impact decisions always require human review.

## Fairness

- Author identity excluded to prevent source-based bias
- Balanced test set (50/50) ensures fair metric interpretation
- No demographic features used
- Performance on low-resource languages not evaluated
