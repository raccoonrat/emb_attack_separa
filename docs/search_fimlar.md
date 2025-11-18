Nice — I searched the recent literature (2023–2025 preprints and conference papers) for explicit uses of **Chernoff information** (or Chernoff _bounds_/error exponents) in the context of **LLM watermark detection**. Short verdict up front:

**Short answer:**

* **Yes — Chernoff information / Chernoff bounds appear in the mathematical analysis of AI-generated-text detection and in several watermarking papers**, but **explicit, dedicated Chernoff-information theorems for LLM watermark detection are still uncommon**. Instead, many papers (especially those that frame detection as hypothesis testing) either _implicitly_ use the same large-deviation / error-exponent ideas or _explicitly_ apply Chernoff-style bounds to particular statistics. ([arXiv](https://arxiv.org/html/2304.04736?utm_source=chatgpt.com "On the Possibilities of AI-Generated Text Detection"))

What I found (high-value sources)
---------------------------------

1. **"On the Possibilities of AI-Generated Text Detection" — Chakraborty et al. (arXiv)**
   
   * This paper explicitly writes detection rates in terms of the **Chernoff information** (I_c(m,h)) and uses it to give exponential error-rate statements (total-variation / AUC growth with number of samples). This is a canonical example of applying Chernoff information to text-detection/hypothesis-testing for LLM outputs. ([arXiv](https://arxiv.org/html/2304.04736?utm_source=chatgpt.com "On the Possibilities of AI-Generated Text Detection"))

2. **"Baselines for Identifying Watermarked Large Language" (Tang et al., 2023)**
   
   * Uses **two-sided Chernoff bounds** to derive probabilistic bounds for misclassification rates under some watermarking baselines (gives concrete concentration bounds for their detection statistic). Good source for seeing Chernoff bounds applied to token-level / count statistics. ([arXiv](https://arxiv.org/pdf/2305.18456?utm_source=chatgpt.com "Baselines for Identifying Watermarked Large Language ..."))

3. **"A Statistical Framework of Watermarks for Large Language Models" (Li et al., 2024 slides / paper)**
   
   * Formalizes watermark detection as hypothesis testing and references asymptotic test efficiency (Herman Chernoff). The paper frames detection efficiency in an information-theoretic way — a natural place where Chernoff information would fit (they principally use LR/efficiency language). ([arXiv](https://arxiv.org/html/2404.01245?utm_source=chatgpt.com "A Statistical Framework of Watermarks for Large Language ..."))

4. **Survey / technique papers & detectors**
   
   * Surveys and robust-detector papers (e.g., RADAR and other detection surveys) reference Chernoff information when discussing theoretical detection limits / error exponents for distinguishing distributions (they may not produce a single Chernoff-info formula for a watermark scheme, but they point to the same family of arguments). ([OpenReview](https://openreview.net/pdf?id=QGrkbaan79&utm_source=chatgpt.com "RADAR: Robust AI-Text Detection via Adversarial Learning"))

5. **Very recent / 2025 preprints**
   
   * I found very recent preprints (2025) explicitly using Chernoff bounds in parts of their watermark analysis (e.g., _Multi-use LLM Watermarking and the False Detection Problem_, Fu et al., 2025 uses Chernoff bounds in bounding false-detection tails). This indicates adoption of Chernoff tools is increasing in the 2024–2025 literature. ([arXiv](https://arxiv.org/pdf/2506.15975?utm_source=chatgpt.com "Multi-use LLM Watermarking and the False Detection ..."))

6. **Classic digital-watermark literature**
   
   * Outside LLMs, classic watermark detection theory (images/audio) has long used **Chernoff information** and likelihood-ratio asymptotics to quantify optimal detection exponents — these methods translate conceptually to token/statistic-based watermarks for LLMs. ([ResearchGate](https://www.researchgate.net/publication/228607191_Optimal_Detection_of_Multiplicative_Watermarks?utm_source=chatgpt.com "(PDF) Optimal Detection of Multiplicative Watermarks"))

What this means practically / gaps & opportunities
--------------------------------------------------

* **Practical implication:** If you model the watermarked vs un-watermarked text distributions (or the distribution of a chosen detection statistic) and you can (even approximately) treat observations as i.i.d. or weakly dependent, **Chernoff information gives the optimal exponential rate** at which detection error decays with sample length. Several papers either exploit that directly or use Chernoff-style concentration to bound false positives/negatives. ([arXiv](https://arxiv.org/html/2304.04736?utm_source=chatgpt.com "On the Possibilities of AI-Generated Text Detection"))

* **Why explicit Chernoff statements are not ubiquitous:** Real LLM outputs are not i.i.d.; token dependencies, paraphrasing attacks, and complex editing operations make exact Chernoff-info calculations hard. So many authors prefer deriving bounds for a chosen pivot statistic (counts, LR approximations) rather than computing a closed-form Chernoff information between high-dimensional sequence distributions. ([arXiv](https://arxiv.org/html/2404.01245?utm_source=chatgpt.com "A Statistical Framework of Watermarks for Large Language ..."))

* **Research opportunity:** A useful and rigorous contribution would be deriving Chernoff-information (or tight upper/lower bounds on it) for realistic watermark models (e.g., green/ red token splits, or sampling-with-bias schemes) under explicit adversary models (bounded edits, paraphrase models, or adversarial re-sampling). Fu et al. (2025) shows these tools are already starting to appear. ([arXiv](https://arxiv.org/pdf/2506.15975?utm_source=chatgpt.com "Multi-use LLM Watermarking and the False Detection ..."))

Concrete next steps I can do right now (pick any)
-------------------------------------------------

* **A. Extract and summarize the exact Chernoff-type theorem statements** from Chakraborty et al., Tang et al., Li et al., and the Fu et al. 2025 preprint — side-by-side (assumptions, statistic, bound/exponent). (I can do this immediately.) ([arXiv](https://arxiv.org/html/2304.04736?utm_source=chatgpt.com "On the Possibilities of AI-Generated Text Detection"))

* **B. Take a simple watermark model (e.g., binary green/red token process) and compute the Chernoff information for that model** (analytically where possible, otherwise numeric). That would give an explicit error-exponent for detection vs text length under i.i.d. assumptions.

* **C. Produce an annotated bibliography (PDF) collecting the exact passages where Chernoff/Chernoff bounds are used in LLM watermarking/detection papers (2023–2025).**

Which one would you like? I can start with **A (extract exact theorem statements)** immediately and paste concise theorem-level quotes + short plain-English explanation for each.
