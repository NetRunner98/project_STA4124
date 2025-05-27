
# C2GAM: Counterfactual Generation for Repeated Cross-Sectional Data

Final Project for STA4124 Generative Model Spring 2025, Yonsei University

This project extends the C2GAM framework to transform Repeated Cross-Sectional (RCS) data into pseudo-panel data, enabling counterfactual outcome generation and collider bias correction using deep generative models.

## Project Objective

To reconstruct individual-level sequences from repeated cross-sectional datasets (e.g., CENSUS), and infer missing outcomes (e.g., income) by leveraging panel data (e.g., KLIPS). The system uses KLIPS to model realistic income distributions and generates pseudo-individuals and outcomes for CENSUS entries using generative adversarial training.

---

## Dataset Overview

| Dataset   | Type                     | Description                               |
|-----------|--------------------------|-------------------------------------------|
| KLIPS     | Panel Data               | Contains pid, covariates, and `p_wage`    |
| CENSUS    | Repeated Cross-Sectional | No pid, no `p_wage`, but same covariates  |

### Common Covariates (X)
- `p_region`, `p_age`, `p_sex`, `p_married`, `p_edu`, `Ind`

---

## Data Pipeline
```bash
python run.py
```

### 1. **KLIPS Sequence Construction**
```bash
python preprocessing/klips_sequence_builder.py
```
- Groups rows by `pid`
- Creates sequences of length 5
- Outputs: `klips_census_sequences.npz`

### 2. **KLIPS Embedding via Transformer**
```bash
python preprocessing/transformer_encoder.py
```
- Applies Transformer to encode each KLIPS sequence
- Mean-pooling → latent vector
- Output: `matched_sequence_vectors.csv`

### 3. **CENSUS Matching via Cosine Similarity**
```bash
python preprocessing/census_embed_matcher.py
```
- Samples 1% of CENSUS and KLIPS
- Matches CENSUS (X) with KLIPS (Z) by cosine similarity
- Infers income `Y`, builds C2GAM input
- Output: `transformed_input_for_c2gam.csv`

### 4. **Selection Bias Injection**
```bash
python selection/make_selection.py
```
- Applies collider structure: `S ~ sigmoid(Y - 3*T + wᵀX + ε)`
- Splits data into:
  - `D_obs.csv` (S=1, observed)
  - `D_rep.csv` (representative)
  - `D_unsel.csv` (S=0, unobserved)

---

## C2GAM Architecture

| Module         | Input          | Output        | Role |
|----------------|----------------|---------------|------|
| Generator₁ (Gₛ) | (X, T)          | pseudo-id     | *pseudo-panel identity generation* |
| Generator₂ (G_d) | (X, T)         | `p_wage` (Y)  | *realistic income generation* |
| Discriminator₁ (Do) | sequence (X, T, Y) | real/fake | *real vs. generated sequence* |
| Discriminator₂ (Du) | (X, Y)        | real/fake     | *real vs. generated income distribution* |

> In current implementation, **G_d is active** (VAE decoder as income generator),  
> **Gₛ (pseudo-id generator) is planned but not yet implemented.**

---

## Model Training

```bash
python C2GAM_training/main.py
```
- Loads `D_obs.csv`, `D_rep.csv`, `D_unsel.csv`
- Trains VAE, GAN, and BNN estimator
- Evaluates counterfactual predictions via PEHE


## Notes

- `X1` is set as binary treatment: **high education = (`p_edu` ≥ 4)**
- `Y` (income) is generated from KLIPS representations
- Collider bias is injected through the S-selection function
- Only **G_d (income generator)** is actively trained and evaluated

---

## Evaluation (Planned)
- PEHE: Precision in Estimation of Heterogeneous Effect
- t-SNE: Distributional similarity of observed vs. generated samples

---

## Reference
- [C2GAM GitHub Repo](https://github.com/ZJUBaohongLi/C2GAM)
- [Original Paper (ICML 2024)] [A Generative Approach for Treatment Effect Estimation under Collider Bias: From an Out-of-Distribution Perspective](https://proceedings.mlr.press/v235/li24al.html)
