# 🏛️ FedGuard — Conceptual System Architecture

> **Focus:** This document outlines the conceptual system design, data flow, and methodology for the **Micro Project** stage of FedGuard. The future federated stages (Mini / Minor / Major) are summarized briefly at the end.

---

## 1. High-Level Micro Project Pipeline

The **Micro Project** focuses on building an explainable, centralized multi-institutional benchmark ahead of any future federation steps:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FedGuard Micro Project Architecture                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐      ┌──────────────┐                                   │
│   │   PaySim     │      │  IEEE-CIS    │    Two structurally distinct       │
│   │  (Mobile     │      │  (Card /     │    datasets simulating two         │
│   │   Money)     │      │  E-commerce) │    financial institution types     │
│   │  471 MB      │      │  ~677 MB     │                                   │
│   │  11 columns  │      │  394+41 cols │                                   │
│   └──────┬───────┘      └──────┬───────┘                                   │
│          │                     │                                            │
│          ▼                     ▼                                            │
│   ┌─────────────────────────────────────┐                                   │
│   │     STAGE 1: Data Preprocessing     │                                   │
│   │  ┌───────────────────────────────┐  │                                   │
│   │  │ • Null / duplicate handling   │  │                                   │
│   │  │ • Feature scaling             │  │                                   │
│   │  │ • Categorical encoding        │  │                                   │
│   │  │ • Schema alignment            │  │                                   │
│   │  └───────────────────────────────┘  │                                   │
│   └──────────────┬──────────────────────┘                                   │
│                  │                                                          │
│                  ▼                                                          │
│   ┌─────────────────────────────────────┐                                   │
│   │   STAGE 2: Class-Imbalance Fix      │                                   │
│   │  ┌───────────────────────────────┐  │                                   │
│   │  │ SMOTE / SMOTE-ENN            │  │                                   │
│   │  │ (applied independently       │  │                                   │
│   │  │  per dataset)                │  │                                   │
│   │  └───────────────────────────────┘  │                                   │
│   └──────────────┬──────────────────────┘                                   │
│                  │                                                          │
│          ┌───────┴────────┐                                                 │
│          ▼                ▼                                                 │
│   ┌────────────┐   ┌────────────┐                                          │
│   │ PaySim     │   │ IEEE-CIS   │   Independent centralized                │
│   │ Baseline   │   │ Baseline   │   classifiers per institution            │
│   │ (XGB/LGBM) │   │ (XGB/LGBM) │                                          │
│   └─────┬──────┘   └─────┬──────┘                                          │
│         │                │                                                  │
│         └───────┬────────┘                                                  │
│                 ▼                                                           │
│   ┌─────────────────────────────────────┐                                   │
│   │   STAGE 3: SHAP Explainability      │                                   │
│   │  ┌───────────────────────────────┐  │                                   │
│   │  │ Per-transaction feature       │  │                                   │
│   │  │ attributions + plain-language │  │                                   │
│   │  │ explanation templates         │  │                                   │
│   │  └───────────────────────────────┘  │                                   │
│   └──────────────┬──────────────────────┘                                   │
│                  │                                                          │
│                  ▼                                                          │
│   ┌─────────────────────────────────────┐                                   │
│   │   STAGE 4: Monitoring Dashboard     │                                   │
│   │  ┌───────────────────────────────┐  │                                   │
│   │  │ • Transaction table           │  │                                   │
│   │  │ • Risk scores & gauge         │  │                                   │
│   │  │ • SHAP audit explanations     │  │                                   │
│   │  │ • Model benchmark metrics     │  │                                   │
│   │  └───────────────────────────────┘  │                                   │
│   └─────────────────────────────────────┘                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. End-to-End Data Flow & Modular Breakdown

```
                        ┌──────────────────┐
                        │  Raw Data Files  │
                        │  (data/raw/)     │
                        └────────┬─────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 ▼                               ▼
        ┌────────────────┐              ┌────────────────────┐
        │    PaySim      │              │     IEEE-CIS       │
        │  6.3M rows     │              │   ~590K rows       │
        │  11 columns    │              │  394 + 41 columns  │
        │                │              │  (transaction +    │
        │  step, type,   │              │   identity tables) │
        └───────┬────────┘              └────────┬───────────┘
                │                                │
                ▼                                ▼
        ┌────────────────┐              ┌────────────────────┐
        │ Data Cleaning  │              │  Data Cleaning     │
        │ • Balance errs │              │  • Relational join │
        │ • Type encode  │              │  • Impute NaNs     │
        │ • Scale amount │              │  • Encode device/id│
        └───────┬────────┘              └────────┬───────────┘
                │                                │
                ▼                                ▼
        ┌────────────────┐              ┌────────────────────┐
        │ SMOTE / ENN    │              │ SMOTE / ENN        │
        │ Fraud: ~0.13%  │              │ Fraud: ~3.5%       │
        │ → Rebalanced   │              │ → Rebalanced       │
        └───────┬────────┘              └────────┬───────────┘
                │                                │
                ▼                                ▼
        ┌────────────────────────────────────────────────────┐
        │             Independent Centralized Classifiers     │
        │  • Logistic Regression (Linear Benchmark)          │
        │  • Random Forest (Bagged Benchmark)                │
        │  • XGBoost / LightGBM (Gradient-Boosted Baseline)  │
        └───────────────────────┬────────────────────────────┘
                                │
                                ▼
        ┌────────────────────────────────────────────────────┐
        │           SHAP Explainability Layer                 │
        │  Converts TreeSHAP attributions into natural        │
        │  language justifications for human compliance:     │
        │  "Flagged: TRANSFER + zero dest balance + high amt"│
        └───────────────────────┬────────────────────────────┘
                                │
                                ▼
        ┌────────────────────────────────────────────────────┐
        │         Lightweight Monitoring Interface           │
        │  Displays live risk gauge, plain-English reasons,  │
        │  and per-institution benchmark comparisons.        │
        └────────────────────────────────────────────────────┘
```

---

## 3. Dataset Characteristics

### PaySim (Mobile Money Simulator)
* **Context:** Agent-based simulation based on real mobile money transaction logs.
* **Fields:** Transaction type (`TRANSFER`, `CASH_OUT`, etc.), amounts, initial and updated balances for origin and destination accounts.
* **Key Fraud Signatures:** Sudden account drainage via `TRANSFER` followed by `CASH_OUT` with zero destination balance history.

### IEEE-CIS (Card & E-commerce Fraud)
* **Context:** Real-world payment gateway transactions provided by Vesta Corp.
* **Fields:** Transaction amounts, card attributes (issuer, type, network), address distances, email domains, device fingerprints, and engineered behavioral features ($V_1$–$V_{339}$).
* **Key Fraud Signatures:** Card-not-present volume bursts, disposable/proxy email domains, browser/operating system mismatches.

---

## 4. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Dual Datasets over Single Split** | Real fraud networks span different entity types (mobile wallets vs card networks). Combining PaySim and IEEE-CIS creates genuinely non-IID conditions. |
| **SMOTE over Undersampling** | Undersampling discards up to 99% of genuine transaction patterns, leading to excessive False Positive Rates. |
| **TreeSHAP Integration** | Fast, exact polynomial-time Shapley computation for tree-based models, providing complete audit trails for every flagged transaction. |
| **PR-AUC as Primary Metric** | Traditional ROC-AUC can be misleadingly high on extremely imbalanced fraud datasets; PR-AUC strictly penalizes false alarms. |

---

## 5. Overview of Planned Future Stages

```
Current                    Planned
═══════                    ═══════

🟢 MICRO                   🟡 MINI              🟠 MINOR           🔴 MAJOR
Centralized               Federated             Differential       Secure
Baselines +               Learning              Privacy            Aggregation
SHAP + Dashboard          (Flower/FedAvg)       (Opacus)           (HE/SMPC)
                          + Full-Stack UI       + PostgreSQL       + Real-time
                          + FL vs Centralized   + ε vs Utility     + Production
                            benchmark             trade-off          compliance
```

* **🟡 Mini Project:** Introduces Federated Averaging (`Flower`) across simulated bank nodes so institutions collaboratively train without sharing raw records.
* **🟠 Minor Project:** Adds Differential Privacy (`Opacus`) on model weight updates to guarantee mathematical privacy ($\varepsilon, \delta$).
* **🔴 Major Project:** Incorporates cryptographic Secure Aggregation (Homomorphic Encryption / SMPC) to secure the aggregation server itself.
