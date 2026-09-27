# 🛡️ FedGuard: Banking Fraud Detection System
### Privacy-Preserving Multi-Institutional Financial Fraud Detection Using Machine Learning and Explainable AI

[![Micro Project Stage](https://img.shields.io/badge/Stage-🟢_MICRO_IDEA_&_DESIGN-brightgreen?style=for-the-badge)](#-project-overview--idea-introduction)
[![Academic Project](https://img.shields.io/badge/Academic-BVCOE_CSE_2026--27-blue.svg?style=for-the-badge)](#-academic-context)

---

## 📌 Project Overview & Idea Introduction

**FedGuard** is a multi-institutional fraud detection concept designed to address two conflicting banking imperatives:
1. **The need for collective intelligence:** Financial fraud patterns (specifically money-laundering "mule" chains) routinely span multiple banks and payment processors. No single institution sees the complete fraud pattern in isolation.
2. **Strict data privacy compliance:** Regulations like RBI's Master Directions on Fraud Risk Management, India's **DPDP Act (2023)**, and the **EU GDPR** prohibit banks from pooling raw customer transaction data with one another.

### 🎯 Long-term Vision vs. Micro Project Focus
* **Long-Term Vision (Mini → Minor → Major):** Build a distributed Federated Learning system with Differential Privacy and Secure Multi-Party Aggregation so banks train a unified defense model without ever sharing raw customer data.
* **🟢 Current Focus — MICRO PROJECT (The Foundation):**
  * Establish the **centralized, empirical proof-of-concept**.
  * Acquire and prepare two distinct institutional datasets (**PaySim** for mobile money, **IEEE-CIS** for card/e-commerce).
  * Design the class-imbalance correction framework using **SMOTE / SMOTE-ENN**.
  * Formulate independent centralized benchmark classifiers (**XGBoost / LightGBM**).
  * Design a **SHAP Explainable AI (XAI)** auditing mechanism that produces human-readable justifications for every flagged transaction.
  * Plan a lightweight monitoring dashboard for compliance officers.

---

## 🏛️ Micro Project Workflow & Proposed Architecture

The diagram below outlines the full proposed workflow of the **Micro Project**:

```
═════════════════════════════════════════════════════════════════════════════════
                       FEDGUARD — MICRO PROJECT PIPELINE
═════════════════════════════════════════════════════════════════════════════════

  [ INSTITUTION 1: Mobile Money ]               [ INSTITUTION 2: Card / E-commerce ]
       PaySim Dataset (~471 MB)                     IEEE-CIS Dataset (~677 MB)
     (data/raw/PS_..._log.csv)                     (data/raw/train_transaction.csv +
         [ 6.3M Transactions ]                      data/raw/train_identity.csv)
                   │                                             │
                   ▼                                             ▼
  ┌─────────────────────────────────┐           ┌─────────────────────────────────┐
  │   1. Data Cleaning & Encoding   │           │   1. Data Cleaning & Encoding   │
  │   • Feature Engg (Balance Errs) │           │   • Merge transaction & identity│
  │   • Drop identifier strings     │           │   • Impute sparse V-features    │
  │   • Encode transaction types    │           │   • Encode card, email, device  │
  └────────────────┬────────────────┘           └────────────────┬────────────────┘
                   │                                             │
                   ▼                                             ▼
  ┌─────────────────────────────────┐           ┌─────────────────────────────────┐
  │   2. Class-Imbalance Handling   │           │   2. Class-Imbalance Handling   │
  │   • Original fraud: ~0.13%      │           │   • Original fraud: ~3.5%       │
  │   • SMOTE / SMOTE-ENN synthesis │           │   • SMOTE / SMOTE-ENN synthesis │
  └────────────────┬────────────────┘           └────────────────┬────────────────┘
                   │                                             │
                   ▼                                             ▼
  ┌─────────────────────────────────┐           ┌─────────────────────────────────┐
  │   3. Centralized ML Modeling    │           │   3. Centralized ML Modeling    │
  │   • XGBoost / LightGBM Baseline │           │   • XGBoost / LightGBM Baseline │
  │   • Stratified Train/Val/Test   │           │   • Stratified Train/Val/Test   │
  └────────────────┬────────────────┘           └────────────────┬────────────────┘
                   │                                             │
                   └──────────────────────┬──────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │    4. Explainable AI (SHAP Layer)      │
                      │   • TreeSHAP Exact Attributions        │
                      │   • Natural Language Decision Rationale│
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  5. Proposed Monitoring Dashboard      │
                      │   • Real-Time Risk Gauge (0-100%)      │
                      │   • Plain-English Audit Justification  │
                      │   • Cross-Institution Benchmark Metrics│
                      └────────────────────────────────────────┘
```

---

## 🔬 Dataset Rationale & Defense for Review

A common reviewer question during project evaluation:
> *"Why not use the popular Kaggle European Credit Card dataset?"*

```
Typical Kaggle Credit Card Dataset:
┌───────────┬───────────┬───────────┬───────────┐
│    V1     │    V2     │    ...    │    V28    │ ───► SHAP: "Flagged because V14 = -4.2"
└───────────┴───────────┴───────────┴───────────┘      ❌ Meaningless to bank compliance auditors!

FedGuard Dataset Design (PaySim + IEEE-CIS):
┌───────────┬───────────┬───────────┬───────────┐
│ Tx Type   │ Old Bal   │ Dest Bal  │ Device    │ ───► SHAP: "Flagged: Entire account balance
└───────────┴───────────┴───────────┴───────────┘      emptied via TRANSFER to new zero-balance dest"
                                                       ✅ Human-readable & regulatory audit-ready!
```

1. **Human-Readable Schema:** The Kaggle European dataset replaces feature names with anonymous PCA components ($V_1$ to $V_{28}$). Explaining fraud as *"feature $V_{14}$ was low"* is useless for bank risk officers and regulatory audits. PaySim and IEEE-CIS maintain real attributes (balances, transaction types, card types, device IDs).
2. **Authentic Cross-Institution Heterogeneity (Non-IID):** Rather than artificially splitting a single dataset, FedGuard models two distinct financial environments:
   * **PaySim:** Simulates a **Mobile Money / P2P Wallet** institution.
   * **IEEE-CIS:** Simulates a **Commercial Card Issuer / E-commerce Gateway**.

---

## 📋 Micro Project Deliverables & Scope Boundary

| Component | In Micro Scope? | Proposed Approach |
|-----------|:---------------:|-------------------|
| **Dataset Ingestion & Cleaning** | ✅ **YES** | Multi-table join, missing value imputation, schema alignment |
| **Class Imbalance Correction** | ✅ **YES** | SMOTE & SMOTE-ENN applied independently per institution |
| **Centralized Baseline Classifiers** | ✅ **YES** | Logistic Regression, Random Forest, XGBoost/LightGBM |
| **Explainable AI (SHAP)** | ✅ **YES** | TreeSHAP feature attributions converted to plain-English justification |
| **Monitoring Dashboard** | ✅ **YES** | Lightweight Streamlit dashboard for audit & benchmark review |
| **Federated Learning (Flower)** | ❌ *Mini Scope* | Documented as future progression roadmap |
| **Differential Privacy (Opacus)** | ❌ *Minor Scope* | Documented as future progression roadmap |
| **Secure Aggregation (HE/SMPC)** | ❌ *Major Scope* | Documented as future progression roadmap |

---

## 📈 Evaluation Framework (Fraud-Specific Metrics)

Financial fraud requires specialized evaluation criteria because fraud makes up **$< 1\%$** of all transactions:

```
                  ┌──────────────────────────────────────────────┐
                  │          Model Evaluation Strategy           │
                  ├──────────────────────┬───────────────────────┤
                  │  Metric              │  Why It Matters       │
                  ├──────────────────────┼───────────────────────┤
                  │  PR-AUC              │  Critical for severe  │
                  │  (Precision-Recall)  │  imbalance; unaffected│
                  │                      │  by true-negative skew│
                  ├──────────────────────┼───────────────────────┤
                  │  Recall (TPR)        │  Directly reflects    │
                  │                      │  prevented financial  │
                  │                      │  losses               │
                  ├──────────────────────┼───────────────────────┤
                  │  False Positive Rate │  Minimizes operational│
                  │  (FPR)               │  friction & false     │
                  │                      │  customer freezes     │
                  ├──────────────────────┼───────────────────────┤
                  │  F1-Score / ROC-AUC  │  Overall baseline     │
                  │                      │  discrimination       │
                  └──────────────────────┴───────────────────────┘
```

---

## 📁 Repository Structure

```
FedGuard/
├── README.md                           # Main project idea & Micro pipeline overview
├── ARCHITECTURE.md                     # In-depth architectural & data-flow specification
├── requirements.txt                    # Planned Python environment dependencies
├── .gitignore                          # Strict gitignore protecting repository from large data
├── data/
│   ├── README.md                       # Dataset overview and setup details
│   ├── raw/                            # 📁 Contains PaySim & IEEE-CIS CSVs (git-ignored)
│   ├── processed/                      # 📁 Cleaned & resampled datasets (git-ignored)
│   └── sample/                         # 📁 Placeholder for small demo batches
└── docs/                               # Project documentation & review materials
```

---

## 🗺️ Future Roadmap Beyond Micro (Brief Overview)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FUTURE STAGES AT A GLANCE                          │
├───────────────┬───────────────────────────────┬─────────────────────────────┤
│ Stage         │ Key Technical Addition        │ Research Objective          │
├───────────────┼───────────────────────────────┼─────────────────────────────┤
│ 🟡 MINI       │ Federated Learning (Flower)   │ Train shared model via      │
│               │ & Full-Stack UI               │ FedAvg across institutions  │
├───────────────┼───────────────────────────────┼─────────────────────────────┤
│ 🟠 MINOR      │ Differential Privacy (Opacus) │ Privacy budget (ε) vs       │
│               │ & PostgreSQL DB               │ accuracy trade-off analysis │
├───────────────┼───────────────────────────────┼─────────────────────────────┤
│ 🔴 MAJOR      │ Secure Aggregation (HE/SMPC)  │ Cryptographic protection of │
│               │ & Production Deployment       │ plaintext model weights     │
└───────────────┴───────────────────────────────┴─────────────────────────────┘
```

---

## 👥 Academic & Team Details

* **Project Title:** Privacy-Preserving Multi-Institutional Financial Fraud Detection Using Federated Learning and Explainable AI
* **Project Name:** FedGuard
* **Institution:** Bharati Vidyapeeth's College of Engineering, New Delhi
* **Department:** Department of Computer Science and Engineering
* **Project Mentor:** Prof. Mohit Tiwari, Assistant Professor, Dept. of CSE
* **Team Members:**
  * **Naman Dugar** (Enrollment No: 05911502724)
  * **Anvi Tyagi** (Enrollment No: 00211502724)
  * **Riyanshu** (Enrollment No: 05711502724)
