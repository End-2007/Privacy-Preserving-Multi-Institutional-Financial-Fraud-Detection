# FedGuard

### Privacy-Preserving Financial Fraud Detection Using Machine Learning and Explainable AI

---

## 1. Project Idea and Motivation

Financial fraud schemes—especially money-laundering "mule" networks—routinely operate across multiple banks and payment processors. Because each financial institution only monitors its own transactions, no single bank has full visibility into the fraud chain.

At the same time, privacy regulations (such as Reserve Bank of India directions, the Digital Personal Data Protection Act 2023, and GDPR) strictly prohibit banks from pooling raw customer transaction records.

**FedGuard** addresses this challenge in stages. While the long-term vision incorporates Federated Learning across institutions, the **Micro Project** establishes the foundational machine learning and explainability layer.

---

## 2. Micro Project Overview

In the Micro Project, we simulate two distinct financial institutions using public, human-readable datasets:
- **Institution 1 (Mobile Money):** Simulated using the **PaySim** dataset (peer-to-peer transfers and cash-out operations).
- **Institution 2 (Card / E-Commerce):** Simulated using the **IEEE-CIS** dataset (card transactions, identity attributes, and device info).

The diagram below illustrates how transactions flow through the Micro Project pipeline:

```mermaid
flowchart TD
    subgraph DataSources["Simulated Financial Institutions"]
        D1["Institution 1: Mobile Money\n(PaySim Dataset)"]
        D2["Institution 2: Card / E-Commerce\n(IEEE-CIS Dataset)"]
    end

    subgraph MicroPipeline["FedGuard Micro Pipeline"]
        P["1. Data Cleaning and Preprocessing"]
        S["2. Imbalance Handling (SMOTE)"]
        M["3. Machine Learning Baselines\n(XGBoost / LightGBM)"]
        X["4. Explainable AI Layer\n(SHAP Decision Attributions)"]

        P --> S
        S --> M
        M --> X
    end

    subgraph Deliverables["Outputs for Review"]
        R1["Fraud Risk Score (0 to 100%)"]
        R2["Human-Readable Explanation\nfor Compliance Auditors"]
        R3["Performance Benchmarks\n(PR-AUC, Recall, FPR)"]
    end

    D1 --> P
    D2 --> P
    X --> R1
    X --> R2
    X --> R3

    style DataSources fill:#F0F4F8,stroke:#90A4AE,stroke-width:1px
    style MicroPipeline fill:#F9FBE7,stroke:#AFB42B,stroke-width:1px
    style Deliverables fill:#FCE4EC,stroke:#D81B60,stroke-width:1px

    style D1 fill:#E1F5FE,stroke:#0288D1,stroke-width:2px,color:#01579B
    style D2 fill:#E8EAF6,stroke:#3949AB,stroke-width:2px,color:#1A237E
    style P fill:#E8F5E9,stroke:#43A047,stroke-width:2px,color:#1B5E20
    style S fill:#FFF8E1,stroke:#FFA000,stroke-width:2px,color:#E65100
    style M fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px,color:#4A148C
    style X fill:#E0F7FA,stroke:#00ACC1,stroke-width:2px,color:#006064
    style R1 fill:#FFEBEE,stroke:#E53935,stroke-width:2px,color:#B71C1C
    style R2 fill:#E8F8F5,stroke:#16A085,stroke-width:2px,color:#0E6251
    style R3 fill:#FFF3E0,stroke:#FB8C00,stroke-width:2px,color:#E65100
```

---

## 3. Why These Datasets?

Many academic papers rely on the Kaggle European Credit Card dataset. However, that dataset anonymizes all feature names into mathematical principal components (`V1` through `V28`). 

If a model flags a transaction and explains it as `"V14 was -4.2"`, a compliance officer cannot understand or defend that flag in an audit.

FedGuard deliberately uses **PaySim** and **IEEE-CIS** because they retain real, human-readable attributes:
- **PaySim:** Account balances, transaction types (`TRANSFER`, `CASH_OUT`), and amount shifts.
- **IEEE-CIS:** Card networks, transaction amounts, email domains, and device types.

This allows the **SHAP** explainability layer to generate clear, plain-language audit trails (e.g., *"Entire sender balance transferred out to an account with zero prior history"*).

---

## 4. Multi-Stage Roadmap

The project is structured across four academic stages, with **Micro** as the current active foundation:

| Stage | Status | Focus Area | Planned Deliverables |
|---|:---:|---|---|
| <img src="https://img.shields.io/badge/Stage-Micro-00C853?style=for-the-badge" alt="Micro" /> | <img src="https://img.shields.io/badge/Status-ACTIVE-00C853?style=flat-square" alt="Active" /> | **Centralized Baselines & Explainability** | Data cleaning, SMOTE rebalancing, XGBoost/LightGBM baselines, and SHAP decision explanations |
| <img src="https://img.shields.io/badge/Stage-Mini-FBC02D?style=for-the-badge" alt="Mini" /> | <img src="https://img.shields.io/badge/Status-PLANNED-FBC02D?style=flat-square" alt="Planned" /> | **Federated Learning** | Cross-institution collaborative training using Flower (FedAvg) without sharing raw customer data |
| <img src="https://img.shields.io/badge/Stage-Minor-FB8C00?style=for-the-badge" alt="Minor" /> | <img src="https://img.shields.io/badge/Status-PLANNED-FB8C00?style=flat-square" alt="Planned" /> | **Differential Privacy** | Mathematical privacy guarantees ($\varepsilon, \delta$) via Opacus to protect model update gradients |
| <img src="https://img.shields.io/badge/Stage-Major-E53935?style=for-the-badge" alt="Major" /> | <img src="https://img.shields.io/badge/Status-PLANNED-E53935?style=flat-square" alt="Planned" /> | **Secure Aggregation** | Cryptographic protection (HE / SMPC) so the central server never sees plaintext model weights |

For complete technical specifications and data flow mechanics, refer to [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 5. Repository Structure

```
FedGuard/
├── README.md             # Project idea, overview, and roadmap
├── ARCHITECTURE.md       # Technical system flow and design details
├── requirements.txt      # List of dependencies
├── .gitignore            # Git rules excluding datasets and local documents
├── data/
│   ├── README.md         # Dataset descriptions and download instructions
│   ├── raw/              # Stored dataset CSVs (excluded from Git)
│   └── processed/        # Preprocessed data directory
└── docs/                 # Project documentation and review materials
```

---

## Academic and Team Details

- **Project Title:** Privacy-Preserving Multi-Institutional Financial Fraud Detection Using Federated Learning and Explainable AI
- **College:** Bharati Vidyapeeth's College of Engineering, New Delhi
- **Department:** Department of Computer Science and Engineering
- **Mentor:** Prof. Mohit Tiwari, Assistant Professor
- **Team Members:**
  - Naman Dugar (05911502724)
  - Anvi Tyagi (00211502724)
  - Riyanshu (05711502724)
