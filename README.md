# FedGuard: Banking Fraud Detection System
### Privacy-Preserving Multi-Institutional Financial Fraud Detection Using Machine Learning and Explainable AI

<p align="center">
  <img src="https://img.shields.io/badge/Project_Stage-🟢_MICRO_PROJECT_(ACTIVE)-00C853?style=for-the-badge" alt="Micro Project Stage" />
  <img src="https://img.shields.io/badge/Academic_Year-2026--2027-2979FF?style=for-the-badge" alt="Academic Year" />
  <img src="https://img.shields.io/badge/Domain-FinTech_&_Security-AA00FF?style=for-the-badge" alt="Domain" />
  <img src="https://img.shields.io/badge/Affiliation-BVCOE_New_Delhi-FF6D00?style=for-the-badge" alt="Affiliation" />
</p>

---

## 1. Project Overview and Core Idea

Digital banking and instant payment systems have expanded exponentially. Concurrently, organized fraud rings execute coordinated multi-hop transactions (such as money-mule networks) across **multiple independent banks and payment processors**. 

Because each financial institution only observes its own siloed records, no single bank captures the global fraud pattern.

```mermaid
flowchart LR
    subgraph Problem["The Modern Banking Dilemma"]
        direction TB
        A["Institution A\n(Mobile Wallet)"] -->|"Sees partial transfer"| M["Money-Mule Ring\n(Cross-Bank Scheme)"]
        B["Institution B\n(Commercial Bank)"] -->|"Sees partial cash-out"| M
        C["Privacy Laws\n(RBI Directions, DPDP 2023, GDPR)"] -.->|"Legally blocks data pooling"| D["No Shared Raw Data Allowed"]
    end

    subgraph Solution["FedGuard Micro Strategy"]
        direction TB
        S1["Heterogeneous Datasets\n(PaySim + IEEE-CIS)"] --> S2["SMOTE / SMOTE-ENN\nImbalance Correction"]
        S2 --> S3["Independent Boosted Classifiers\n(XGBoost / LightGBM)"]
        S3 --> S4["TreeSHAP Explainable AI\n(Audit-Ready Rationale)"]
    end

    style Problem fill:#FFF3E0,stroke:#E65100,stroke-width:2px
    style Solution fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px
    style A fill:#FFE082,stroke:#FFA000,stroke-width:2px,color:#000
    style B fill:#FFE082,stroke:#FFA000,stroke-width:2px,color:#000
    style C fill:#FFCDD2,stroke:#C62828,stroke-width:2px,color:#000
    style D fill:#FF8A80,stroke:#B71C1C,stroke-width:2px,color:#000
    style S1 fill:#81D4FA,stroke:#0277BD,stroke-width:2px,color:#000
    style S2 fill:#FFCC80,stroke:#EF6C00,stroke-width:2px,color:#000
    style S3 fill:#CE93D8,stroke:#6A1B9A,stroke-width:2px,color:#000
    style S4 fill:#A5D6A7,stroke:#2E7D32,stroke-width:2px,color:#000
```

> [!IMPORTANT]
> **Current Submission Scope:** This repository covers the foundational **MICRO PROJECT** — establishing empirical centralized baselines, resolving extreme class imbalance, and integrating human-readable SHAP explanations ahead of future federated learning extensions.

---

## 2. Micro Project Workflow Architecture

The flowchart below details the end-to-end data pipeline, modeling, and explainability layer planned for the **Micro Project**:

```mermaid
flowchart TD
    classDef mobile fill:#00B0FF,stroke:#01579B,stroke-width:3px,color:#ffffff,font-weight:bold;
    classDef card fill:#2979FF,stroke:#0D47A1,stroke-width:3px,color:#ffffff,font-weight:bold;
    classDef clean fill:#00E676,stroke:#007E33,stroke-width:2px,color:#000000,font-weight:bold;
    classDef smote fill:#FFAB00,stroke:#FF6D00,stroke-width:2px,color:#000000,font-weight:bold;
    classDef ml fill:#AA00FF,stroke:#4A148C,stroke-width:3px,color:#ffffff,font-weight:bold;
    classDef xai fill:#00C853,stroke:#1B5E20,stroke-width:3px,color:#ffffff,font-weight:bold;
    classDef ui fill:#FF1744,stroke:#B71C1C,stroke-width:3px,color:#ffffff,font-weight:bold;

    subgraph DataIngestion["STAGE 1: Dual-Institution Data Ingestion"]
        P["PaySim Mobile Dataset\n(6.3M records | 11 columns)\nBalance changes & P2P transfers"]:::mobile
        I["IEEE-CIS Card Dataset\n(~590K records | 435 columns)\nTransaction & Identity relational join"]:::card
    end

    subgraph Preprocessing["STAGE 2: Cleaning & Feature Engineering"]
        P_Prep["PaySim Pipeline\n• Balance Error: errorBalOrig/errorBalDest\n• Label Encode 'type' (TRANSFER, etc.)\n• Drop non-predictive name strings"]:::clean
        I_Prep["IEEE-CIS Pipeline\n• Relational join on TransactionID\n• Impute sparse Vesta V-features\n• Encode card, email domain & device info"]:::clean
    end

    subgraph Imbalance["STAGE 3: Class-Imbalance Correction"]
        P_Smote["SMOTE / SMOTE-ENN\nRaw fraud: ~0.13%\nMinority synthetic interpolation"]:::smote
        I_Smote["SMOTE / SMOTE-ENN\nRaw fraud: ~3.5%\nBorderline sample cleaning"]:::smote
    end

    subgraph Modeling["STAGE 4: Centralized Baseline Classifiers"]
        M1["PaySim XGBoost / LightGBM\nIndependent Tree Baseline\nStratified Train/Val/Test Split"]:::ml
        M2["IEEE-CIS XGBoost / LightGBM\nIndependent Tree Baseline\nStratified Train/Val/Test Split"]:::ml
    end

    subgraph XAI["STAGE 5: Explainable AI Layer (SHAP)"]
        S["TreeSHAP Exact Feature Attribution\nCalculates Shapley impact values per feature\nGenerates audit-ready plain-language justifications"]:::xai
    end

    subgraph Dashboard["STAGE 6: Compliance Monitoring Prototype"]
        UI["Compliance Audit Dashboard\n• Real-Time Risk Probability (0-100%)\n• Natural Language Decision Rationale\n• PR-AUC & FPR Benchmark Comparison"]:::ui
    end

    P --> P_Prep --> P_Smote --> M1 --> S
    I --> I_Prep --> I_Smote --> M2 --> S
    S --> UI

    linkStyle default stroke:#37474F,stroke-width:2px;
```

---

## 3. Dataset Strategy: Why Standard PCA Datasets Fail

During review evaluations, examiners frequently ask:
> *"Why not use the popular Kaggle European Credit Card dataset?"*

```mermaid
flowchart TD
    classDef bad fill:#FFCDD2,stroke:#B71C1C,stroke-width:2px,color:#B71C1C;
    classDef good fill:#C8E6C9,stroke:#1B5E20,stroke-width:2px,color:#1B5E20;

    subgraph TypicalDataset["Traditional Anonymized Kaggle Dataset"]
        direction LR
        K1["Feature V1, V2 ... V28\n(Anonymized PCA)"] --> K2["SHAP Output:\n'Flagged because V14 = -4.2'"]
        K2 --> K3["Unusable for Bank Compliance & RBI Audits"]
    end

    subgraph FedGuardStrategy["FedGuard Human-Readable Strategy"]
        direction LR
        F1["PaySim + IEEE-CIS Features\n(Balances, Card Type, Device)"] --> F2["SHAP Output:\n'TRANSFER + emptied balance to new account'"]
        F2 --> F3["Audit-Ready for RBI & DPDP Compliance"]
    end

    TypicalDataset:::bad
    FedGuardStrategy:::good
```

### Institutional Simulation Breakdown
| Institution Simulated | Dataset Selected | Characteristics | Role in Micro Pipeline |
|----------------------|------------------|-----------------|------------------------|
| **Mobile Money Operator** | **PaySim** (471 MB) | Step, Amount, Old/New Balances, P2P Transfers | Evaluates account-drainage and mule cash-outs |
| **Card Network / Gateway** | **IEEE-CIS** (~677 MB) | Card Network, Email Domain, Device OS, TransactionAmt | Evaluates card-not-present and identity spoofing |

---

## 4. Specialized Fraud Evaluation Framework

Traditional machine learning relies on accuracy, but financial fraud is an **extreme needle-in-a-haystack problem** ($<1\%$ fraud). FedGuard's evaluation framework prioritizes:

```mermaid
quadrantChart
    title Operational Fraud Evaluation Matrix
    x-axis "Low Business Impact" --> "High Business Impact"
    y-axis "Misleading on Skewed Data" --> "Reliable on Skewed Data"
    quadrant-1 "Primary Focus (PR-AUC, Recall)"
    quadrant-2 "Secondary Check (F1-Score)"
    quadrant-3 "Misleading (Raw Accuracy)"
    quadrant-4 "Operational Cost (False Positive Rate)"
    "Raw Accuracy": [0.2, 0.15]
    "ROC-AUC": [0.45, 0.45]
    "Precision": [0.7, 0.75]
    "Recall (TPR)": [0.85, 0.85]
    "PR-AUC": [0.9, 0.95]
    "False Positive Rate": [0.85, 0.4]
```

* **PR-AUC (Precision-Recall Area Under Curve):** The decisive metric for highly imbalanced fraud datasets; completely immune to majority-class inflation.
* **Recall (True Positive Rate):** Measures the direct proportion of financial fraud detected.
* **False Positive Rate (FPR):** Controls operational overhead — preventing legitimate customer cards and transactions from being erroneously blocked.

---

## 5. Repository Structure

```
FedGuard/
├── README.md                           # Main project idea & Micro pipeline overview
├── ARCHITECTURE.md                     # In-depth architectural & data-flow specification
├── requirements.txt                    # Planned Python environment dependencies
├── .gitignore                          # Strict gitignore protecting repository from large data
├── data/
│   ├── README.md                       # Dataset overview and setup details
│   ├── raw/                            # Contains PaySim & IEEE-CIS CSVs (git-ignored)
│   ├── processed/                      # Cleaned & resampled datasets (git-ignored)
│   └── sample/                         # Placeholder for small demo batches
└── docs/                               # Project documentation & review materials
```

---

## 6. Full Four-Stage Project Roadmap

While our current deliverable is strictly the **Micro Project**, the full technical trajectory progresses across all four academic phases:

```mermaid
timeline
    title FedGuard Academic Progression Roadmap
    section 🟢 MICRO (Current Active)
        Centralized Benchmarks : Ingestion of PaySim & IEEE-CIS
        Class Imbalance : Independent SMOTE & SMOTE-ENN
        Explainable AI : TreeSHAP plain-language decision rationale
        Interface : Lightweight monitoring dashboard prototype
    section 🟡 MINI (Planned)
        Federated Learning : Flower (flwr) FedAvg across simulated bank nodes
        Full-Stack Migration : FastAPI backend + React frontend with auth
        Benchmark Comparison : Centralized vs. Federated performance evaluation
    section 🟠 MINOR (Planned)
        Differential Privacy : Opacus PyTorch DP on model gradient updates
        Privacy-Utility Curve : Rigorous epsilon budget vs. recall trade-off analysis
        Production Database : Migration to PostgreSQL / Supabase
    section 🔴 MAJOR (Planned)
        Secure Aggregation : Cryptographic Homomorphic Encryption / SMPC
        Regulatory Validation : RBI Fraud Management & DPDP Act compliance audit
        Research Publication : Scopus-indexed conference paper submission
```

---

## Academic and Team Details

* **Project Title:** Privacy-Preserving Multi-Institutional Financial Fraud Detection Using Federated Learning and Explainable AI
* **Short Title:** FedGuard
* **College:** Bharati Vidyapeeth's College of Engineering, New Delhi
* **Department:** Department of Computer Science and Engineering
* **Project Mentor:** Prof. Mohit Tiwari, Assistant Professor, Dept. of CSE
* **Team Members:**
  * **Naman Dugar** (Enrollment No: 05911502724)
  * **Anvi Tyagi** (Enrollment No: 00211502724)
  * **Riyanshu** (Enrollment No: 05711502724)
