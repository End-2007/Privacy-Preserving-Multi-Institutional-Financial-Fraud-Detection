# FedGuard — System Architecture

This document details the technical architecture, data processing stages, and design decisions for the **Micro Project** phase of FedGuard.

---

## 1. System Pipeline Overview

The Micro Project establishes a centralized machine learning and explainability pipeline across two heterogeneous financial datasets:

```mermaid
flowchart TD
    D1["Institution A: Mobile Money (PaySim)"]
    D2["Institution B: Card / E-Commerce (IEEE-CIS)"]

    P1["PaySim Preprocessing\n- Feature engineering on balance discrepancies\n- Drop non-predictive IDs\n- Label encode transaction types"]
    P2["IEEE-CIS Preprocessing\n- Relational join on TransactionID\n- Encode categorical features\n- Impute sparse numerical fields"]

    S1["Class Imbalance Handling\n(SMOTE Resampling)"]
    S2["Class Imbalance Handling\n(SMOTE-ENN Resampling)"]

    M1["PaySim Baseline Model\n(XGBoost / LightGBM)"]
    M2["IEEE-CIS Baseline Model\n(XGBoost / LightGBM)"]

    XAI["Explainable AI Core (TreeSHAP)\n- Feature attribution computation\n- Translation to plain-English audit rationale"]

    OUT["Monitoring Interface\n- Risk probability score (0 to 100%)\n- Audit justification text\n- Performance metric comparison"]

    D1 --> P1
    D2 --> P2
    P1 --> S1
    P2 --> S2
    S1 --> M1
    S2 --> M2
    M1 --> XAI
    M2 --> XAI
    XAI --> OUT

    style D1 fill:#E1F5FE,stroke:#0288D1,stroke-width:2px,color:#01579B
    style D2 fill:#E8EAF6,stroke:#3949AB,stroke-width:2px,color:#1A237E
    style P1 fill:#E8F5E9,stroke:#43A047,stroke-width:2px,color:#1B5E20
    style P2 fill:#E8F5E9,stroke:#43A047,stroke-width:2px,color:#1B5E20
    style S1 fill:#FFF8E1,stroke:#FFA000,stroke-width:2px,color:#E65100
    style S2 fill:#FFF8E1,stroke:#FFA000,stroke-width:2px,color:#E65100
    style M1 fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px,color:#4A148C
    style M2 fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px,color:#4A148C
    style XAI fill:#E0F7FA,stroke:#00ACC1,stroke-width:2px,color:#006064
    style OUT fill:#FFEBEE,stroke:#E53935,stroke-width:2px,color:#B71C1C
```

---

## 2. Data Processing and Feature Engineering

### Dataset Specifications

| Dataset | Institution Type | Volume | Key Engineered Signals |
|---|:---:|:---:|---|
| <img src="https://img.shields.io/badge/Dataset-PaySim-0288D1?style=for-the-badge" alt="PaySim" /> | Mobile Money (P2P) | ~6.3M rows | `errorBalanceOrig`, `errorBalanceDest`, transaction type encoding |
| <img src="https://img.shields.io/badge/Dataset-IEEE--CIS-3949AB?style=for-the-badge" alt="IEEE-CIS" /> | Card / E-Commerce Gateway | ~590K rows | Relational join on `TransactionID`, device OS, email risk |

### PaySim (Mobile Money)
- **Characteristics:** Agent-based simulation based on real mobile money transaction logs (~6.3 million rows, 11 columns).
- **Engineered Features:** 
  - `errorBalanceOrig`: tracks discrepancy between origin account initial balance, transfer amount, and resulting balance.
  - `errorBalanceDest`: tracks discrepancy on the destination account.
- **Categorical Handling:** Encodes `type` (`TRANSFER`, `CASH_OUT`, `PAYMENT`, `DEBIT`, `CASH_IN`). Drops identifier strings (`nameOrig`, `nameDest`).

### IEEE-CIS (Card / E-Commerce)
- **Characteristics:** Real-world payment gateway records provided by Vesta Corp (~590,000 transactions, 435 combined features).
- **Relational Merging:** Joins `train_transaction.csv` with `train_identity.csv` using `TransactionID`.
- **Categorical Handling:** Encodes payment product codes, card network attributes (`card1` to `card6`), email domains, and device fingerprints.
- **Missing Value Imputation:** Imputes missing entries across numerical and behavioral columns.

---

## 3. Class Imbalance Handling

Financial fraud accounts for less than 1% of transactions (approximately 0.13% in PaySim, 3.5% in IEEE-CIS).

```mermaid
flowchart LR
    A["Raw Imbalanced Data\n(Under 1% Fraud)"]
    
    B["Random Undersampling\n(Discards 99% of valid records)"]
    C["High False Positive Rate\n(Model forgets normal behavior)"]
    
    D["SMOTE / SMOTE-ENN\n(Synthesizes minority fraud samples)"]
    E["High Recall + Low False Alarms\n(Preserves valid transaction patterns)"]

    A -->|"Option 1"| B --> C
    A -->|"FedGuard Approach"| D --> E

    style A fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px,color:#F57F17
    style B fill:#FFCDD2,stroke:#E53935,stroke-width:2px,color:#B71C1C
    style C fill:#FFEBEE,stroke:#D32F2F,stroke-width:2px,color:#B71C1C
    style D fill:#C8E6C9,stroke:#43A047,stroke-width:2px,color:#1B5E20
    style E fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20
```

Rather than discarding legitimate transactions through random undersampling, the pipeline applies:
- **SMOTE:** Synthesizes new fraud instances by interpolating between nearest neighbors of known fraud transactions.
- **SMOTE-ENN:** Follows SMOTE with Edited Nearest Neighbors to prune noisy samples along the decision boundary.

---

## 4. TreeSHAP Decision Explainability

Standard gradient-boosted trees output a raw probability score, which provides no justification to human risk officers. FedGuard integrates TreeSHAP to compute exact Shapley feature attributions:

```mermaid
flowchart TD
    T["Flagged Transaction\n(e.g., TRANSFER of $250,000)"]
    S["TreeSHAP Computation\nEvaluates impact of each feature on the decision"]
    J["Plain-Language Audit Justification\n'Flagged because account was emptied via TRANSFER to a new zero-balance account'"]

    T --> S --> J

    style T fill:#E1F5FE,stroke:#0288D1,stroke-width:2px,color:#01579B
    style S fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px,color:#4A148C
    style J fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20
```

This ensures every flagged decision complies with audit and transparency requirements under RBI guidelines and data protection regulations.

---

## 5. Evaluation Strategy

Because fraud datasets are severely skewed, traditional accuracy is uninformative. The system is evaluated using the following operational hierarchy:

| Metric | Priority | Operational Significance |
|---|:---:|---|
| **PR-AUC (Precision-Recall AUC)** | <img src="https://img.shields.io/badge/Priority-CRITICAL-E53935?style=flat-square" alt="Critical" /> | Immune to true-negative inflation; the definitive measure for extreme class imbalance |
| **Recall (True Positive Rate)** | <img src="https://img.shields.io/badge/Priority-CRITICAL-E53935?style=flat-square" alt="Critical" /> | Quantifies the direct volume of fraudulent capital prevented from escaping |
| **False Positive Rate (FPR)** | <img src="https://img.shields.io/badge/Priority-HIGH-FB8C00?style=flat-square" alt="High" /> | Minimizes legitimate customer friction and reduces operational alert fatigue |
| **Precision & F1-Score** | <img src="https://img.shields.io/badge/Priority-STANDARD-1E88E5?style=flat-square" alt="Standard" /> | Harmonic balance between alert volume and detection accuracy |
| **ROC-AUC** | <img src="https://img.shields.io/badge/Priority-BASELINE-757575?style=flat-square" alt="Baseline" /> | Threshold-independent discrimination baseline for literature comparison |

---

## 6. Multi-Stage Scope Progression

```mermaid
flowchart LR
    S1["Stage 1: Micro (Current)\n- Centralized baselines\n- Class imbalance correction\n- TreeSHAP explainability"]
    S2["Stage 2: Mini (Planned)\n- Federated Learning (Flower)\n- Simulated bank nodes\n- Full-stack dashboard"]
    S3["Stage 3: Minor (Planned)\n- Differential Privacy (Opacus)\n- Privacy budget optimization\n- Production database"]
    S4["Stage 4: Major (Planned)\n- Secure Aggregation (HE / SMPC)\n- Real-time pipeline\n- Regulatory audit validation"]

    S1 --> S2 --> S3 --> S4

    style S1 fill:#E8F5E9,stroke:#00C853,stroke-width:2px,color:#1B5E20
    style S2 fill:#FFFDE7,stroke:#FDD835,stroke-width:2px,color:#F57F17
    style S3 fill:#FFF3E0,stroke:#FB8C00,stroke-width:2px,color:#E65100
    style S4 fill:#FFEBEE,stroke:#E53935,stroke-width:2px,color:#B71C1C
```
