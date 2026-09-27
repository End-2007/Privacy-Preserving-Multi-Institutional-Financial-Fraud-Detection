# FedGuard — Conceptual System Architecture and Technical Design

> **Document Scope:** Detailed architectural specification for the **Micro Project** stage of FedGuard, including multi-institution data ingestion, class-imbalance correction, centralized model formulation, and Explainable AI (SHAP) decision auditing. Future federated stages are summarized at the end.

---

## 1. End-to-End Micro Pipeline Flowchart

The following diagram illustrates the complete architectural workflow of the **Micro Project**:

```mermaid
flowchart TD
    classDef paySimNode fill:#00E5FF,stroke:#00838F,stroke-width:2px,color:#000000,font-weight:bold;
    classDef ieeeNode fill:#2979FF,stroke:#0D47A1,stroke-width:2px,color:#ffffff,font-weight:bold;
    classDef processNode fill:#00E676,stroke:#007E33,stroke-width:2px,color:#000000,font-weight:bold;
    classDef smoteNode fill:#FFD600,stroke:#FF6D00,stroke-width:2px,color:#000000,font-weight:bold;
    classDef modelNode fill:#AA00FF,stroke:#4A148C,stroke-width:2px,color:#ffffff,font-weight:bold;
    classDef shapNode fill:#FF6D00,stroke:#BF360C,stroke-width:2px,color:#ffffff,font-weight:bold;
    classDef uiNode fill:#FF1744,stroke:#B71C1C,stroke-width:2px,color:#ffffff,font-weight:bold;

    subgraph DataSources["Simulated Multi-Institutional Data Sources"]
        D1["Institution A: PaySim\n• Mobile-Money Simulator\n• 6.3M Transactions | 11 Features\n• P2P Transfers & Cash-Outs"]:::paySimNode
        D2["Institution B: IEEE-CIS\n• Real-World Card & E-Commerce\n• ~590K Transactions | 435 Features\n• Relational Transaction + Identity"]:::ieeeNode
    end

    subgraph Pipeline["Data Preprocessing & Hygiene Layer"]
        P1["PaySim Preprocessing\n• Feature Engg: Balance Discrepancy\n• Drop Non-Predictive IDs (nameOrig)\n• Categorical Label Encoding"]:::processNode
        P2["IEEE-CIS Preprocessing\n• Relational Join on TransactionID\n• Categorical Encoding: card, email, device\n• Imputation of Sparse V-features"]:::processNode
    end

    subgraph Rebalance["Extreme Imbalance Correction"]
        S1["SMOTE Resampling\nRaw Fraud: ~0.13%\nInterpolates Synthetic Fraud Vectors"]:::smoteNode
        S2["SMOTE-ENN Resampling\nRaw Fraud: ~3.5%\nRemoves Borderline Ambiguous Noise"]:::smoteNode
    end

    subgraph Baselines["Independent Centralized Classifiers"]
        M1["PaySim Baseline\n• XGBoost / LightGBM\n• Stratified Train/Val/Test (80/20)"]:::modelNode
        M2["IEEE-CIS Baseline\n• XGBoost / LightGBM\n• Stratified Train/Val/Test (80/20)"]:::modelNode
    end

    subgraph Explainability["Explainable AI (XAI) Engine"]
        XAI["TreeSHAP Attribution Core\n• Computes Exact Shapley Additive Values\n• Generates Plain-English Audit Justification\n• Ranks Risk Drivers for Compliance Officers"]:::shapNode
    end

    subgraph AuditInterface["Compliance Audit Dashboard"]
        UI["Interactive Monitoring Interface\n• Dynamic Risk Gauge (0-100%)\n• Natural Language Decision Rationale\n• PR-AUC & FPR Benchmark Visualization"]:::ui
    end

    D1 --> P1 --> S1 --> M1 --> XAI
    D2 --> P2 --> S2 --> M2 --> XAI
    XAI --> UI

    linkStyle default stroke:#455A64,stroke-width:2px;
```

---

## 2. Data Transformation and Preprocessing Mechanics

```mermaid
flowchart LR
    classDef raw fill:#ECEFF1,stroke:#607D8B,stroke-width:2px,color:#263238;
    classDef transform fill:#B2EBF2,stroke:#0097A7,stroke-width:2px,color:#006064;
    classDef ready fill:#C8E6C9,stroke:#388E3C,stroke-width:2px,color:#1B5E20;

    subgraph PaySimIngest["PaySim Feature Engineering"]
        direction TB
        PR1["Raw: step, amount, oldbalanceOrg,\nnewbalanceOrig, oldbalanceDest, newbalanceDest"]:::raw
        PT1["Engineered:\n• errorBalanceOrig = new + amt - old\n• errorBalanceDest = old + amt - new\n• type_encoded = LabelEncoder(type)"]:::transform
        PO1["Model-Ready Tabular Matrix\n(Clean non-zero balance tracking)"]:::ready
        PR1 --> PT1 --> PO1
    end

    subgraph IEEEIngest["IEEE-CIS Relational Join"]
        direction TB
        IR1["Raw Tables:\n• train_transaction.csv (394 cols)\n• train_identity.csv (41 cols)"]:::raw
        IT1["Engineered:\n• Merge on TransactionID\n• Encode: ProductCD, card4, P_email, DeviceType\n• Median Imputation for 300+ V-features"]:::transform
        IO1["Model-Ready Tabular Matrix\n(Unified identity & transaction signal)"]:::ready
        IR1 --> IT1 --> IO1
    end
```

---

## 3. Class Imbalance Strategy: SMOTE vs. Random Undersampling

In banking fraud, non-fraud transactions constitute over **99%** of observations. The architecture applies **SMOTE (Synthetic Minority Over-sampling Technique)** and **SMOTE-ENN**:

```mermaid
flowchart TD
    classDef redBox fill:#FFCDD2,stroke:#D32F2F,stroke-width:2px,color:#B71C1C;
    classDef greenBox fill:#C8E6C9,stroke:#388E3C,stroke-width:2px,color:#1B5E20;
    classDef yellowBox fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px,color:#F57F17;

    Raw["Raw Imbalanced Data\n99.87% Legitimate | 0.13% Fraud"]:::yellowBox

    Raw --> OptionA["Path A: Random Undersampling\nDiscards 99% of Legitimate Records"]:::redBox
    OptionA --> Danger["Severe Information Loss\nDestroys legitimate patterns\nResult: High False Positive Rate (FPR)"]:::redBox

    Raw --> OptionB["Path B: SMOTE / SMOTE-ENN\nSynthesizes Minority Vectors in Feature Space"]:::greenBox
    OptionB --> Benefit["Preserves 100% of Legitimate Data\nRemoves noisy boundary cases via ENN\nResult: High Recall + Controlled FPR"]:::greenBox
```

---

## 4. TreeSHAP Explainability Mechanism

Standard black-box ML models output only a probability score (e.g. `0.94`), which fails regulatory compliance under **RBI Directions** and the **DPDP Act (2023)**. FedGuard's XAI layer translates model decisions into audit-ready explanations:

```mermaid
flowchart TD
    classDef input fill:#E1F5FE,stroke:#0288D1,stroke-width:2px,color:#01579B;
    classDef shap fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#4A148C;
    classDef output fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20;

    TX["Flagged Transaction Input\n• type = TRANSFER\n• amount = $250,000\n• oldbalanceOrg = $250,000\n• newbalanceOrig = $0.0\n• newbalanceDest = $0.0"]:::input

    TX --> Tree["TreeSHAP Exact Computation\nCalculates Shapley value contribution for each feature:\n• type(TRANSFER): +0.45\n• newbalanceDest(0.0): +0.32\n• oldbalanceOrg(emptied): +0.28"]:::shap

    Tree --> Sentence["Natural Language Translation Engine\n'Flagged Suspicious: Entire sender balance ($250,000)\ntransferred out in a single transaction to a destination account\nwith zero prior history and zero resulting balance (Mule signature).'"]:::output
```

---

## 5. Evaluation Framework Architecture

```mermaid
flowchart LR
    classDef crit fill:#FF1744,stroke:#B71C1C,stroke-width:2px,color:#ffffff,font-weight:bold;
    classDef high fill:#FF9100,stroke:#E65100,stroke-width:2px,color:#000000,font-weight:bold;
    classDef std fill:#2979FF,stroke:#0D47A1,stroke-width:2px,color:#ffffff,font-weight:bold;

    subgraph Metrics["Financial Fraud Metric Hierarchy"]
        direction TB
        M1["PR-AUC (Precision-Recall Area)\nPrimary threshold-free metric\nImmune to True-Negative skew"]:::crit
        M2["Recall / True Positive Rate (TPR)\nDirectly measures stopped fraud\nMinimizes financial loss"]:::crit
        M3["False Positive Rate (FPR)\nControls legitimate customer impact\nMinimizes alert fatigue for bank ops"]:::high
        M4["Precision & F1-Score\nHarmonic balance of alerts"]:::std
        M5["ROC-AUC\nBaseline reference"]:::std
    end
```

---

## 6. Full Four-Stage Scope Boundary

FedGuard progresses through four clearly demarcated academic phases:

```mermaid
flowchart TD
    classDef micro fill:#00C853,stroke:#1B5E20,stroke-width:3px,color:#ffffff,font-weight:bold;
    classDef mini fill:#FFD600,stroke:#FF6D00,stroke-width:2px,color:#000000,font-weight:bold;
    classDef minor fill:#FF9100,stroke:#E65100,stroke-width:2px,color:#000000,font-weight:bold;
    classDef major fill:#D50000,stroke:#B71C1C,stroke-width:2px,color:#ffffff,font-weight:bold;

    S1["STAGE 1: MICRO PROJECT (CURRENT ACTIVE)\n• Dual Ingestion: PaySim (Mobile) & IEEE-CIS (Card)\n• SMOTE / SMOTE-ENN Class Rebalancing\n• Independent XGBoost / LightGBM Baselines\n• TreeSHAP Explainability & Natural-Language Auditing\n• Lightweight Monitoring Interface"]:::micro

    S2["STAGE 2: MINI PROJECT (PLANNED NEXT)\n• Federated Learning via Flower (flwr) FedAvg\n• Simulated Cross-Institution Training (No Raw Data Sharing)\n• Full-Stack FastAPI Backend + React UI with Auth\n• Empirical Centralized vs. Federated Benchmark Report"]:::mini

    S3["STAGE 3: MINOR PROJECT (PLANNED)\n• Differential Privacy via Opacus (PyTorch-native)\n• Mathematical Privacy Guarantees (ε, δ budget)\n• Privacy-Utility Trade-off Analysis\n• PostgreSQL / Supabase Analytical Storage"]:::minor

    S4["STAGE 4: MAJOR PROJECT (PLANNED)\n• Cryptographic Secure Aggregation (HE / SMPC)\n• Production Real-Time Fraud Pipeline\n• Full Regulatory Validation (RBI & DPDP Act Compliance)\n• Scopus-Indexed Conference Paper Submission"]:::major

    S1 ==> S2 ==> S3 ==> S4

    linkStyle default stroke:#263238,stroke-width:3px;
```
