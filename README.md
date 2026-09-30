# 🌊 Flood Risk Category Classification System

### Overview
An end-to-end, open-source Machine Learning pipeline and Streamlit application designed to predict real-time flood occurrence risk (`occured`) using pre-disaster environmental indicators. 

### Key Engineering & Data Science Highlights
* **Domain:** Disaster Management | **Task:** Binary Classification (`0 = Low/No Risk`, `1 = High Risk`)[cite: 1]
* **Data Audit & Leakage Prevention:** Audited 6,237 observations and eliminated target leakage (`Disaster Type`) and post-disaster impact metrics (`Total Deaths`, `Total Affected`) prior to model training[cite: 1].
* **Feature Engineering & Scaling:** Standardized key environmental features (`Rainfall`, `Elevation`, `Slope`, `distance`, `duration`, `time`, `Latitude`, `Longitude`) using `StandardScaler`[cite: 1].
* **Best Performing Model:** **Random Forest Classifier** achieved an **81.33% Accuracy**, **0.8753 F1-Score**, **0.9750 Recall**, and **0.8386 ROC-AUC**[cite: 1].
* **Core Insight:** `Rainfall` emerged as the primary risk driver, contributing **46.77%** of overall feature importance[cite: 1].
* **Deployment:** Interactive Streamlit web application deployed on Streamlit Community Cloud[cite: 1, 2].

---

### Benchmark Performance Summary
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |

| **Random Forest** | **81.33%** | **0.7942** | **0.9750** | **0.8753** | **0.8386**[cite: 1] |
| Decision Tree | 78.04% | 0.7794 | 0.9392 | 0.8519 | 0.7566[cite: 1] |
| Logistic Regression | 73.48% | 0.7374 | 0.9404 | 0.8266 | 0.6793[cite: 1] |
