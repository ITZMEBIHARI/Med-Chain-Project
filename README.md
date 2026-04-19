# 🏥 Med-Chain: Secure Medical AI & Blockchain Integration

**Med-Chain** is a decentralized healthcare platform designed to diagnose Pneumonia from Chest X-rays using **Deep Learning (MobileNetV2)**. To ensure data integrity and privacy, the system integrates **IPFS** for decentralized storage and **Ethereum Blockchain** for securing medical records.

---

## 🛠️ Step-by-Step Implementation Process

1.  **Data Acquisition:** Utilized the Chest X-ray dataset (Pneumonia vs. Normal) for training and validation.
2.  **AI Model Development:** Implemented **Transfer Learning** using the **MobileNetV2** architecture to achieve high accuracy with minimal computational resources.
3.  **Explainable AI (XAI):** Integrated **Grad-CAM** to generate heatmaps, allowing clinicians to visualize the specific lung regions the AI focused on during diagnosis.
4.  **Security Layer:** Developed a hashing module using **SHA-256** to create a unique digital fingerprint for every medical image.
5.  **Decentralized Storage:** Integrated **IPFS** to store high-resolution X-ray images, retrieving a unique **CID** (Content Identifier) for each upload.
6.  **Blockchain Integration:** Authored a **Solidity Smart Contract** to record the IPFS CID and image hash on a local Ethereum network (Ganache).
7.  **Web Dashboard:** Developed a real-time diagnostic interface using **Streamlit**.

---

## 🐞 Bug Log & Troubleshooting (Documentation)

| S.No | Issue Encountered | Resolution Strategy |
| :--- | :--- | :--- |
| 1 | **Environment Conflicts** | Encountered `ModuleNotFoundError` for TensorFlow. Resolved by creating a dedicated Conda environment (`med_ai`) and managing dependencies via `requirements.txt`. |
| 2 | **IPFS API Incompatibility** | Recent updates in IPFS Kubo caused issues with standard Python libraries. Resolved by using direct **HTTP POST requests** to the IPFS API. |
| 3 | **Model Prediction Bias** | Model initially showed a bias towards Pneumonia. Corrected by implementing **Class Weights** during the training phase to balance the dataset. |
| 4 | **Git Remote Configuration** | A syntax error in the remote URL (extra characters) prevented code pushes. Resolved by manually editing the `.git/config` file and using **GitHub Desktop** for a clean sync. |
| 5 | **GitHub File Size Limits** | The dataset exceeded GitHub's 100MB limit. Resolved by configuring a **.gitignore** file to exclude large binary data while keeping the source code intact. |

---

## 🚀 Installation & Usage

### 1. Environment Setup

```bash
conda activate med_ai
pip install -r requirements.txt
```

### 2. Local Blockchain

Ensure Ganache is running and the smart contract is deployed using Truffle:

```bash
truffle migrate --reset
```

### 3. Launch App

```bash
streamlit run app.py
```
