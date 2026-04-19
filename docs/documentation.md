# 📄 Full Technical Documentation: Med-Chain (AI + Blockchain)

---

## 1. Project Architecture (Kaam Kaise Karta Hai?)

Is project ka flow 5 steps mein divided hai:

1. **Input:** Doctor ya patient Chest X-ray upload karta hai.
2. **AI Layer:** Model image ko resize karta hai aur Pneumonia predict karta hai.
3. **XAI Layer:** Grad-CAM us image par heatmap banata hai (infection area highlight karta hai).
4. **Storage Layer:** Image IPFS par upload hoti hai (jisse ek unique CID milti hai).
5. **Blockchain Layer:** Image ka Hash, CID, aur AI result Ganache (Ethereum) par hamesha ke liye secure ho jata hai.

---

## 2. Phase 1: AI Model & Preprocessing (Deep Learning)

### A. Image Resize & Preprocessing

MobileNetV2 sirf 224x224 pixels ki images ko samajh sakta hai. Isliye har X-ray ko upload hote hi resize karna zaroori tha.

**Code snippet (Image Processing):**

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img_path):
    # Image ko 224x224 mein load aur resize karna
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Image ko array mein convert karna
    img_array = image.img_to_array(img)
    
    # Model ke hisaab se dimensions badhana (Batch size add karna)
    img_array = np.expand_dims(img_array, axis=0)
    
    # MobileNetV2 ke format mein normalize karna (Pixel values 0-1 ke beech)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    return img_array
```

### B. MobileNetV2 Architecture

Humne scratch se model nahi banaya kyunki usme mahino lagte. Humne Transfer Learning use ki aur pre-trained weights (imagenet) par apna output layer lagaya.

**Code snippet (Model Building):**

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Base model load karna (upar ki layers hatakar)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Nayi custom layers add karna
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)

# Final output layer (2 classes: Normal ya Pneumonia)
predictions = Dense(2, activation='softmax')(x)

# Final model tayyar karna
model = Model(inputs=base_model.input, outputs=predictions)
```

---

## 3. Phase 2: Explainable AI (Grad-CAM)

AI ne agar Pneumonia bola, toh doctor us par direct trust kyu kare? Isliye humne Grad-CAM (Gradient-weighted Class Activation Mapping) implement kiya. Ye image par red/yellow color ka dhabba (heatmap) banata hai wahan jahan problem hai.

**Code snippet (Heatmap Logic):**

```python
import cv2

def generate_gradcam_heatmap(model, img_array, last_conv_layer_name="Conv_1"):
    # Model ko do hisso mein batna: inputs aur last layer
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Gradients calculate karna
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        top_pred_index = tf.argmax(preds[0])
        class_channel = preds[:, top_pred_index]

    # Gradients aur pooling apply karna
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Heatmap generate karna
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()
```

---

## 4. Phase 3: Decentralized Storage (IPFS)

Medical X-rays file size mein badi hoti hain. Unhe direct Blockchain par daalna bahut expensive aur slow hota. Isliye image IPFS par daali gayi, aur wahan se humein ek CID (Content ID) mila.

> **Note:** Kyunki library error de rahi thi, humne direct HTTP Request use ki.

**Code snippet (IPFS Upload):**

```python
import requests

def upload_to_ipfs(file_path):
    # IPFS Kubo node ka local address
    ipfs_api_url = 'http://127.0.0.1:5001/api/v0/add'
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(ipfs_api_url, files=files)
        
    if response.status_code == 200:
        # CID return karna
        return response.json()['Hash']
    else:
        return "Error uploading to IPFS"
```

---

## 5. Phase 4: Blockchain Implementation (Ethereum/Ganache)

Data ko secure, tamper-proof, aur private rakhne ke liye humne Solidity mein Smart Contract likha aur use Python se joda.

### A. The Smart Contract (Solidity)

Ye code batata hai ki blockchain par kya kya save hoga.

**Code snippet (MedChain.sol):**

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MedChain {
    // Record ka structure
    struct MedicalRecord {
        string patientId;
        string imageHash;   // SHA-256 hash security ke liye
        string ipfsCID;     // IPFS ka address
        string aiDiagnosis; // Pneumonia ya Normal
        uint256 timestamp;
    }

    // Records ko store karne ke liye mapping
    mapping(string => MedicalRecord) public records;

    // Naya record add karne ka function
    function addRecord(string memory _id, string memory _hash, string memory _cid, string memory _diagnosis) public {
        records[_id] = MedicalRecord(_id, _hash, _cid, _diagnosis, block.timestamp);
    }
}
```

### B. Python to Ganache Connection (Web3.py)

Streamlit app se Blockchain par data bhejne ka code.

**Code snippet (blockchain_layer.py):**

```python
from web3 import Web3

# Ganache se connect karna
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))

# Contract Address aur ABI (Truffle se milta hai)
contract_address = '0xYourContractAddressHere...'
contract_abi = [...] # Yahan contract ka JSON ABI lagta hai

contract = w3.eth.contract(address=contract_address, abi=contract_abi)
account = w3.eth.accounts[0] # Ganache ka pehla account

def store_on_blockchain(patient_id, image_hash, ipfs_cid, diagnosis):
    # Transaction build aur send karna
    tx_hash = contract.functions.addRecord(
        patient_id, image_hash, ipfs_cid, diagnosis
    ).transact({'from': account})
    
    # Transaction complete hone ka wait karna
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt
```

---

## 6. Phase 5: The Web Interface (Streamlit)

Ye sab kuch ek aasan screen par laane ke liye Streamlit ka use hua jise koi bhi doctor chala sakta hai.

**Code snippet (app.py):**

```python
import streamlit as st
import hashlib

st.title("🏥 Med-Chain: AI & Blockchain Diagnostics")

# File uploader UI
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Image dikhana
    st.image(uploaded_file, caption="Uploaded X-ray", width=300)
    
    # 2. SHA-256 Hash Generate karna
    file_bytes = uploaded_file.getvalue()
    img_hash = hashlib.sha256(file_bytes).hexdigest()
    st.write(f"🔒 **Secure Image Hash:** {img_hash}")
    
    if st.button("Diagnose & Secure"):
        # Yahan se AI prediction, IPFS upload aur Blockchain store function call hote hain
        st.success("Diagnosis Complete: PNEUMONIA DETECTED")
        st.info("Record successfully permanently secured on Ethereum Blockchain!")
```
