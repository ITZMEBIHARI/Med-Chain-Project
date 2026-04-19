import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests
from web3 import Web3
import hashlib
import json
import matplotlib.pyplot as plt
import cv2

# --- Page Config ---
st.set_page_config(page_title="Med-Chain AI Node", layout="wide")
st.title("🏥 Med-Chain: Federated Medical AI & Blockchain")
st.write("Secure X-Ray Analysis with Decentralized Storage")

# --- Load AI Model ---
@st.cache_resource
def load_my_model():
    # Rebuilding the architecture we trained
    base_model = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    # Load your trained weights
    model.load_weights("local_model_weights.h5")
    return model

model = load_my_model()


#grad cam
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# --- Sidebar: Connection Status ---
st.sidebar.header("Connection Status")
ipfs_status = st.sidebar.empty()
eth_status = st.sidebar.empty()

# Check IPFS
try:
    requests.get("http://127.0.0.1:5001/api/v0/version")
    ipfs_status.success("IPFS: Connected")
except:
    ipfs_status.error("IPFS: Disconnected")

# Check Blockchain (Ganache)
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
if w3.is_connected():
    eth_status.success("Blockchain: Connected")
else:
    eth_status.error("Blockchain: Disconnected")

# --- Main UI ---
uploaded_file = st.file_uploader("Upload Chest X-Ray...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        # ADD THIS .convert('RGB') LINE HERE
        img = Image.open(uploaded_file).convert('RGB') 
        st.image(img, caption="Uploaded X-Ray", use_container_width=True)
        
    with col2:
        st.subheader("AI Diagnostic Results")
        try:
            # 1. Preprocessing
            img_reshaped = img.resize((224, 224))
            img_array = np.array(img_reshaped) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # 2. Predict (Added a status message here)
            with st.spinner("AI is analyzing..."):
                prediction = model.predict(img_array)
                print(f"DEBUG: Raw Prediction: {prediction}") # This shows in your terminal
            
            # --- XAI Heatmap Generation ---
            st.subheader("Explainable AI (XAI) - Grad-CAM")
            with st.spinner("Generating Heatmap..."):
                # MobileNetV2 ki last conv layer ka naam 'Conv_1' hota hai
                heatmap = make_gradcam_heatmap(img_array, model, 'Conv_1')
    
                # Heatmap ko original image par overlay karna
                img_bgr = cv2.cvtColor(np.array(img_reshaped), cv2.COLOR_RGB2BGR)
                heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
                superimposed_img = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
                superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
                st.image(superimposed_img_rgb, caption="Heatmap: Why AI made this decision?", use_container_width=True)
                st.write("🔴 Red areas show where the AI detected abnormalities.")
            
            # 3. Process Results
            classes = ['NORMAL', 'PNEUMONIA']
            class_idx = np.argmax(prediction)
            result = classes[class_idx]
            confidence = np.max(prediction) * 100
            
            # 4. Display to UI
            st.success(f"Analysis Complete!")
            st.info(f"Diagnosis: **{result}**")
            st.progress(int(confidence))
            st.write(f"Confidence: {confidence:.2f}%")
            
        except Exception as e:
            st.error(f"AI Error: {e}")
            print(f"ERROR: {e}")

    # --- Security & Sync Section ---
    st.divider()
    if st.button("Secure & Sync to Blockchain"):
        with st.spinner("Encrypting and Uploading..."):
            # 1. Generate Hash
            file_bytes = uploaded_file.getvalue()
            img_hash = hashlib.sha256(file_bytes).hexdigest()
            
            # 2. Upload to IPFS (Simplified version of your script)
            files = {'file': file_bytes}
            ipfs_res = requests.post("http://127.0.0.1:5001/api/v0/add", files=files).json()
            cid = ipfs_res['Hash']
            
            # 3. Record on Blockchain
            contract_address = "0x7732384A96eBD07974515DD3A9ED3Fd9287697c8" # Update if changed
            with open('blockchain_layer/build/contracts/MedicalRecords.json') as f:
                abi = json.load(f)['abi']
            
            contract = w3.eth.contract(address=contract_address, abi=abi)
            account = w3.eth.accounts[0]
            tx_hash = contract.functions.addRecord(cid, img_hash).transact({'from': account})
            
            st.success("Successfully Synced!")
            st.code(f"IPFS CID: {cid}\nTx Hash: {tx_hash.hex()}")