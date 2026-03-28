import streamlit as st
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import pandas as pd

# 1. PAGE SETUP
st.set_page_config(page_title="Pro Food Analyzer", page_icon="🍔", layout="wide")

# ==========================================
# NEW: PRESENTATION SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=50)
    st.header("🧠 AI Training Specs")
    st.write("Custom Vision Transformer (ViT) fine-tuned on the 5GB Food-101 Dataset.")
    
    st.subheader("Final Evaluation Metrics")
    colA, colB = st.columns(2)
    colA.metric("Accuracy", "88.87%")
    colB.metric("F1 Score", "88.83%")
    colA.metric("Precision", "88.86%")
    colB.metric("Recall", "88.87%")
    
    st.divider()
    st.caption("Hardware: NVIDIA RTX 4050 (6GB VRAM)")
    st.caption("Optimizations: FP16 Mixed Precision, Gradient Accumulation, Multi-core Dataloading")

# ==========================================
# MAIN APP BODY
# ==========================================
st.title("🍔 Pro Food Analyzer & Nutrition Estimator")
st.write("Upload or paste a food image to test the Master Model!")

# 2. LOAD YOUR CUSTOM MODEL (UPDATED PATH)
@st.cache_resource
def load_model():
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    
    # 🎯 NOW POINTING TO YOUR NEW 89% ACCURATE MODEL!
    model_path = "./food_master_model/best_model"
    model = ViTForImageClassification.from_pretrained(model_path, use_safetensors=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

with st.spinner("Loading Master AI brain..."):
    processor, model, device = load_model()

# 3. NUTRITIONAL DATABASE (Core Items)
nutrition_db = {
    "chicken_wings": {"healthy": False, "calories": 430, "protein": 30, "carbs": 0, "fat": 35},
    "pizza": {"healthy": False, "calories": 285, "protein": 12, "carbs": 36, "fat": 10},
    "salad": {"healthy": True, "calories": 150, "protein": 5, "carbs": 10, "fat": 7},
    "hamburger": {"healthy": False, "calories": 500, "protein": 25, "carbs": 40, "fat": 20},
    "grilled_salmon": {"healthy": True, "calories": 250, "protein": 25, "carbs": 0, "fat": 15},
    "apple_pie": {"healthy": False, "calories": 300, "protein": 3, "carbs": 40, "fat": 15},
    "samosa": {"healthy": False, "calories": 260, "protein": 3, "carbs": 24, "fat": 17},
    "ramen": {"healthy": False, "calories": 450, "protein": 15, "carbs": 60, "fat": 15}
}

# 4. THE UPLOAD WIDGET
uploaded_file = st.file_uploader("Upload or paste (Ctrl+V) an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Use columns to keep the image and stats nicely formatted
    image_col, stats_col = st.columns([1, 1])
    
    image = Image.open(uploaded_file).convert("RGB")
    
    with image_col:
        st.image(image, caption="Your Meal", width="stretch") 
    
    # 5. THE PREDICTION BUTTON
    with stats_col:
        if st.button("Analyze with Master Model", type="primary", use_container_width=True):
            with st.spinner("Processing pixels..."):
                
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
                predicted_food = model.config.id2label[predicted_class_idx]
                formatted_name = predicted_food.replace("_", " ").title()
                
                # 6. SHOW THE PREDICTION
                st.success(f"**Prediction:** {formatted_name}")
                
                # THE PROTOTYPE FALLBACK GENERATOR
                if predicted_food not in nutrition_db:
                    name_length = len(predicted_food)
                    is_healthy = any(word in predicted_food for word in ["salad", "grilled", "seaweed", "edamame", "soup", "vegetable"])
                    
                    nutrition_db[predicted_food] = {
                        "healthy": is_healthy,
                        "calories": 250 + (name_length * 15),
                        "protein": 10 + name_length,
                        "carbs": 20 + (name_length * 2),
                        "fat": 5 + name_length
                    }
                
                # 7. DISPLAY ADVANCED NUTRITION & GRAPHS
                data = nutrition_db[predicted_food]
                
                if data["healthy"]:
                    st.info("✅ **Health Assessment:** Generally considered a healthy option.")
                else:
                    st.warning("⚠️ **Health Assessment:** Indulgent / Higher calorie item.")
                
                st.subheader("Estimated Breakdown")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Kcal", f"{data['calories']}")
                m2.metric("Pro", f"{data['protein']}g")
                m3.metric("Carb", f"{data['carbs']}g")
                m4.metric("Fat", f"{data['fat']}g")
                
                chart_data = pd.DataFrame({
                    "Grams": [data["protein"], data["carbs"], data["fat"]]
                }, index=["Protein", "Carbs", "Fats"])
                
                st.bar_chart(chart_data, color="#ff4b4b")
                st.balloons()