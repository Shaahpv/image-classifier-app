# 🔍 AI Image Classifier

A computer vision web app that identifies objects in images using 
deep learning with 99%+ accuracy on clear photos.

## 🚀 Live Demo
> Upload any image → AI identifies it instantly

## 🛠️ Tech Stack
- **PyTorch** — MobileNetV2 pretrained on ImageNet (1000 classes)
- **Streamlit** — Interactive web interface
- **Transfer Learning** — No training from scratch

## ✨ Features
- Upload JPG/PNG images
- Returns Top 3 predictions with confidence scores
- Identifies 1000 different object categories

## ▶️ Run Locally
git clone https://github.com/Shaahpv/image-classifier-app
cd image-classifier-app
pip install -r requirements.txt
streamlit run app.py

## 📌 Example Output
| Image | Top Prediction | Confidence |
|-------|---------------|------------|
| Dog photo | Blenheim Spaniel | 99.0% |
