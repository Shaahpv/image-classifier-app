import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    return model

@st.cache_resource
def load_labels():
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        "imagenet_classes.txt"
    )
    with open("imagenet_classes.txt") as f:
        return [line.strip() for line in f.readlines()]

model = load_model()
labels = load_labels()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# UI
st.title("🔍 AI Image Classifier")
st.write("Upload any image and AI will identify what's in it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
        percentages = torch.nn.functional.softmax(outputs[0], dim=0)
        top3 = torch.topk(percentages, 3)

    st.subheader("Top 3 Predictions:")
    for i in range(3):
        label = labels[top3.indices[i]]
        score = top3.values[i].item()
        st.progress(score)
        st.write(f"**{i+1}. {label}** — {score*100:.1f}%")
