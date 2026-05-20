import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request

st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🔍",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    return model

@st.cache_resource
def load_labels():
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

# Header
st.title("🔍 AI Image Classifier")
st.markdown("Upload any image and AI will identify what's in it using **MobileNetV2** trained on 1000 categories.")
st.divider()

# Upload
uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded image", use_column_width=True)

    with col2:
        with st.spinner("🤖 Analyzing image..."):
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
            percentages = torch.nn.functional.softmax(outputs[0], dim=0)
            top3 = torch.topk(percentages, 3)

        st.subheader("Top Predictions")
        for i in range(3):
            label = labels[top3.indices[i]]
            score = top3.values[i].item()
            medal = ["🥇", "🥈", "🥉"][i]
            st.metric(
                label=f"{medal} {label.replace('_', ' ').title()}",
                value=f"{score*100:.1f}%"
            )
            st.progress(score)

    st.divider()
    st.caption("Built with PyTorch + Streamlit · MobileNetV2 · ImageNet 1000 classes")

else:
    # Show example info when no image uploaded
    st.info("👆 Upload a photo of any object, animal, food, or vehicle to get started!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("🐶 Animals")
    with col2:
        st.success("🚗 Vehicles")
    with col3:
        st.success("🍎 Food & Objects")
