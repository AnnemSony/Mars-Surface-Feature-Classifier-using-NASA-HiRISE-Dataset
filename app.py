import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from groq import Groq

# ----- Label Mapping -----
label_mapping = {
    0: "other",
    1: "crater",
    2: "dark dune",
    3: "slope streak",
    4: "bright dune",
    5: "impact ejecta",
    6: "swiss cheese",
    7: "spider"
}

# ----- Image Transform -----
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ----- Load Image -----
def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    return transform(image).unsqueeze(0)

# ----- CNN Model -----
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 16 * 16, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, len(label_mapping))
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

# ----- Load Model -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(label_mapping))
model.load_state_dict(torch.load("cnn_model_weights.pth", map_location=device))
model.to(device)
model.eval()

# ----- Predict -----
def predict(image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return label_mapping[predicted.item()]

# ----- Groq Explanation -----
client = Groq(api_key="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")

def query_groq(label):
    prompt = f"What is a '{label}' on the surface of Mars? Explain in simple terms for a general audience."
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": "You are a planetary science expert."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# ----- Streamlit UI -----
st.title("🔍 Mars Surface Feature Classifier")
st.write("Upload a satellite image and get predictions with explanations!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    image_tensor = load_image(uploaded_file)
    label = predict(image_tensor)
    st.success(f"Predicted Class: {label}")

    with st.spinner("Generating explanation..."):
        explanation = query_groq(label)
    st.info("🧠 AI Explanation:")
    st.write(explanation)
