import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image

CLASS_NAMES = [
    "bent",
    "broken_sheet",
    "crack",
    "graffiti",
    "normal",
    "paint_loss",
    "rust",
    "scratch"
]

# Rebuild model architecture to match the fine-tuned model
def build_model(num_classes):
    model = efficientnet_v2_s(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),

        nn.Dropout(p=0.3),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),

        nn.Linear(128, num_classes)
    )

    return model

# Load model weights
def load_model(weight_path, device):
    model = build_model(len(CLASS_NAMES))
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Preprocessing and normalization
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# Predict class and confidence for a single image
def predict_image(image_path, model, device):
    transform = get_transform()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    predicted_class = CLASS_NAMES[pred.item()]
    confidence = conf.item()

    return predicted_class, confidence

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_PATH = "D:\\Swinburne\\Jan_2026\\COS40007_AI-Engineer\\Project\\code\\models\\finetuned_best.pth"
    IMAGE_PATH = "D:\\Downloads\\download.jpg"

    model = load_model(MODEL_PATH, DEVICE)

    pred_class, conf = predict_image(IMAGE_PATH, model, DEVICE)

    print(f"Prediction: {pred_class}")
    print(f"Confidence: {conf:.4f}")