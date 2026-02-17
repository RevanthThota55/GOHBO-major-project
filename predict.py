"""
Brain Tumor Prediction Script
Use your trained model to classify new MRI images
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# ============= CONFIGURATION =============
MODEL_PATH = 'models/brain_tumor_resnet18.pth'
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ============= LOAD MODEL =============
print("🤖 Loading trained model...")

# Create model architecture
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

print("✅ Model loaded successfully!")

# ============= IMAGE PREPROCESSING =============
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============= PREDICTION FUNCTION =============
def predict_image(image_path):
    """
    Predict brain tumor type from MRI image
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    # Get results
    predicted_label = CLASS_NAMES[predicted_class.item()]
    confidence_score = confidence.item() * 100
    
    return predicted_label, confidence_score, probabilities[0]

# ============= MAIN PREDICTION =============
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BRAIN TUMOR CLASSIFICATION - PREDICTION")
    print("=" * 60)
    
    # Ask user for image path
    print("\nEnter the path to your MRI image:")
    print("Example: data/brain_tumor/archive/Testing/glioma/Te-gl_0010.jpg")
    image_path = input("\nImage path: ").strip()
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found at {image_path}")
        exit()
    
    print(f"\n🔍 Analyzing image: {image_path}")
    print("Please wait...")
    
    # Make prediction
    predicted_label, confidence, all_probabilities = predict_image(image_path)
    
    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\n🎯 Diagnosis: {predicted_label.upper()}")
    print(f"📊 Confidence: {confidence:.2f}%")
    
    print(f"\n📈 All Probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        prob = all_probabilities[i].item() * 100
        bar = "█" * int(prob / 2)
        print(f"  {class_name:12s}: {prob:5.2f}% {bar}")
    
    print("\n" + "=" * 60)
    
    # Ask if user wants to predict another image
    print("\nPredict another image? (yes/no)")
    choice = input("Your choice: ").strip().lower()
    
    if choice in ['yes', 'y']:
        print("\nRestart the script to predict another image!")