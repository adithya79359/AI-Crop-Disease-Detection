from flask import Flask, render_template, request
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

app = Flask(__name__)

# Disease Classes
classes = [
    "Potato_Early_Blight",
    "Potato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Healthy"
]

# Disease Information
disease_info = {

    "Potato Early Blight": {
        "description": "A fungal disease that causes brown circular spots on potato leaves.",
        "treatment": "Apply fungicide and remove infected leaves immediately.",
        "recovery": "7-14 days",
        "prevention": "Avoid overwatering and maintain proper spacing."
    },

    "Potato Healthy": {
        "description": "Your potato plant appears healthy.",
        "treatment": "No treatment required.",
        "recovery": "Healthy",
        "prevention": "Continue regular watering and monitoring."
    },

    "Tomato Early Blight": {
        "description": "A fungal disease causing dark spots with concentric rings.",
        "treatment": "Apply copper fungicide and remove infected leaves.",
        "recovery": "7-14 days",
        "prevention": "Avoid wet leaves and rotate crops."
    },

    "Tomato Healthy": {
        "description": "Your tomato plant appears healthy.",
        "treatment": "No treatment required.",
        "recovery": "Healthy",
        "prevention": "Continue normal plant care."
    }
}

# Fertilizer Information
fertilizer_info = {

    "Potato Early Blight": {
        "fertilizer": "NPK 19:19:19",
        "dosage": "10g per plant",
        "pesticide": "Copper Oxychloride",
        "application": "Spray once every 7 days"
    },

    "Potato Healthy": {
        "fertilizer": "NPK 20:20:20",
        "dosage": "5g per plant every 15 days",
        "pesticide": "Not Required",
        "application": "Maintain regular care"
    },

    "Tomato Early Blight": {
        "fertilizer": "NPK 19:19:19",
        "dosage": "10g per plant",
        "pesticide": "Mancozeb",
        "application": "Spray every 7 days"
    },

    "Tomato Healthy": {
        "fertilizer": "NPK 20:20:20",
        "dosage": "5g per plant every 15 days",
        "pesticide": "Not Required",
        "application": "Maintain regular care"
    }
}

# Load Model
model = models.resnet18()

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model.load_state_dict(
    torch.load(
        "model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Home Page
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""
    confidence = ""
    info = {}
    fertilizer = {}

    if request.method == "POST":

        file = request.files["image"]

        if file:

            img = Image.open(file).convert("RGB")

            img = transform(img)

            img = img.unsqueeze(0)

            with torch.no_grad():

                output = model(img)

                probabilities = torch.nn.functional.softmax(
                    output,
                    dim=1
                )

                confidence_value, pred = torch.max(
                    probabilities,
                    1
                )

                prediction = classes[pred.item()]

                prediction = prediction.replace("_", " ")

                confidence = round(
                    confidence_value.item() * 100,
                    2
                )

                info = disease_info.get(
                    prediction,
                    {}
                )

                fertilizer = fertilizer_info.get(
                    prediction,
                    {}
                )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        info=info,
        fertilizer=fertilizer
    )

# Run Application
if __name__ == "__main__":
    app.run(debug=True)