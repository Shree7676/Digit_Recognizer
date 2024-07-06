import base64
import io
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, jsonify, render_template, request
from PIL import Image

from src.model import CNN
from src.train_eval import load_model

model_save_path = "models/cnn_model.pth"

app = Flask(__name__)

# Load Model for Inference or Further Evaluation
model = CNN()
load_model(model, model_save_path)
model.eval()

# Define transformations
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ]
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/save_image", methods=["POST"])
def save_image():
    data = request.get_json()
    if "image" in data:
        base64_image = data["image"].split(",")[1]

        current_time = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        image_file_name = f"saved_image_{current_time}.png"
        with open(image_file_name, "wb") as f:
            f.write(base64.b64decode(base64_image))

        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data)).convert("LA")

        image_array = np.array(image)
        grayscale = image_array[:, :, 0]
        alpha = image_array[:, :, 1]

        adjusted_grayscale = np.where(alpha > 0, grayscale, 255)
        normalized_grayscale = adjusted_grayscale / 255.0
        reshaped_array = normalized_grayscale.reshape(1, 1, 28, 28)

        input_tensor = torch.tensor(reshaped_array, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.argmax(dim=1).item()

        return jsonify({"prediction": prediction})
    else:
        return jsonify({"error": "No image data received"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
