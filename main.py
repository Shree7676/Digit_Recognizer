import os

import numpy as np
import pandas as pd
import torch

from src.data_loader import get_data_loaders, load_data
from src.model import CNN
from src.plot_utils import plot_learning_curve
from src.train_eval import evaluate_model, load_model, save_model, train, validate

# Paths
train_path = "data/train.csv"
test_path = "data/test.csv"
submission_path = "data/submission.csv"
model_save_path = "models/cnn_model.pth"

# Load Data
X_train, y_train, X_test = load_data(train_path, test_path)
train_loader, val_loader, test_loader = get_data_loaders(X_train, y_train, X_test)

# Initialize Model, Criterion, Optimizer
model = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training and Validation
epochs = 10
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(1, epochs + 1):
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion)
    val_loss, val_accuracy = validate(model, val_loader, criterion)

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch}: Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

# Save Model
save_model(model, model_save_path)

# Evaluate Model
evaluate_model(model, val_loader)

# Plot Learning Curve
plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies, epochs)

# Load Model for Inference or Further Evaluation
model = CNN()
load_model(model, model_save_path)
model.eval()
predictions = []
with torch.no_grad():
    for data in test_loader:
        data = data
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        predictions.extend(pred.cpu().numpy())

predictions = np.array(predictions).flatten()

submission_df = pd.DataFrame(
    {"ImageId": np.arange(1, len(predictions) + 1), "Label": predictions}
)
submission_df.to_csv(submission_path, index=False)
