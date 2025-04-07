import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load data (make sure to adjust the path if needed)
df = pd.read_csv('time_series_data.csv')

# Split into train and test
df_train = df[df["Year"] <= 2023]
df_test = df[df["Year"] > 2023]

# Columns and feature selection
categorical_cols = ["location_name"]
numerical_cols = ["sin_month", "cos_month", "Year", "Month", "num_days", "holiday_num"]
numerical_features = [
    "crime_count",
    "crime_pct_change",
    "morning",
    "afternoon",
    "evening",
    "night",
    "domestic",
    "arrest",
]
for i in [1, 2, 3, 6, 12]:
    for j in numerical_features:
        numerical_cols.append(f"{j}_lag{i}")
for i in [3, 6]:
    for j in numerical_features:
        numerical_cols.append(f"{j}_ma{i}")

# Split features and target
X_train = df_train[categorical_cols + numerical_cols]
y_train = df_train["crime_count"]
X_test = df_test[categorical_cols + numerical_cols]
y_test = df_test["crime_count"]

# Preprocessing pipeline
cat_pipe = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat_pipe", cat_pipe, categorical_cols),
    ]
)

# Apply preprocessing to training and testing data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Custom dataset class for PyTorch
class CrimeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CrimeDataset(X_train_tensor, y_train_tensor)
test_dataset = CrimeDataset(X_test_tensor, y_test_tensor)

# DataLoader for batch processing
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Attention Layer Implementation
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = None  # We will initialize scale inside forward

    def forward(self, x):
        # Get the device of the input tensor x
        device = x.device
        
        # Initialize scale on the same device as input tensor
        if self.scale is None:
            self.scale = torch.sqrt(torch.FloatTensor([x.size(-1)])).to(device)
        
        # Calculate query, key, and value
        Q = self.query(x)  # (batch_size, seq_len, input_dim)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        return attended_values

class CrimePredictionModel(nn.Module):
    def __init__(self, input_dim):
        super(CrimePredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)  # Output layer for crime count prediction
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)  # Adding dropout for regularization

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # No activation function here (regression task)
        return x
    
# Train the model with tqdm progress bar
# Training function with evaluation and saving the best model
def train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, save_path="best_model.pth"):
    best_mae = float('inf')  # Start with a very high RMSE
    best_epoch = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Train for one epoch
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

        # Evaluate after each epoch
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
                outputs = model(inputs)
                y_true.extend(labels.numpy())
                y_pred.extend(outputs.squeeze().numpy())

        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Compute RMSE, R², and MAE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Root Mean Squared Error
        r2 = r2_score(y_true, y_pred)  # R² (Coefficient of Determination)
        mae = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error

        print(f"Epoch [{epoch+1}/{num_epochs}] RMSE: {rmse}, R²: {r2}, MAE: {mae}")

        # Save the best model based on RMSE, MAE, or R² (whichever is best)
        if mae < best_mae:
            best_mae= mae
            best_epoch = epoch + 1
            best_model = model.state_dict()  # Save the model's state_dict (weights)

    # Save the best model after training
    if best_model is not None:
        torch.save(best_model, save_path)
        print(f"Best model saved at epoch {best_epoch} with RMSE: {rmse} with MAE: {best_mae} with R²: {r2}")

# Initialize the model and optimizer
input_dim = X_train_processed.shape[1]  # Number of input features
model = CrimePredictionModel(input_dim)
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # Train and evaluate the model
train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=800, save_path="best_model.pth")