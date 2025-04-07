import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load data (adjust the path if needed)
df = pd.read_csv('time_series_data.csv')

# Split into train and test sets based on year
df_train = df[df["Year"] <= 2023]
df_test = df[df["Year"] > 2023]

# Define feature columns
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

# Add lagged and moving average features
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

# Define preprocessing pipeline
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

# Define custom dataset class for PyTorch
class CrimeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Load data into datasets
train_dataset = CrimeDataset(X_train_tensor, y_train_tensor)
test_dataset = CrimeDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders for batch processing
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN model (same architecture as CrimePredictionModel)
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
        x = self.fc4(x)  # No activation function (regression task)
        return x

# Initialize and load the pre-trained model
input_dim = X_train_processed.shape[1]
best_model = CrimePredictionModel(input_dim)
best_model.load_state_dict(torch.load("best_model.pth"))
best_model.eval()

# Make predictions with the model
with torch.no_grad():
    y_train_pred = best_model(X_train_tensor).numpy().squeeze()
    y_test_pred = best_model(X_test_tensor).numpy().squeeze()

# Create a copy of the original dataframe to store results
df_results = df.copy()

# Add predictions to the dataframe using the original indices
df_results['cnn_pred'] = 0.0  # Initialize with zeros
df_results.loc[df_train.index, 'cnn_pred'] = y_train_pred
df_results.loc[df_test.index, 'cnn_pred'] = y_test_pred

# Save the results to a CSV file
df_results.to_csv('cnn_predictions.csv', index=False)

print("Predictions saved to 'cnn_predictions.csv'")