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
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

# Load the pre-trained model
input_dim = X_train_processed.shape[1]
best_model = CrimePredictionModel(input_dim)
best_model.load_state_dict(torch.load("best_model.pth"))
best_model.eval()  

with torch.no_grad():
    y_test_pred = best_model(X_test_tensor)
    y_train_pred = best_model(X_train_tensor)

y_test_pred = y_test_pred.numpy()  
y_train_pred = y_train_pred.numpy() 

y_test_true = y_test_tensor.numpy() 
y_train_true = y_train_tensor.numpy() 

df_train_result = df_train.copy()
df_test_result = df_test.copy()

df_test_result["cnn_pred"] = y_test_pred
df_train_result["cnn_pred"] = y_train_pred

df_test_result["cnn_res"] = df_test_result["crime_count"] - df_test_result["cnn_pred"]
df_train_result["cnn_res"] = df_train_result["crime_count"] - df_train_result["cnn_pred"]

# Summarize the results for CNN model
def summarize_results(df, model):
    rmse = np.sqrt(mean_squared_error(df["crime_count"], df[model]))
    mae = mean_absolute_error(df["crime_count"], df[model])
    r2 = r2_score(df["crime_count"], df[model])
    mean_value = df["crime_count"].mean()
    mae_percentage = (mae / mean_value) * 100
    return rmse, mae, r2, mae_percentage

# Print out the results for CNN model
rmse, mae, r2, mae_percentage = summarize_results(df_test_result, "cnn_pred")
print(f"Model: CNN")
print(f"RMSE\tR²\tMAE\tMAE Percentage")
print(f"{rmse:.2f}\t{r2:.2f}\t{mae:.2f}\t{mae_percentage:.2f}%")

# Aggregate by month for the CNN model
monthly_train = (
    df_train_result.groupby(["Year", "Month"])
    .agg(
        {
            "crime_count": "sum",
            "cnn_pred": "sum",  # Only using CNN predictions
        }
    )
    .reset_index()
)

monthly_test = (
    df_test_result.groupby(["Year", "Month"])
    .agg(
        {
            "crime_count": "sum",
            "cnn_pred": "sum",  # Only using CNN predictions
        }
    )
    .reset_index()
)

# Create date columns
monthly_train["date"] = pd.to_datetime(
    monthly_train["Year"].astype(str) + "-" + monthly_train["Month"].astype(str) + "-01"
)
monthly_test["date"] = pd.to_datetime(
    monthly_test["Year"].astype(str) + "-" + monthly_test["Month"].astype(str) + "-01"
)

# Sort by date
monthly_train = monthly_train.sort_values("date")
monthly_test = monthly_test.sort_values("date")

# Plot monthly comparison - Training data with CNN predictions
plt.figure(figsize=(15, 7))
plt.plot(
    monthly_train["date"],
    monthly_train["crime_count"],
    marker="o",
    linewidth=2,
    label="Actual",
    color="black",
)
plt.plot(
    monthly_train["date"],
    monthly_train["cnn_pred"],  # CNN predictions
    marker="*",
    linestyle="-",
    label="CNN",
    color="yellow",
)
plt.title("Monthly Crime Count Prediction - Training Data (CNN)", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Total Crime Count", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("train_monthly_comparison_train_data_cnn.pdf", format="pdf")
plt.show()

# Plot monthly comparison - Test data with CNN predictions
plt.figure(figsize=(15, 7))
plt.plot(
    monthly_test["date"],
    monthly_test["crime_count"],
    marker="o",
    linewidth=2,
    label="Actual",
    color="black",
)
plt.plot(
    monthly_test["date"],
    monthly_test["cnn_pred"],  # CNN predictions
    marker="*",
    linestyle="-",
    label="CNN",
    color="yellow",
)
plt.title("Monthly Crime Count Prediction - Test Data (CNN)", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Total Crime Count", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("test_monthly_comparison_test_data_cnn.pdf", format="pdf")
plt.show()

# Summarize the results for CNN model
def summarize_results(df, model):
    rmse = np.sqrt(mean_squared_error(df["crime_count"], df[model]))
    mae = mean_absolute_error(df["crime_count"], df[model])
    r2 = r2_score(df["crime_count"], df[model])
    mean_value = df["crime_count"].mean()
    mae_percentage = (mae / mean_value) * 100
    return rmse, mae, r2, mae_percentage

# Summarize for CNN model only
rmse, mae, r2, mae_percentage = summarize_results(monthly_test, "cnn_pred")
print(f"Model: CNN")
print(f"RMSE\tR²\tMAE\tMAE Percentage")
print(f"{rmse:.2f}\t{r2:.2f}\t{mae:.2f}\t{mae_percentage:.2f}%")

# Create a figure for the CNN model
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the scatter plot for CNN predictions
ax.scatter(
    df_test_result["crime_count"], df_test_result["cnn_pred"], alpha=0.5, color="yellow"
)

# Add diagonal line for perfect predictions (y=x line)
ax.plot(
    [0, df_test_result["crime_count"].max()],
    [0, df_test_result["crime_count"].max()],
    "r--",  # Red dashed line
)

# Set the title to "CNN"
ax.set_title("CNN", fontsize=14)  # Title changed to CNN
ax.set_xlabel("Actual", fontsize=12)
ax.set_ylabel("Predicted", fontsize=12)
ax.grid(True, alpha=0.3)

# Set the same limits for easier comparison
max_val = max(df_test_result["crime_count"].max(), df_test_result["cnn_pred"].max())
ax.set_xlim(0, max_val * 1.05)
ax.set_ylim(0, max_val * 1.05)

# Adjust layout
plt.tight_layout()
plt.savefig("cnn_predictions_vs_actual_values_test_data.pdf", format="pdf")
plt.show()