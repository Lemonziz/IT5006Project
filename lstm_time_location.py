#!/usr/bin/env python
# coding: utf-8

# In[884]:


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


# Set random seed for reproducibility
# 

# In[885]:


torch.manual_seed(42)
np.random.seed(42)


# Device configuration
# 

# In[886]:


device = "mps"
print(f"Using device: {device}")


# Load and prepare data
# 

# In[887]:


data = pd.read_csv("./data/rnn_full_data.csv")
data


# Create a unique location identifier and time key
# 

# In[888]:


data["District"].unique()


# In[889]:


data["time_key"] = data["Year"] * 12 + data["Month"]
data["location_key"] = data["Location Group"] + "_" + data["District"].astype(str)
data


# Get unique locations and time points
# 

# In[890]:


locations = data["location_key"].unique()
locations.sort()
n_location = len(locations)
time_points = sorted(data["time_key"].unique())
n_time = len(time_points)
location_map = {location: i for i, location in enumerate(locations)}
time_map = {time_point: i for i, time_point in enumerate(time_points)}
data["location_id"] = data["location_key"].map(location_map)
data["time_id"] = data["time_key"].map(time_map)
print(location_map)
print(time_map)


# In[891]:


print(f"Number of unique locations: {len(locations)}")
print(f"Number of time points: {len(time_points)}")
data.describe()


# In[917]:


cat_features = ["location_id"]
num_features = ["crime_count", "holiday_count", "Year", "sin_month", "cos_month"]


# In[918]:


cat_matrix = np.zeros((len(time_points), len(locations), len(cat_features)))
num_matrix = np.zeros((len(time_points), len(locations), len(num_features)))


# In[919]:


for _, row in data.iterrows():
    location_id = round(row["location_id"])
    time_id = round(row["time_id"])
    cat_matrix[time_id, location_id] = row[cat_features]
    num_matrix[time_id, location_id] = row[num_features]
cat_matrix.shape, num_matrix.shape


# In[920]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
matrix_scaled = num_matrix.copy()
matrix_scaled = scaler.fit_transform(num_matrix.reshape(-1, 1)).reshape(
    num_matrix.shape
)


# In[921]:


seq_length = 12


# In[922]:


def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i : i + seq_length]
        target = data[i + seq_length, :, 0]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


# Create sequences
# 

# In[923]:


X_cat, _ = create_sequences(cat_matrix, seq_length)
X_num, y = create_sequences(num_matrix, seq_length)
print(X_cat.shape)
print(X_num.shape)
print(y.shape)


# In[925]:


print(f"Number of sequences: {len(X_cat)}")
print(
    f"Categorical Input shape: {X_cat.shape}"
)  # [n_sequences, seq_length, n_locations, n_features]
print(
    f"Numerical Input shape: {X_num.shape}"
)  # [n_sequences, seq_length, n_locations, n_features]
print(f"Target shape: {y.shape}")  # [n_sequences, n_locations]


# In[ ]:


train_size = int(0.75 * len(X_cat))
X_train, X_val = (X_cat[:train_size], X_num[:train_size]), (
    X_cat[train_size:],
    X_num[train_size:],
)
y_train, y_val = y[:train_size], y[train_size:]


# In[930]:


print(f"Training sequences: {len(X_train[0])}")
print(f"Validation sequences: {len(X_val[0])}")


# Convert to PyTorch tensors
# 

# In[ ]:


X_train = (
    torch.tensor(X_train[0], dtype=torch.float32).to(device),
    torch.tensor(X_train[1], dtype=torch.float32).to(device),
)
X_val = (
    torch.tensor(X_val[0], dtype=torch.float32).to(device),
    torch.tensor(X_val[1], dtype=torch.float32).to(device),
)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)


# Define the RNN model
# 

# In[903]:


input_size = len(features) - 1
hidden_size = 32
batch_size = 16
num_layers = 3
embed_dim = 8
output_size = 1
num_epochs = 50


# In[904]:


class CrimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, embed_dim, num_layers, output_size):
        super(CrimeLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.location_embedding = nn.Embedding(len(locations), embed_dim)
        self.lstm = nn.LSTM(
            input_size=input_size + embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, h_0, c_0):
        # x: [seq_length, batch_size, n_features (crime_count, location_id)]

        # location_embed: [seq_length, batch_size, embed_dim]
        location_embed = self.location_embedding(x[:, :, 1].long())
        # crime: [seq_length, batch_size, 1]
        crime = x[:, :, 0].unsqueeze(-1)
        num = x[:, :, 2:]
        # Combine crime and location embeddings
        x = torch.cat((crime, location_embed, num), dim=-1)
        # x: [seq_length, batch_size, embed_dim + 1]
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        output = self.fc1(output[-1])
        output = self.relu(output)
        output = self.fc2(output)

        return output, (h_n, c_n)


# In[905]:


print(X_train.shape)
model = CrimeLSTM(input_size, hidden_size, embed_dim, num_layers, output_size).to(
    device
)
print(model)


# Define loss function and optimizer
# 

# In[906]:


criterion = nn.MSELoss()


# Training function
# 

# In[907]:


# Validation
def evaluate_model(model, X, y):
    model.eval()
    val_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in range(0, n_location, batch_size):
            actual_batch_size = min(n_location - batch, batch_size)
            h = torch.zeros(num_layers, actual_batch_size, hidden_size).to(device)
            c = torch.zeros(num_layers, actual_batch_size, hidden_size).to(device)
            for i in range(len(X_val)):
                sequence = X[i, :, batch : batch + batch_size]
                target = y[i, batch : batch + batch_size]
                h, c = h.detach(), c.detach()
                score, (h, c) = model(sequence, h, c)
                loss = criterion(score, target.unsqueeze(1))
                val_loss += loss.item()
                num_batches += 1
    total_val_loss = val_loss / num_batches
    return total_val_loss


# In[908]:


start_time = time.time()
n_location = len(locations)
train_losses = []
val_losses = []
my_lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=my_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
n_seq = len(X_train)
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in range(0, n_location, batch_size):
        actual_batch_size = min(n_location - batch, batch_size)
        h = torch.zeros(num_layers, actual_batch_size, hidden_size).to(device)
        c = torch.zeros(num_layers, actual_batch_size, hidden_size).to(device)
        for i in range(n_seq):
            optimizer.zero_grad()
            sequence = X_train_tensor[i, :, batch : batch + actual_batch_size]
            target = y_train_tensor[i, batch : batch + actual_batch_size]
            h, c = h.detach(), c.detach()
            score, (h, c) = model(sequence, h, c)
            loss = criterion(score, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
    scheduler.step()
    train_loss /= num_batches
    train_losses.append(train_loss)
    val_loss = evaluate_model(model, X_val_tensor, y_val_tensor)
    val_losses.append(val_loss)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.12f}, Val Loss: {val_loss:.12f}, learning rate: {scheduler.get_last_lr()[0]:.12f}, time: {time.time() - start_time:.2f} seconds"
    )


# Train the model
# 

# Plot training and validation loss
# 

# In[ ]:


plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("training_validation_loss.pdf")
plt.show()


# In[ ]:


def predictions(model, X, y):
    num_seq, _, n_location, n_features = X.shape
    result = np.zeros((num_seq, n_location, 2))
    with torch.no_grad():
        for i in range(len(X)):
            for batch in range(0, n_location, batch_size):
                batch_increment = min(batch_size, n_location - batch)
                sequence = X[i, :, batch : batch + batch_increment]
                target = y[i, batch : batch + batch_increment]
                score = model(sequence)
                result[i, batch : batch + batch_increment, 0] = (
                    score.cpu().numpy().flatten()
                )
                result[i, batch : batch + batch_increment, 1] = (
                    target.cpu().numpy().flatten()
                )
    return result


# In[ ]:


prediction_model = model.to("mps")
result = predictions(model, X_val_tensor.to("mps"), y_val_tensor.to("mps"))
unscaled_result = scaler.inverse_transform(result.reshape(-1, 2)).reshape(result.shape)


# In[ ]:


unscaled_result


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def plot_district_predictions(data, save_path=None):
    """
    Plot predicted vs actual values for each district over time and calculate error metrics.

    Parameters:
    -----------
    data : numpy.ndarray
        3D array of shape (months, districts, 2) where the last dimension
        contains [predicted, actual] values.
    save_path : str, optional
        If provided, the figure will be saved to this path.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the plots.
    dict
        Dictionary containing error metrics for each district.
    """
    num_months, num_districts, _ = data.shape

    # Determine grid size for subplots (5x5 grid to fit 23 districts)
    cols = 5
    rows = 5

    # Create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 18))
    axes = axes.flatten()  # Flatten to make indexing easier

    # Create x-axis data (months)
    months = np.arange(1, num_months + 1)

    # Dictionary to store metrics for each district
    metrics = {}

    # Plot each district
    for i in range(num_districts):
        ax = axes[i]

        # Extract predicted and actual values for this district
        predicted = data[:, i, 0]
        actual = data[:, i, 1]

        # Calculate metrics for non-NaN values
        valid_indices = ~np.isnan(predicted) & ~np.isnan(actual)
        valid_predicted = predicted[valid_indices]
        valid_actual = actual[valid_indices]

        if len(valid_actual) > 0:
            # Calculate Mean Absolute Error
            mae = mean_absolute_error(valid_actual, valid_predicted)

            # Calculate MAE percentage (MAPE)
            # Avoid division by zero by adding a small epsilon
            epsilon = 1e-10
            mape = (
                np.mean(
                    np.abs((valid_actual - valid_predicted) / (valid_actual + epsilon))
                )
                * 100
            )

            # Calculate Root Mean Squared Error
            rmse = np.sqrt(mean_squared_error(valid_actual, valid_predicted))

            # Calculate R-squared (coefficient of determination)
            r2 = r2_score(valid_actual, valid_predicted)

            # Store metrics
            metrics[f"District_{i+1}"] = {
                "MAE": mae,
                "MAPE(%)": mape,
                "RMSE": rmse,
                "R²": r2,
            }

            # Add metrics to the plot
            metrics_text = (
                f"MAE: {mae:.2f}\nMAPE: {mape:.2f}%\nRMSE: {rmse:.2f}\nR²: {r2:.2f}"
            )
            ax.text(
                0.05,
                0.95,
                metrics_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Plot the data
        ax.plot(months, predicted, "b-o", label="Predicted", markersize=4)
        ax.plot(months, actual, "r-o", label="Actual", markersize=4)

        # Add title and labels
        ax.set_title(f"District {i+1}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Value")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.7)

        # Add legend to each subplot
        ax.legend(loc="best")

        # Set y limits with a bit of padding
        all_values = np.concatenate([predicted, actual])

        # Filter out NaN values for computing limits
        valid_values = all_values[~np.isnan(all_values)]
        if len(valid_values) > 0:
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            padding = 0.1 * (max_val - min_val)
            ax.set_ylim(min_val - padding, max_val + padding)

    # Hide any unused subplots
    for i in range(num_districts, len(axes)):
        axes[i].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Add a title for the whole figure
    fig.suptitle("Predicted vs Actual Values by District", fontsize=20, y=0.995)

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Create a summary of metrics across all districts
    district_metrics_df = None
    try:
        import pandas as pd

        district_metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
        print("Overall Metrics Summary:")
        print(district_metrics_df)
        print("\nAverage Metrics Across All Districts:")
        print(district_metrics_df.mean())
    except ImportError:
        print("Pandas not available for summary table, printing raw metrics instead:")
        for district, metric in metrics.items():
            print(f"{district}: {metric}")

    return fig, metrics


# Example usage with your data
# Assuming your data is already loaded into a variable called 'unscaled_result'
fig, metrics = plot_district_predictions(unscaled_result)
plt.show()


# In[ ]:


rounded_result = np.round(unscaled_result)
month_result = np.sum(rounded_result, axis=1)
month_result


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming month_result is already calculated as shown in your code
# month_result = np.sum(rounded_result, axis=1)

# Create a list of month labels
months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

# Plot the month_result
plt.figure(figsize=(10, 6))
plt.plot(
    months,
    month_result[:, 0],
    marker="s",
    linestyle="-",
    color="green",
    label="Predicted",
)
plt.plot(
    months, month_result[:, 1], marker="^", linestyle="-", color="red", label="Actual"
)

# Add labels and title
plt.xlabel("Month")
plt.ylabel("Value")
plt.title("Monthly Predicted vs Actual Values")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:


locations = data["District"].unique()
locations.sort()
n_location = len(locations)
time_points = sorted(data["time_key"].unique())
n_time = len(time_points)
location_map = {location: i for i, location in enumerate(locations)}
time_map = {time_point: i for i, time_point in enumerate(time_points)}


# In[ ]:




