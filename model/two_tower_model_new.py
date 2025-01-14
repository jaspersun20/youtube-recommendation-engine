import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# ************************
# MovieLensDataset
# ************************
class MovieLensDataset(Dataset):
    def __init__(self, csv_path):
        """
        Expects columns in CSV:
            userId (int, but we don't use it for embedding here, just for reference),
            avg_user_tag (str with 32 dims, e.g. "0.12,0.34,..."),
            user_like_genre (str with 32 dims),
            imdb (float),
            movie_genre_embedding (str with 32 dims),
            yes/no (str).

        We parse:
          - label = 1 if yes/no == "yes" else 0
          - user_features = concat( avg_user_tag (32-d), user_like_genre (32-d) ) => 64-d float
          - movie_features = concat( imdb(1-d), movie_genre_embedding(32-d) ) => 33-d float
        """
        super().__init__()

        self.data = pd.read_csv(csv_path)

        # yes/no -> numeric label
        self.data["label"] = self.data["yes/no"].apply(lambda x: 1 if x.lower().strip() == "yes" else 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 1) Parse user features (64-d)
        avg_user_tag_str = row["avg_user_tag"]
        avg_user_tag_arr = np.array([float(x) for x in avg_user_tag_str.split(",")], dtype=np.float32)

        user_like_genre_str = row["user_like_genre"]
        user_like_genre_arr = np.array([float(x) for x in user_like_genre_str.split(",")], dtype=np.float32)

        # Concat => (64,)
        user_features = np.concatenate([avg_user_tag_arr, user_like_genre_arr], axis=0)

        # 2) Parse movie features (33-d)
        imdb_val = float(row["imdb"]) if not pd.isna(row["imdb"]) else 0.0
        imdb_tensor = np.array([imdb_val], dtype=np.float32)  # shape (1,)

        genre_str = row["movie_genre_embedding"]
        genre_arr = np.array([float(x) for x in genre_str.split(",")], dtype=np.float32)  # shape (32,)

        movie_features = np.concatenate([imdb_tensor, genre_arr], axis=0)  # shape (33,)

        # 3) label
        label = float(row["label"])  # 0 or 1

        # Convert to Tensors for returning
        return (
            torch.tensor(user_features, dtype=torch.float32),
            torch.tensor(movie_features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


# ************************
# UserTower
# ************************
class UserTower(nn.Module):
    """
    Input: 64-d user features
    => 2-hidden-layer MLP => final output dimension = 64
    """

    def __init__(self, input_dim=64, hidden_dim=128, output_dim=64):
        super(UserTower, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Initialize weights with Xavier
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.relu = nn.ReLU()

    def forward(self, user_x):
        """
        user_x: (batch_size, 64)
        """
        x = self.relu(self.fc1(user_x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))  # final user embedding (batch_size, 64)
        return x


# ************************
# MovieTower
# ************************
class MovieTower(nn.Module):
    """
    Input: 33-d movie features (imdb + genre embedding)
    => 2-hidden-layer MLP => final output dimension = 64
    """

    def __init__(self, input_dim=33, hidden_dim=128, output_dim=64):
        super(MovieTower, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Initialize weights with Xavier
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.relu = nn.ReLU()

    def forward(self, movie_x):
        """
        movie_x: (batch_size, 33)
        """
        x = self.relu(self.fc1(movie_x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))  # final movie embedding (batch_size, 64)
        return x


# ************************
# TwoTowerModel
# ************************
class TwoTowerModel(nn.Module):
    def __init__(self, user_tower, movie_tower):
        super(TwoTowerModel, self).__init__()
        self.user_tower = user_tower
        self.movie_tower = movie_tower

    def forward(self, user_features, movie_features):
        """
        user_features: (batch_size, 64)
        movie_features: (batch_size, 33)
        returns probability in [0,1], shape (batch_size,)
        """
        user_emb = self.user_tower(user_features)   # (batch_size, 64)
        movie_emb = self.movie_tower(movie_features) # (batch_size, 64)
        # Dot product => scalar => sigmoid => probability
        logits = (user_emb * movie_emb).sum(dim=1)
        return torch.sigmoid(logits)


# ************************
# train_two_tower
# ************************
def train_two_tower(
        csv_path="user_movie_training_dataset.csv",
        hidden_dim=128,
        num_epochs=20,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-5,  # L2 reg
        patience=3,        # early stopping
        model_save_path="two_tower_model.pth"
):
    """
    - Reads the CSV dataset -> trains a 2-tower model -> saves state_dict
    - 80/20 train/test split
    - Early stopping with patience
    """

    dataset = MovieLensDataset(csv_path)

    # **Train/test split**
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Build user tower, movie tower, final model
    user_tower = UserTower(input_dim=64, hidden_dim=hidden_dim, output_dim=64)
    movie_tower = MovieTower(input_dim=33, hidden_dim=hidden_dim, output_dim=64)
    model = TwoTowerModel(user_tower, movie_tower)

    criterion = nn.BCELoss()

    # **Weight decay** for L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float('inf')
    no_improve_count = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for user_feats, movie_feats, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(user_feats, movie_feats)  # shape (batch_size,)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # **Evaluate on test set for early stopping**
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for user_feats_t, movie_feats_t, labels_t in test_loader:
                preds_t = model(user_feats_t, movie_feats_t)
                loss_t = criterion(preds_t, labels_t)
                test_loss += loss_t.item()
        avg_test_loss = test_loss / len(test_loader)

        print(f"[INFO] Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss={avg_train_loss:.4f}, Test Loss={avg_test_loss:.4f}")

        # **Early stopping logic**
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            no_improve_count = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"[INFO] Model saved to {model_save_path} (best so far).")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print("[INFO] Early stopping triggered.")
                break

    print(f"[INFO] Best model saved to {model_save_path} with Test Loss={best_loss:.4f}")
    return model


# ************************
# MAIN
# ************************
if __name__ == "__main__":
    final_model = train_two_tower(
        csv_path="user_movie_training_dataset.csv",
        hidden_dim=128,
        num_epochs=20,
        batch_size=64,
        lr=0.001,
        weight_decay=1e-5,
        patience=3,
        model_save_path="two_tower_model.pth"
    )

    print("[INFO] Training complete.")
