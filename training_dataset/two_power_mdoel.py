import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class MovieLensDataset(Dataset):
    def __init__(self, parquet_path):
        """
        Expects columns in the Parquet:
          userId (int),
          avg_user_tag (str, e.g. "0.12,0.34,...(32 dims)"),
          user_like_genre (str, e.g. "0.11,0.09,...(32 dims)") [optional usage],
          imdb (float),
          movie_genre_embedding (str: "0.12,0.54,...(32 dims)"),
          yes/no (str: "yes" or "no").
        We'll parse 'yes/no' => label=1/0.
        We'll parse 32D from 'avg_user_tag' => user tower input.
        We'll parse (1D imdb + 32D from 'movie_genre_embedding') => movie tower input => total 33 dims.
        """
        super().__init__()
        # Read entire parquet into memory
        self.data = pd.read_parquet(parquet_path)

        # Convert "yes/no" -> numeric label
        self.data["label"] = self.data["yes/no"].apply(lambda x: 1 if x.lower() == "yes" else 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 1) user tower: parse 32D from "avg_user_tag"
        user_tag_str = row["avg_user_tag"]
        user_tag_arr = np.array([float(x) for x in user_tag_str.split(",")], dtype=np.float32)
        # We'll pass it as a list so that userTower(*user_features) can unpack it:
        user_features = [user_tag_arr]  # shape (32,)

        # 2) movie tower: imdb(1) + 32D => total 33
        imdb_val = float(row["imdb"]) if not pd.isna(row["imdb"]) else 0.0
        imdb_tensor = np.array([imdb_val], dtype=np.float32)  # shape (1,)

        genre_str = row["movie_genre_embedding"]
        genre_arr = np.array([float(x) for x in genre_str.split(",")], dtype=np.float32)  # shape (32,)

        movie_features = [imdb_tensor, genre_arr]

        # 3) label
        label = float(row["label"])  # 0 or 1

        return user_features, movie_features, label



class UserTower(nn.Module):
    def __init__(self, user_dim=64, hidden_dim=64):
        """
        We'll feed in a 32D user vector -> hidden -> hidden -> user embedding
        add separate dimension for output dimension
        """
        super(UserTower, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # add another hidden layer variable as output dim
            nn.ReLU()
        )

    def forward(self, tag_embedding):
        # shape: (batch_size, 32)
        return self.fc(tag_embedding)  # shape (batch_size, hidden_dim)


class MovieTower(nn.Module):
    def __init__(self, genre_dim=32, hidden_dim=64):
        """
        We'll feed in imdb(1D) + movie_genre_embedding(32D) => 33D total
        => hidden -> hidden -> final movie embedding
        """
        super(MovieTower, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1 + genre_dim, hidden_dim),  # 33 in
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, imdb_rating, genre_embedding):
        """
        imdb_rating: shape (batch_size,1)
        genre_embedding: shape (batch_size,32)
        => cat => (batch_size,33)
        => forward => (batch_size, hidden_dim)
        """
        x = torch.cat([imdb_rating, genre_embedding], dim=1)
        return self.fc(x)


class TwoTowerModel(nn.Module):
    def __init__(self, user_tower, movie_tower):
        super(TwoTowerModel, self).__init__()
        self.user_tower = user_tower
        self.movie_tower = movie_tower

    def forward(self, user_features, movie_features):
        """
        user_features = [ (batch_size,32) ]
        movie_features= [ (batch_size,1), (batch_size,32) ]
        returns probability in [0,1], shape (batch_size,)
        """
        user_emb = self.user_tower(*user_features)  # (batch_size, hidden_dim)
        movie_emb = self.movie_tower(*movie_features)  # (batch_size, hidden_dim)
        # Dot product => scalar => sigmoid => probability
        return torch.sigmoid((user_emb * movie_emb).sum(dim=1))



def train_two_tower(
        parquet_path="user_movie_training_dataset.parquet",
        user_dim=32,
        genre_dim=32,
        hidden_dim=64,
        num_epochs=5,
        batch_size=64,
        lr=0.001,
        model_save_path="two_tower_model.pth"
):
    """
    Reads the Parquet dataset -> trains a 2-tower model -> saves state_dict
    user_dim=32 => we parse "avg_user_tag" as 32D
    genre_dim=32 => "movie_genre_embedding" is 32D
    hidden_dim => tower's hidden
    """
    dataset = MovieLensDataset(parquet_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    user_tower = UserTower(user_dim=user_dim, hidden_dim=hidden_dim)
    movie_tower = MovieTower(genre_dim=genre_dim, hidden_dim=hidden_dim)
    model = TwoTowerModel(user_tower, movie_tower)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # C) Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for user_features, movie_features, labels in dataloader:
            user_features = [f.float() for f in user_features]  # [ (batch,32) ]
            movie_features = [f.float() for f in movie_features]  # [ (batch,1),(batch,32) ]
            labels = labels.float()  # (batch,)

            optimizer.zero_grad()
            outputs = model(user_features, movie_features)  # (batch,)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # [set beak point]
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"[INFO] Epoch {epoch + 1}/{num_epochs}, Loss={avg_loss:.4f}")

    # D) Save model
    torch.save(model.state_dict(), model_save_path)
    print(f"[INFO] Model saved to {model_save_path}")

    return model



if __name__ == "__main__":
    csv_input = "user_movie_training_dataset.csv"


    # 2) Train the two-tower model
    final_model = train_two_tower(
        parquet_path="user_movie_training_dataset.parquet",
        user_dim=32,
        genre_dim=32,
        hidden_dim=64,
        num_epochs=5,
        batch_size=64,
        lr=0.001,
        model_save_path="two_tower_model.pth"
    )

    print("[INFO] Training complete.")
