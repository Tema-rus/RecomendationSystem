import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DATA = {
    "train": "data/u1.base",
    "test": "data/u1.test",
    "users": "data/u.user",
    "movies": "data/u.item",
}

RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
USER_COLUMNS = ["user_id", "age", "gender", "occupation", "zip_code"]
MOVIE_COLUMNS = [
    "movie_id", "movie_title", "release_date", "video_release_date", "imdb_url",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]


@dataclass
class PreparedData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    users_df: pd.DataFrame
    movies_df: pd.DataFrame
    user2idx: dict
    movie2idx: dict
    idx2user: dict
    idx2movie: dict
    user_features: pd.DataFrame


class MovieLensDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.user_idx = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.movie_idx = torch.tensor(df["movie_idx"].values, dtype=torch.long)
        self.age_norm = torch.tensor(df["age_norm"].values, dtype=torch.float32)
        self.gender_num = torch.tensor(df["gender_num"].values, dtype=torch.float32)
        self.rating = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.rating)

    def __getitem__(self, idx: int):
        return (
            self.user_idx[idx],
            self.movie_idx[idx],
            self.age_norm[idx],
            self.gender_num[idx],
            self.rating[idx],
        )


class NCFModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embedding_dim: int = 32,
        use_demographics: bool = False,
    ) -> None:
        super().__init__()
        self.use_demographics = use_demographics

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        input_dim = embedding_dim * 2
        if self.use_demographics:
            input_dim += 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        user_idx: torch.Tensor,
        movie_idx: torch.Tensor,
        age_norm: torch.Tensor | None = None,
        gender_num: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user_vec = self.user_embedding(user_idx)
        movie_vec = self.movie_embedding(movie_idx)

        x = torch.cat([user_vec, movie_vec], dim=1)

        if self.use_demographics:
            if age_norm is None or gender_num is None:
                raise ValueError("Для модели с демографией нужны age_norm и gender_num.")

            extra = torch.stack([age_norm, gender_num], dim=1)
            x = torch.cat([x, extra], dim=1)

        out = self.mlp(x).squeeze(1)
        return out


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(
        DATA["train"],
        sep="\t",
        header=None,
        names=RATING_COLUMNS,
    )

    test_df = pd.read_csv(
        DATA["test"],
        sep="\t",
        header=None,
        names=RATING_COLUMNS,
    )

    users_df = pd.read_csv(
        DATA["users"],
        sep="|",
        header=None,
        names=USER_COLUMNS,
        encoding="latin-1",
    )

    movies_df = pd.read_csv(
        DATA["movies"],
        sep="|",
        header=None,
        names=MOVIE_COLUMNS,
        encoding="latin-1",
    )

    return train_df, test_df, users_df, movies_df


def prepare_data() -> PreparedData:
    train_df, test_df, users_df, movies_df = load_raw_data()

    train_df = train_df.merge(users_df[["user_id", "age", "gender"]], on="user_id", how="left")
    test_df = test_df.merge(users_df[["user_id", "age", "gender"]], on="user_id", how="left")

    gender_map = {"F": 0.0, "M": 1.0}
    train_df["gender_num"] = train_df["gender"].map(gender_map)
    test_df["gender_num"] = test_df["gender"].map(gender_map)
    users_df["gender_num"] = users_df["gender"].map(gender_map)

    age_mean = train_df["age"].mean()
    age_std = train_df["age"].std()
    if pd.isna(age_std) or age_std == 0:
        age_std = 1.0

    train_df["age_norm"] = (train_df["age"] - age_mean) / age_std
    test_df["age_norm"] = (test_df["age"] - age_mean) / age_std
    users_df["age_norm"] = (users_df["age"] - age_mean) / age_std

    user_ids = sorted(users_df["user_id"].unique().tolist())
    movie_ids = sorted(movies_df["movie_id"].unique().tolist())

    user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie2idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    idx2user = {idx: user_id for user_id, idx in user2idx.items()}
    idx2movie = {idx: movie_id for movie_id, idx in movie2idx.items()}

    train_df["user_idx"] = train_df["user_id"].map(user2idx)
    test_df["user_idx"] = test_df["user_id"].map(user2idx)
    train_df["movie_idx"] = train_df["movie_id"].map(movie2idx)
    test_df["movie_idx"] = test_df["movie_id"].map(movie2idx)

    movies_df["movie_idx"] = movies_df["movie_id"].map(movie2idx)
    users_df["user_idx"] = users_df["user_id"].map(user2idx)

    for df in [train_df, test_df]:
        df["gender_num"] = df["gender_num"].fillna(0.0)
        df["age_norm"] = df["age_norm"].fillna(0.0)

    user_features = users_df[["user_idx", "age_norm", "gender_num"]].drop_duplicates("user_idx")
    user_features = user_features.set_index("user_idx")

    return PreparedData(
        train_df=train_df,
        test_df=test_df,
        users_df=users_df,
        movies_df=movies_df,
        user2idx=user2idx,
        movie2idx=movie2idx,
        idx2user=idx2user,
        idx2movie=idx2movie,
        user_features=user_features,
    )


def train_model(
    model: NCFModel,
    train_loader: DataLoader,
    device: torch.device,
    use_demographics: bool,
    epochs: int = 15,
    learning_rate: float = 1e-3,
) -> NCFModel:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for user_idx, movie_idx, age_norm, gender_num, rating in train_loader:
            user_idx = user_idx.to(device)
            movie_idx = movie_idx.to(device)
            age_norm = age_norm.to(device)
            gender_num = gender_num.to(device)
            rating = rating.to(device)

            optimizer.zero_grad()

            if use_demographics:
                pred = model(user_idx, movie_idx, age_norm, gender_num)
            else:
                pred = model(user_idx, movie_idx)

            loss = loss_fn(pred, rating)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(rating)

        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Эпоха {epoch}/{epochs} | loss = {epoch_loss:.4f}")

    return model


def evaluate_regression(
    model: NCFModel,
    data_loader: DataLoader,
    device: torch.device,
    use_demographics: bool,
) -> tuple[float, float]:
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for user_idx, movie_idx, age_norm, gender_num, rating in data_loader:
            user_idx = user_idx.to(device)
            movie_idx = movie_idx.to(device)
            age_norm = age_norm.to(device)
            gender_num = gender_num.to(device)

            if use_demographics:
                pred = model(user_idx, movie_idx, age_norm, gender_num)
            else:
                pred = model(user_idx, movie_idx)

            pred = pred.cpu().numpy()
            pred = np.clip(pred, 1.0, 5.0)

            y_pred.extend(pred.tolist())
            y_true.extend(rating.numpy().tolist())

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return mae, rmse


def precision_at_k(
    model: NCFModel,
    prepared: PreparedData,
    device: torch.device,
    use_demographics: bool,
    k: int = 10,
    relevance_threshold: int = 4,
) -> float:
    model.eval()

    train_history = prepared.train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    test_relevant = (
        prepared.test_df.loc[prepared.test_df["rating"] >= relevance_threshold]
        .groupby("user_idx")["movie_idx"]
        .apply(set)
        .to_dict()
    )

    all_movie_idxs = np.array(sorted(prepared.movies_df["movie_idx"].dropna().unique().tolist()))
    precisions = []

    with torch.no_grad():
        for user_idx, relevant_movies in test_relevant.items():
            watched_train = train_history.get(user_idx, set())
            candidate_movies = [m for m in all_movie_idxs if m not in watched_train]
            if len(candidate_movies) == 0:
                continue

            user_tensor = torch.full(
                (len(candidate_movies),),
                int(user_idx),
                dtype=torch.long,
                device=device,
            )
            movie_tensor = torch.tensor(candidate_movies, dtype=torch.long, device=device)

            if use_demographics:
                age_val = float(prepared.user_features.loc[user_idx, "age_norm"])
                gender_val = float(prepared.user_features.loc[user_idx, "gender_num"])

                age_tensor = torch.full(
                    (len(candidate_movies),),
                    age_val,
                    dtype=torch.float32,
                    device=device,
                )
                gender_tensor = torch.full(
                    (len(candidate_movies),),
                    gender_val,
                    dtype=torch.float32,
                    device=device,
                )

                scores = model(user_tensor, movie_tensor, age_tensor, gender_tensor)
            else:
                scores = model(user_tensor, movie_tensor)

            scores = scores.cpu().numpy()
            top_idx = np.argsort(scores)[-k:][::-1]
            top_k_movies = {candidate_movies[i] for i in top_idx}

            precision = len(top_k_movies.intersection(relevant_movies)) / k
            precisions.append(precision)

    if len(precisions) == 0:
        return 0.0

    return float(np.mean(precisions))


def recommend_top_n(
    model: NCFModel,
    prepared: PreparedData,
    raw_user_id: int,
    device: torch.device,
    use_demographics: bool,
    n: int = 5,
) -> pd.DataFrame:
    if raw_user_id not in prepared.user2idx:
        raise ValueError(f"Пользователь {raw_user_id} не найден.")

    model.eval()

    user_idx = prepared.user2idx[raw_user_id]
    watched_train = set(
        prepared.train_df.loc[prepared.train_df["user_idx"] == user_idx, "movie_idx"].tolist()
    )

    all_movie_idxs = sorted(prepared.movies_df["movie_idx"].dropna().unique().tolist())
    candidate_movies = [m for m in all_movie_idxs if m not in watched_train]

    with torch.no_grad():
        user_tensor = torch.full(
            (len(candidate_movies),),
            int(user_idx),
            dtype=torch.long,
            device=device,
        )
        movie_tensor = torch.tensor(candidate_movies, dtype=torch.long, device=device)

        if use_demographics:
            age_val = float(prepared.user_features.loc[user_idx, "age_norm"])
            gender_val = float(prepared.user_features.loc[user_idx, "gender_num"])

            age_tensor = torch.full(
                (len(candidate_movies),),
                age_val,
                dtype=torch.float32,
                device=device,
            )
            gender_tensor = torch.full(
                (len(candidate_movies),),
                gender_val,
                dtype=torch.float32,
                device=device,
            )

            scores = model(user_tensor, movie_tensor, age_tensor, gender_tensor)
        else:
            scores = model(user_tensor, movie_tensor)

        scores = scores.cpu().numpy()
        scores = np.clip(scores, 1.0, 5.0)

    rec_df = pd.DataFrame({
        "movie_idx": candidate_movies,
        "predicted_rating": scores,
    })

    rec_df = rec_df.sort_values("predicted_rating", ascending=False).head(n)
    rec_df = rec_df.merge(
        prepared.movies_df[["movie_idx", "movie_id", "movie_title"]],
        on="movie_idx",
        how="left",
    )

    return rec_df[["movie_id", "movie_title", "predicted_rating"]]


def main() -> None:
    print("Подготовка данных...")
    prepared = prepare_data()

    print("Train:", prepared.train_df.shape)
    print("Test :", prepared.test_df.shape)
    print("Users:", prepared.users_df.shape)
    print("Movies:", prepared.movies_df.shape)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)
    print()

    train_dataset = MovieLensDataset(prepared.train_df)
    test_dataset = MovieLensDataset(prepared.test_df)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    num_users = len(prepared.user2idx)
    num_movies = len(prepared.movie2idx)

    print("=== БАЗОВАЯ МОДЕЛЬ NCF (user_id + movie_id) ===")
    baseline_model = NCFModel(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=32,
        use_demographics=False,
    )

    baseline_model = train_model(
        model=baseline_model,
        train_loader=train_loader,
        device=device,
        use_demographics=False,
        epochs=15,
        learning_rate=1e-3,
    )

    baseline_mae, baseline_rmse = evaluate_regression(
        model=baseline_model,
        data_loader=test_loader,
        device=device,
        use_demographics=False,
    )

    baseline_precision = precision_at_k(
        model=baseline_model,
        prepared=prepared,
        device=device,
        use_demographics=False,
        k=10,
        relevance_threshold=4,
    )

    print(f"MAE        : {baseline_mae:.4f}")
    print(f"RMSE       : {baseline_rmse:.4f}")
    print(f"Precision@K: {baseline_precision:.4f}")
    print()

    print("=== РАСШИРЕННАЯ МОДЕЛЬ NCF (user_id + movie_id + age + gender) ===")
    extended_model = NCFModel(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=32,
        use_demographics=True,
    )

    extended_model = train_model(
        model=extended_model,
        train_loader=train_loader,
        device=device,
        use_demographics=True,
        epochs=15,
        learning_rate=1e-3,
    )

    extended_mae, extended_rmse = evaluate_regression(
        model=extended_model,
        data_loader=test_loader,
        device=device,
        use_demographics=True,
    )

    extended_precision = precision_at_k(
        model=extended_model,
        prepared=prepared,
        device=device,
        use_demographics=True,
        k=10,
        relevance_threshold=4,
    )

    print(f"MAE        : {extended_mae:.4f}")
    print(f"RMSE       : {extended_rmse:.4f}")
    print(f"Precision@K: {extended_precision:.4f}")
    print()

    results_df = pd.DataFrame([
        {
            "Вариант модели": "Базовая модель NCF",
            "Используемые признаки": "user_id, movie_id",
            "MAE": round(baseline_mae, 4),
            "RMSE": round(baseline_rmse, 4),
            "Precision@K": round(baseline_precision, 4),
        },
        {
            "Вариант модели": "Расширенная модель NCF",
            "Используемые признаки": "user_id, movie_id, age, gender",
            "MAE": round(extended_mae, 4),
            "RMSE": round(extended_rmse, 4),
            "Precision@K": round(extended_precision, 4),
        },
    ])

    print("=== ИТОГОВОЕ СРАВНЕНИЕ ===")
    print(results_df.to_string(index=False))
    print()

    raw_user_id = 1
    print(f"=== ТОП-5 РЕКОМЕНДАЦИЙ ДЛЯ ПОЛЬЗОВАТЕЛЯ {raw_user_id} ===")
    recs_df = recommend_top_n(
        model=extended_model,
        prepared=prepared,
        raw_user_id=raw_user_id,
        device=device,
        use_demographics=True,
        n=5,
    )
    print(recs_df.to_string(index=False))


if __name__ == "__main__":
    main()
