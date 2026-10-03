"""import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# BASE_DIR is computed relative to this script's own location, so it works
# no matter whose machine or username it's run under. This script lives in
# .../Rice_training/scripts/, so BASE_DIR resolves to .../Rice_training/.
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# 1. Dataset wrapper
# -----------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# 2. MLP model
# -----------------------------
class RiceMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # binary output, no sigmoid here
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -----------------------------
# 3. Train / val split (same GroupShuffleSplit logic you're already using)
# -----------------------------
def get_splits(embeddings, labels, location_ids, test_size=0.2, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(gss.split(embeddings, labels, groups=location_ids))
    return train_idx, val_idx


# -----------------------------
# 4. Helper: run a full pass over a loader and collect predictions
# -----------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_preds, all_labels, all_probs


# -----------------------------
# 5. Training loop
# -----------------------------
def train_mlp(embeddings, labels, location_ids, epochs=100, batch_size=64, lr=1e-3, patience=10):
    train_idx, val_idx = get_splits(embeddings, labels, location_ids)

    X_train, X_val = embeddings[train_idx], embeddings[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    loc_val = location_ids[val_idx]

    # Standardize (fit only on train, apply to val)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_loader = DataLoader(EmbeddingDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EmbeddingDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiceMLP(input_dim=embeddings.shape[1]).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        val_loss, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, device)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| val_acc {val_acc:.4f} | val_f1 {val_f1:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best checkpoint and re-run evaluation on it so the final report/AUC
    # match the actual saved model, not just whatever epoch ran last.
    model.load_state_dict(best_state)
    final_val_loss, final_preds, final_labels, final_probs = evaluate(model, val_loader, criterion, device)

    # -----------------------------
    # Overall metrics (matches the XGBoost/SVM script style)
    # -----------------------------
    acc       = accuracy_score(final_labels, final_preds)
    precision = precision_score(final_labels, final_preds)
    recall    = recall_score(final_labels, final_preds)
    f1        = f1_score(final_labels, final_preds)
    auc       = roc_auc_score(final_labels, final_probs)

    print("\n" + "=" * 60)
    print("Overall metrics (best checkpoint)")
    print("=" * 60)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    print("\nConfusion matrix:")
    cm = confusion_matrix(final_labels, final_preds)
    print(pd.DataFrame(
        cm,
        index=["True Non-rice", "True Rice"],
        columns=["Pred Non-rice", "Pred Rice"]
    ))

    print("\nFinal validation report (best checkpoint):")
    print(classification_report(final_labels, final_preds, target_names=["non-rice", "rice"]))

    fpr, tpr, thresholds = roc_curve(final_labels, final_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"MLP (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — MLP Rice Classifier")
    plt.legend()
    roc_path = os.path.join(MODEL_DIR, "roc_mlp.png")
    plt.savefig(roc_path, dpi=150)
    plt.show()
    print(f"ROC curve saved → {roc_path}")

    # -----------------------------
    # Save model + scaler + predictions (consistent with your other scripts)
    # -----------------------------
    model_path = os.path.join(MODEL_DIR, "mlp_rice_classifier.pt")
    torch.save({
        "model_state_dict": best_state,
        "input_dim": embeddings.shape[1],
        "hidden_dims": [256, 128, 64],
        "dropout": 0.3,
    }, model_path)
    print(f"\nModel saved → {model_path}")

    scaler_path = os.path.join(MODEL_DIR, "mlp_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved → {scaler_path}  (needed at inference time!)")

    results_df = pd.DataFrame({
        "location_id": loc_val,
        "y_true":      final_labels,
        "y_pred":      final_preds,
        "y_proba":     final_probs,
    })
    results_path = os.path.join(MODEL_DIR, "test_predictions_mlp_torch.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved → {results_path}")

    return model, scaler, auc


# -----------------------------
# 6. Usage
# -----------------------------
if __name__ == "__main__":
    embeddings   = np.load(os.path.join(DATA_DIR, "embeddings.npy"))
    labels       = np.load(os.path.join(DATA_DIR, "y.npy"))
    location_ids = np.load(os.path.join(DATA_DIR, "location_ids.npy"), allow_pickle=True)

    model, scaler, auc = train_mlp(embeddings, labels, location_ids)"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# BASE_DIR is computed relative to this script's own location, so it works
# no matter whose machine or username it's run under. This script lives in
# .../Rice_training/scripts/, so BASE_DIR resolves to .../Rice_training/.
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# 1. Dataset wrapper
# -----------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# 2. MLP model
# -----------------------------
class RiceMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # binary output, no sigmoid here
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -----------------------------
# 3. Train / val split (same GroupShuffleSplit logic you're already using)
# -----------------------------
def get_splits(embeddings, labels, location_ids, test_size=0.2, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(gss.split(embeddings, labels, groups=location_ids))
    return train_idx, val_idx


# -----------------------------
# 4. Helper: run a full pass over a loader and collect predictions
# -----------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_preds, all_labels, all_probs


# -----------------------------
# 5. Training loop
# -----------------------------
def train_mlp(embeddings, labels, location_ids, epochs=100, batch_size=64, lr=1e-3, patience=10):
    train_idx, val_idx = get_splits(embeddings, labels, location_ids)

    X_train, X_val = embeddings[train_idx], embeddings[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    loc_val = location_ids[val_idx]

    # Standardize (fit only on train, apply to val)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_loader = DataLoader(EmbeddingDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EmbeddingDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    # Separate loader over the training set with shuffle=False, used only to
    # measure final training accuracy after the best checkpoint is loaded —
    # kept distinct from train_loader (which shuffles + is used for the
    # actual gradient updates) so this measurement doesn't interfere with training.
    train_eval_loader = DataLoader(EmbeddingDataset(X_train, y_train), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiceMLP(input_dim=embeddings.shape[1]).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        val_loss, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, device)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| val_acc {val_acc:.4f} | val_f1 {val_f1:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best checkpoint and re-run evaluation on it so the final report/AUC
    # match the actual saved model, not just whatever epoch ran last.
    model.load_state_dict(best_state)
    final_val_loss, final_preds, final_labels, final_probs = evaluate(model, val_loader, criterion, device)

    # Also evaluate the best checkpoint on the training set itself, so the
    # train/val gap is measured on the exact model being saved — not on
    # whatever the last training epoch happened to look like.
    _, train_preds, train_labels, train_probs = evaluate(model, train_eval_loader, criterion, device)
    train_acc = accuracy_score(train_labels, train_preds)

    # -----------------------------
    # Overall metrics (matches the XGBoost/SVM script style)
    # -----------------------------
    acc       = accuracy_score(final_labels, final_preds)
    precision = precision_score(final_labels, final_preds)
    recall    = recall_score(final_labels, final_preds)
    f1        = f1_score(final_labels, final_preds)
    auc       = roc_auc_score(final_labels, final_probs)

    print("\n" + "=" * 60)
    print("Overall metrics (best checkpoint)")
    print("=" * 60)
    print(f"Training accuracy   : {train_acc:.4f}")
    print(f"Validation accuracy : {acc:.4f}")
    gap = train_acc - acc
    if gap > 0.1:
        print(f"  -> WARNING: train-val gap {gap:.4f} — possible overfitting. "
              f"Consider more dropout, stronger weight_decay, or fewer epochs.")
    else:
        print(f"  -> Train-val gap ({gap:.4f}) looks healthy.")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    print("\nConfusion matrix:")
    cm = confusion_matrix(final_labels, final_preds)
    print(pd.DataFrame(
        cm,
        index=["True Non-rice", "True Rice"],
        columns=["Pred Non-rice", "Pred Rice"]
    ))

    print("\nFinal validation report (best checkpoint):")
    print(classification_report(final_labels, final_preds, target_names=["non-rice", "rice"]))

    fpr, tpr, thresholds = roc_curve(final_labels, final_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"MLP (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — MLP Rice Classifier")
    plt.legend()
    roc_path = os.path.join(MODEL_DIR, "roc_mlp.png")
    plt.savefig(roc_path, dpi=150)
    plt.show()
    print(f"ROC curve saved → {roc_path}")

    # -----------------------------
    # Save model + scaler + predictions (consistent with your other scripts)
    # -----------------------------
    model_path = os.path.join(MODEL_DIR, "mlp_rice_classifier.pt")
    torch.save({
        "model_state_dict": best_state,
        "input_dim": embeddings.shape[1],
        "hidden_dims": [256, 128, 64],
        "dropout": 0.3,
    }, model_path)
    print(f"\nModel saved → {model_path}")

    scaler_path = os.path.join(MODEL_DIR, "mlp_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved → {scaler_path}  (needed at inference time!)")

    results_df = pd.DataFrame({
        "location_id": loc_val,
        "y_true":      final_labels,
        "y_pred":      final_preds,
        "y_proba":     final_probs,
    })
    results_path = os.path.join(MODEL_DIR, "test_predictions_mlp_torch.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved → {results_path}")

    return model, scaler, auc, train_acc, acc


# -----------------------------
# 6. Usage
# -----------------------------
if __name__ == "__main__":
    embeddings   = np.load(os.path.join(DATA_DIR, "embeddings.npy"))
    labels       = np.load(os.path.join(DATA_DIR, "y.npy"))
    location_ids = np.load(os.path.join(DATA_DIR, "location_ids.npy"), allow_pickle=True)

    model, scaler, auc, train_acc, val_acc = train_mlp(embeddings, labels, location_ids)