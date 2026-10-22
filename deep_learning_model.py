import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)

def train_and_export_models():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data") if os.path.exists(os.path.join(base_dir, "data")) else base_dir
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "college_placement_data.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, "college_placement_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}.")

    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df)} records.")

    # Feature definitions
    num_cols = [
        'Age', 'SSC_Percentage', 'HSC_Percentage', 'CGPA', 'HistoryOfBacklogs',
        'Internships', 'Projects_Count', 'Certifications_Count', 'Coding_Skills_Score',
        'Aptitude_Test_Score', 'Soft_Skills_Rating', 'Hackathons_Attended', 'Workshops_Attended'
    ]
    cat_cols = ['Gender', 'Stream', 'Hostel']

    # One-hot encode categoricals
    df_encoded = pd.get_dummies(df[cat_cols], drop_first=True, dtype=float)
    encoded_cat_names = list(df_encoded.columns)
    
    # Combined feature matrix X
    X_num = df[num_cols].values
    X_cat = df_encoded.values
    
    # Scale numerical features
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    
    X = np.hstack([X_num_scaled, X_cat])
    feature_names = num_cols + encoded_cat_names
    input_dim = X.shape[1]

    # Target variables
    y_class = df['Placement_Status'].values.astype(np.float32)
    y_reg = df['Salary_LPA'].values.astype(np.float32)

    # Train / Val / Test split (70% train, 15% val, 15% test)
    indices = np.arange(len(df))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=42, stratify=y_class)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=y_class[temp_idx])

    X_train, y_train_cls, y_train_reg = X[train_idx], y_class[train_idx], y_reg[train_idx]
    X_val, y_val_cls, y_val_reg = X[val_idx], y_class[val_idx], y_reg[val_idx]
    X_test, y_test_cls, y_test_reg = X[test_idx], y_class[test_idx], y_reg[test_idx]

    # Convert to PyTorch Tensors
    train_loader_cls = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_cls, dtype=torch.float32)),
        batch_size=64, shuffle=True
    )
    val_loader_cls = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val_cls, dtype=torch.float32)),
        batch_size=64, shuffle=False
    )
    test_tensor_x = torch.tensor(X_test, dtype=torch.float32)
    test_tensor_y = torch.tensor(y_test_cls, dtype=torch.float32)

    # -------------------------------------------------------------
    # 1. DEEP LEARNING ARCHITECTURE: Placement Classification ANN
    # -------------------------------------------------------------
    class PlacementANN(nn.Module):
        def __init__(self, in_features):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.network(x)

    cls_model = PlacementANN(input_dim)
    criterion_cls = nn.BCEWithLogitsLoss()
    optimizer_cls = optim.AdamW(cls_model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler_cls = optim.lr_scheduler.ReduceLROnPlateau(optimizer_cls, mode='min', factor=0.5, patience=5)

    epochs = 45
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    print("\n--- Training Deep Learning Placement Classifier (ANN) ---")
    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, epochs + 1):
        cls_model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for bx, by in train_loader_cls:
            optimizer_cls.zero_grad()
            out = cls_model(bx).squeeze(-1)
            loss = criterion_cls(out, by)
            loss.backward()
            optimizer_cls.step()

            train_loss += loss.item() * len(by)
            preds = (torch.sigmoid(out) >= 0.5).float()
            correct_train += (preds == by).sum().item()
            total_train += len(by)

        train_loss /= total_train
        train_acc = correct_train / total_train

        # Validation
        cls_model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for bx, by in val_loader_cls:
                out = cls_model(bx).squeeze(-1)
                loss = criterion_cls(out, by)
                val_loss += loss.item() * len(by)
                preds = (torch.sigmoid(out) >= 0.5).float()
                correct_val += (preds == by).sum().item()
                total_val += len(by)

        val_loss /= total_val
        val_acc = correct_val / total_val
        scheduler_cls.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = cls_model.state_dict().copy()

        history['epoch'].append(epoch)
        history['train_loss'].append(round(train_loss, 4))
        history['val_loss'].append(round(val_loss, 4))
        history['train_acc'].append(round(train_acc * 100, 2))
        history['val_acc'].append(round(val_acc * 100, 2))

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    # Load best weights
    if best_weights:
        cls_model.load_state_dict(best_weights)

    # Test Evaluation
    cls_model.eval()
    with torch.no_grad():
        test_logits = cls_model(test_tensor_x).squeeze(-1)
        test_probs = torch.sigmoid(test_logits).numpy()
        test_preds = (test_probs >= 0.5).astype(int)

    test_acc = accuracy_score(y_test_cls, test_preds)
    test_prec = precision_score(y_test_cls, test_preds)
    test_rec = recall_score(y_test_cls, test_preds)
    test_f1 = f1_score(y_test_cls, test_preds)
    test_roc_auc = roc_auc_score(y_test_cls, test_probs)
    conf_mat = confusion_matrix(y_test_cls, test_preds).tolist()
    fpr, tpr, _ = roc_curve(y_test_cls, test_probs)

    print("\n--- Test Set Evaluation (Placement Classifier) ---")
    print(f"Accuracy:  {test_acc*100:.2f}%")
    print(f"Precision: {test_prec*100:.2f}%")
    print(f"Recall:    {test_rec*100:.2f}%")
    print(f"F1 Score:  {test_f1*100:.2f}%")
    print(f"ROC AUC:   {test_roc_auc:.4f}")

    # Feature Importance via Permutation Importance on validation set
    cls_model.eval()
    val_tensor_x = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        base_val_acc = accuracy_score(y_val_cls, (torch.sigmoid(cls_model(val_tensor_x).squeeze(-1)).numpy() >= 0.5).astype(int))

    importances = {}
    for fi, fname in enumerate(feature_names):
        X_perm = X_val.copy()
        np.random.shuffle(X_perm[:, fi])
        with torch.no_grad():
            perm_preds = (torch.sigmoid(cls_model(torch.tensor(X_perm, dtype=torch.float32)).squeeze(-1)).numpy() >= 0.5).astype(int)
            perm_acc = accuracy_score(y_val_cls, perm_preds)
        importances[fname] = max(0.0, round(float(base_val_acc - perm_acc), 4))

    # Normalize feature importances to sum to 100%
    total_imp = sum(importances.values()) or 1.0
    normalized_importances = {k: round((v / total_imp) * 100, 2) for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}

    # -------------------------------------------------------------
    # 2. DEEP LEARNING ARCHITECTURE: Salary Regressor ANN (For Placed)
    # -------------------------------------------------------------
    placed_mask_train = (y_train_cls == 1)
    placed_mask_val = (y_val_cls == 1)
    placed_mask_test = (y_test_cls == 1)

    X_train_sal = X_train[placed_mask_train]
    y_train_sal = y_train_reg[placed_mask_train]
    X_val_sal = X_val[placed_mask_val]
    y_val_sal = y_val_reg[placed_mask_val]
    X_test_sal = X_test[placed_mask_test]
    y_test_sal = y_test_reg[placed_mask_test]

    class SalaryANN(nn.Module):
        def __init__(self, in_features):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(in_features, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.network(x)

    sal_model = SalaryANN(input_dim)
    optimizer_sal = optim.Adam(sal_model.parameters(), lr=0.005)
    criterion_sal = nn.MSELoss()

    train_sal_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_sal, dtype=torch.float32), torch.tensor(y_train_sal, dtype=torch.float32)),
        batch_size=32, shuffle=True
    )

    print("\n--- Training Deep Learning Salary Regressor (ANN) ---")
    for epoch in range(1, 35):
        sal_model.train()
        for bx, by in train_sal_loader:
            optimizer_sal.zero_grad()
            out = sal_model(bx).squeeze(-1)
            loss = criterion_sal(out, by)
            loss.backward()
            optimizer_sal.step()

    sal_model.eval()
    with torch.no_grad():
        test_sal_preds = sal_model(torch.tensor(X_test_sal, dtype=torch.float32)).squeeze(-1).numpy()

    sal_mae = float(mean_absolute_error(y_test_sal, test_sal_preds))
    sal_rmse = float(np.sqrt(mean_squared_error(y_test_sal, test_sal_preds)))
    sal_r2 = float(r2_score(y_test_sal, test_sal_preds))
    print(f"Salary Regressor | MAE: {sal_mae:.2f} LPA | RMSE: {sal_rmse:.2f} LPA | R2: {sal_r2:.3f}")

    # Save Models and Artifacts
    torch.save(cls_model.state_dict(), os.path.join(model_dir, "placement_classifier_ann.pth"))
    torch.save(sal_model.state_dict(), os.path.join(model_dir, "salary_regressor_ann.pth"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

    meta_info = {
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'encoded_cat_names': encoded_cat_names,
        'feature_names': feature_names,
        'input_dim': input_dim,
        'history': history,
        'metrics': {
            'accuracy': round(test_acc * 100, 2),
            'precision': round(test_prec * 100, 2),
            'recall': round(test_rec * 100, 2),
            'f1_score': round(test_f1 * 100, 2),
            'roc_auc': round(test_roc_auc, 4),
            'confusion_matrix': conf_mat,
            'fpr': [round(x, 4) for x in fpr[::max(1, len(fpr)//40)]],
            'tpr': [round(x, 4) for x in tpr[::max(1, len(tpr)//40)]],
            'salary_mae': round(sal_mae, 2),
            'salary_rmse': round(sal_rmse, 2),
            'salary_r2': round(sal_r2, 3)
        },
        'feature_importances': normalized_importances,
        'stream_options': ['Computer Science', 'Information Technology', 'Artificial Intelligence & Data Science', 
                           'Mechanical Engineering', 'Electrical Engineering', 'Civil Engineering'],
        'gender_options': ['Male', 'Female'],
        'hostel_options': ['Yes', 'No']
    }

    with open(os.path.join(model_dir, "model_meta.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    print("\nSaved all model checkpoints, scalers, and metadata to 'model/' directory.")

if __name__ == "__main__":
    train_and_export_models()
