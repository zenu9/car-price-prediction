import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import xgboost as xgb


# Folder Setup
def setup_folders():
    # Base folder
    base_plot_dir = "plots"
    subfolders = ["correlation", "feature_selection", "errors", "actual_vs_pred", "feature_importance"]
    
    # Create subfolders if they don't exist
    for folder in subfolders:
        os.makedirs(os.path.join(base_plot_dir, folder), exist_ok=True)
    return base_plot_dir


# Load and Preprocess Data
def load_and_preprocess_data():
    try:
        # Load dataset
        data = pd.read_csv("data/car_data.csv")
    except FileNotFoundError:
        print("File 'car_data.csv' not found! Please check the file location.")
        return None, None, None

    data = data.drop("Car_Name", axis=1)

    # Encode categorical variables
    categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
    le = LabelEncoder()
    for col in categorical_features:
        data[col] = le.fit_transform(data[col])

    # Features and target
    X = data.drop('Selling_Price', axis=1)
    y = data['Selling_Price']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    plot_correlation_matrix(data, "plots")
    feature_selection_selectkbest(X, y, "plots")
    feature_selection_rfe(X_scaled, y, X.columns)

    return data, X, y, scaler, X_scaled


# Correlation Matrix
def plot_correlation_matrix(data, base_plot_dir):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(base_plot_dir, "correlation", "correlation_matrix.png"))
    plt.close()


# Feature Selection (SelectKBest)
def feature_selection_selectkbest(X, y, base_plot_dir):
    selector = SelectKBest(score_func=f_regression, k='all')
    selector.fit(X, y)
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    }).sort_values(by='Score', ascending=False)
    print("\n=== Feature Importance via SelectKBest ===")
    print(feature_scores)

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Score', y='Feature', data=feature_scores)
    plt.title("Feature Importance via SelectKBest")
    plt.tight_layout()
    plt.savefig(os.path.join(base_plot_dir, "feature_selection", "feature_selection_selectkbest.png"))
    plt.close()


# Recursive Feature Elimination (RFE)
def feature_selection_rfe(X_scaled, y, X_columns):
    rfe = RFE(LinearRegression(), n_features_to_select=5)
    rfe.fit(X_scaled, y)
    rfe_features = pd.DataFrame({
        'Feature': X_columns,
        'Selected': rfe.support_
    })
    print("\n=== Features Selected by RFE ===")
    print(rfe_features)
    return rfe_features


# MLPRegressor
class MLPRegressorTorch(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


# Model Training Function
def train_models(X_train, y_train, X_test, y_test, X):
    results = {}

    # Dummy
    dummy = DummyRegressor(strategy='mean').fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    results["Dummy"] = evaluate_model(y_test, y_pred_dummy)

    # Linear Regression
    lr = LinearRegression()
    grid_lr = GridSearchCV(
        lr, 
        {
            'fit_intercept': [True, False]
        }, 
        scoring='r2', 
        cv=3)
    grid_lr.fit(X_train, y_train)
    best_lr = grid_lr.best_estimator_
    y_pred_lr = best_lr.predict(X_test)
    results["LinearRegression"] = evaluate_model(y_test, y_pred_lr)

    # Random Forest
    rf = RandomForestRegressor(random_state=42)
    param_rf = {
        'n_estimators': [100, 200], 
        'max_depth': [None, 5, 10]
    }
    grid_rf = GridSearchCV(rf, param_rf, scoring='r2', cv=3)
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    results["RandomForest"] = evaluate_model(y_test, y_pred_rf)

    # XGBoost
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_xgb = GridSearchCV(xgb_model, 
                            {
                                'n_estimators': [200, 400], 
                                'max_depth': [3, 5]
                            }, 
                            scoring='r2', 
                            cv=3)
    grid_xgb.fit(X_train, y_train)
    best_xgb = grid_xgb.best_estimator_
    y_pred_xgb = best_xgb.predict(X_test)
    results["XGBoost"] = evaluate_model(y_test, y_pred_xgb)

    # XGBoost Feature Importance
    plot_feature_importance_xgb(best_xgb, X.columns, "plots")

    # SVR
    svr = SVR()
    param_svr = {
        'kernel': ['rbf', 'linear'],
        'C': [1, 10, 50, 100],
        'gamma': ['scale', 'auto']
    }
    
    grid_svr = GridSearchCV(svr, param_svr, scoring='r2', cv=3, n_jobs=-1)
    grid_svr.fit(X_train, y_train)
    best_svr = grid_svr.best_estimator_
    y_pred_svr = best_svr.predict(X_test)
    results["SVR"] = evaluate_model(y_test, y_pred_svr)

    # PyTorch MLP
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_mlp = MLPRegressorTorch(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_mlp.parameters(), lr=0.001)

    # Convert numpy to torch tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test_torch  = torch.tensor(X_test, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_torch, y_train_torch),
        batch_size=32,
        shuffle=True
    )

    # Train MLP
    epochs = 200
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model_mlp(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Predict
    model_mlp.eval()
    with torch.no_grad():
        y_pred_mlp = model_mlp(X_test_torch).cpu().numpy().flatten()

    results["MLP_Torch"] = evaluate_model(y_test, y_pred_mlp)

    return results, best_lr, best_rf, best_xgb, best_svr, model_mlp, y_pred_lr, y_pred_rf, y_pred_xgb, y_pred_svr, y_pred_mlp


# Model Evaluation Helper
def evaluate_model(y_true, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }


# Feature Importance (XGBoost)
def plot_feature_importance_xgb(model, feature_names, base_plot_dir):
    feature_importances = pd.Series(model.feature_importances_, index=feature_names)
    plt.figure(figsize=(10, 6))
    feature_importances.sort_values(ascending=False).plot(kind='bar')
    plt.title("Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(base_plot_dir, "feature_importance", "feature_importance_xgb.png"))
    plt.close()


# Visualization Functions
def error_analysis(y_true, y_pred, model_name, base_plot_dir):
    errors = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Error': abs(y_true - y_pred)
    })
    plt.figure(figsize=(8, 6))
    sns.histplot(errors['Error'], bins=20, kde=True)
    plt.title(f"Error Distribution ({model_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(base_plot_dir, "errors", f"error_{model_name.lower()}.png"))
    plt.close()


def plot_actual_vs_pred(y_true, y_pred, model_name, base_plot_dir):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs Predicted ({model_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(base_plot_dir, "actual_vs_pred", f"actual_vs_pred_{model_name.lower()}.png"))
    plt.close()


# Console Interface
def predict_price_console(best_lr, best_rf, best_xgb, best_svr, best_mlp, scaler, X):
    print("\n=== Car Price Prediction Interface ===")

    while True:
        try:
            year = int(input("Enter year of manufacture: "))
            present_price = float(input("Enter current price (in lakhs): "))
            kms = int(input("Enter kilometers driven: "))
            fuel = int(input("Fuel type (0=Diesel, 1=Petrol, 2=CNG): "))
            seller = int(input("Seller type (0=Dealer, 1=Individual): "))
            transmission = int(input("Transmission (0=Manual, 1=Automatic): "))
            owner = int(input("Number of previous owners: "))

            model_choice = int(input("Choose model: 1=RF, 2=LR, 3=XGB, 4=SVR, 5=MLP: "))

            if model_choice == 1:
                model = best_rf
                model_name = "Random Forest"
            elif model_choice == 2:
                model = best_lr
                model_name = "Linear Regression"
            elif model_choice == 3:
                model = best_xgb
                model_name = "XGBoost"
            elif model_choice == 4:
                model = best_svr
                model_name = "SVR"
            elif model_choice == 5:
                model = best_mlp
                model_name = "MLP (PyTorch)"
            else:
                print("Invalid choice")
                continue

            new_data = pd.DataFrame(
                [[year, present_price, kms, fuel, seller, transmission, owner]], 
                columns=X.columns
            )

            new_scaled = scaler.transform(new_data)
            if model_name == "MLP (PyTorch)":
                model.eval()
                with torch.no_grad():
                    input_tensor = torch.tensor(new_scaled, dtype=torch.float32).to(next(model.parameters()).device)
                    predicted_price = model(input_tensor).cpu().numpy()[0][0]
            else:
                predicted_price = model.predict(new_scaled)[0]

            print(f"\nPredicted car price ({model_name}): {predicted_price:.2f} lakhs")

        except ValueError:
            print("Invalid input! Please enter numbers only.")
        except Exception as e:
            print(f"Unexpected error: {e}")

        again = input("Do you want to make another prediction? (y/n): ").lower()
        if again != "y":
            break


# Main
def main():
    base_plot_dir = setup_folders()
    data, X, y, scaler, X_scaled = load_and_preprocess_data()
    if data is None:
        print("Data loading failed. Exiting program.")
        return
    else:
        print("\nData loaded successfully.")
        print("=== Dataset Preview ===")
        print(data.head())

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("\n=== Dataset Size ===")
    print(f"Total: {len(y)}")
    print(f"Train: {len(y_train)}")
    print(f"Test: {len(y_test)}")

    results, best_lr, best_rf, best_xgb, best_svr, best_mlp, y_pred_lr, y_pred_rf, y_pred_xgb, y_pred_svr, y_pred_mlp = train_models(X_train, y_train, X_test, y_test, X)

    print("\n=== Model Performance ===")
    print(pd.DataFrame(results).T)

    model_list = [
        ("Linear_Regression", y_pred_lr),
        ("Random_Forest", y_pred_rf),
        ("XGBoost", y_pred_xgb),
        ("SVR", y_pred_svr),
        ("MLP_Torch", y_pred_mlp)
    ]

    for model_name, preds in model_list:
        error_analysis(y_test, preds, model_name, base_plot_dir)
        plot_actual_vs_pred(y_test, preds, model_name, base_plot_dir)

    predict_price_console(best_lr, best_rf, best_xgb, best_svr, best_mlp, scaler, X)


if __name__ == "__main__":
    main()
