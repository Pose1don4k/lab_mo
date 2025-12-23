import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=ConvergenceWarning)


data = pd.read_csv(
    "C:\\моё\\Универ\\5 семетср\\МО\\Sensorless_drive_diagnosis.txt",
    sep=r"\s+",
    header=None
)

X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values   

print("Форма X:", X.shape)
print("Форма y:", y.shape)
print("Уникальные классы:", np.unique(y))


X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)


scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Перцептрон

perceptron = Perceptron(
    eta0=0.01,
    max_iter=1000,
    random_state=42,
    penalty=None,
    tol=1e-3
)
perceptron.fit(X_train_scaled, y_train)

y_val_pred_p = perceptron.predict(X_val_scaled)
y_test_pred_p = perceptron.predict(X_test_scaled)

print("\nПерцептрон ")
print("Accuracy (val):  ", accuracy_score(y_val, y_val_pred_p))
print("Accuracy (test): ", accuracy_score(y_test, y_test_pred_p))


# Базовый MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=100,
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

y_val_pred_mlp = mlp.predict(X_val_scaled)
y_test_pred_mlp = mlp.predict(X_test_scaled)

print("\nБазовый MLPClassifier")
print("Accuracy (val):  ", accuracy_score(y_val, y_val_pred_mlp))
print("Accuracy (test): ", accuracy_score(y_test, y_test_pred_mlp))




# Эксперименты Перцептрон

eta_list = [0.0001, 0.001, 0.01, 0.1]
penalties = [None, 'l2', 'l1', 'elasticnet']

results_perceptron = []

for eta in eta_list:
    for pen in penalties:
        clf = Perceptron(
            eta0=eta,
            max_iter=1000,
            penalty=pen,
            random_state=42,
            tol=1e-3
        )
        clf.fit(X_train_scaled, y_train)
        val_pred = clf.predict(X_val_scaled)
        acc_val = accuracy_score(y_val, val_pred)
        results_perceptron.append({
            "eta0": eta,
            "penalty": pen,
            "val_accuracy": acc_val
        })

results_perceptron_df = pd.DataFrame(results_perceptron)
print("\nРезультаты экспериментов с Перцептроном (val accuracy):")
print(results_perceptron_df)


#GridSearchCV для MLP


X_train_gs, _, y_train_gs, _ = train_test_split(
    X_train_scaled, y_train,
    test_size=0.7,
    random_state=42,
    stratify=y_train
)

print("\nРазмер выборки для GridSearchCV:", X_train_gs.shape)


param_grid_mlp = {
    "hidden_layer_sizes": [(50,), (100,)],
    "activation": ["relu"],
    "solver": ["adam"],
    "alpha": [0.0001, 0.001],
    "learning_rate_init": [0.001, 0.01]
}

mlp_base = MLPClassifier(
    max_iter=200,
    early_stopping=True,
    n_iter_no_change=5,
    random_state=42
)

grid_mlp = GridSearchCV(
    estimator=mlp_base,
    param_grid=param_grid_mlp,
    scoring="accuracy",
    cv=2,
    n_jobs=1,
    verbose=0
)

grid_mlp.fit(X_train_gs, y_train_gs)

print("\nЛучшие параметры MLP (GridSearchCV):")
print(grid_mlp.best_params_)
print("Лучшая CV-точность:", grid_mlp.best_score_)


best_mlp = grid_mlp.best_estimator_

y_val_pred_best = best_mlp.predict(X_val_scaled)
y_test_pred_best = best_mlp.predict(X_test_scaled)

print("\nЛучший MLP (после GridSearchCV)")
print("Accuracy (val):  ", accuracy_score(y_val, y_val_pred_best))
print("Accuracy (test): ", accuracy_score(y_test, y_test_pred_best))


