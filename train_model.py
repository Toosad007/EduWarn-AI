# -*- coding: utf-8 -*-
"""
Script huấn luyện và đối sánh mô hình dự đoán nguy cơ bỏ học sinh viên (UCI Dataset 697)
- Loại bỏ 6 thuộc tính Học kỳ 2 (chống Data Leakage)
- Sửa chuẩn hóa: đưa 5 cột đếm Học kỳ 1 về đúng nhánh số (num) thay vì OneHotEncoder
- Đối sánh 6 thuật toán trên Stratified 80/20 + Stratified 5-Fold CV
- Đóng gói mô hình tối ưu vào student_dropout_model.pkl (tương thích 100% với app Flask)
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "data.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_OUT = MODELS_DIR / "student_dropout_model.pkl"
BENCHMARK_OUT = MODELS_DIR / "training_benchmark.json"


def load_and_clean_uci():
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Marital status": "Marital Status"})

    # Loại bỏ 6 cột Học kỳ 2 để chống Data Leakage
    sem2_cols = [c for c in df.columns if "2nd sem" in c]
    df = df.drop(columns=sem2_cols)

    y = df["Target"].astype(str)
    X = df.drop(columns=["Target"])
    return X, y


def build_preprocessor(X):
    numerical_features = [
        "Previous qualification (grade)",
        "Admission grade",
        "Age at enrollment",
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
        "Unemployment rate",
        "Inflation rate",
        "GDP",
    ]
    categorical_features = [c for c in X.columns if c not in numerical_features]

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
    ])
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, categorical_features),
            ("num", num_pipeline, numerical_features),
        ]
    )
    return preprocessor, categorical_features, numerical_features


def main():
    X, y = load_and_clean_uci()
    print(f"Dataset sạch: {X.shape[0]} mẫu, {X.shape[1]} thuộc tính (đã loại 6 cột HK2)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor, cat_cols, num_cols = build_preprocessor(X)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    print(f"Sau ColumnTransformer: {X_train_t.shape[1]} đặc trưng ({len(cat_cols)} cat + {len(num_cols)} num)")

    candidates = {
        "Logistic Regression (L2, C=0.5)": LogisticRegression(
            C=0.5, penalty="l2", solver="lbfgs", max_iter=2000, random_state=42
        ),
        "Logistic Regression (Balanced)": LogisticRegression(
            C=0.5, penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=2000, random_state=42
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=5, l2_regularization=1.0, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=16, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "SVM (RBF Kernel)": SVC(
            C=1.5, kernel="rbf", probability=True, random_state=42
        ),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree (CART)": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    fitted_models = {}

    for name, clf in candidates.items():
        cv_f1 = cross_val_score(clf, X_train_t, y_train, cv=cv, scoring="f1_macro").mean()
        clf.fit(X_train_t, y_train)
        y_pred = clf.predict(X_test_t)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        rep = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=["Dropout", "Enrolled", "Graduate"]).tolist()

        row = {
            "model": name,
            "cv5_macro_f1": round(float(cv_f1), 4),
            "test_accuracy": round(float(acc), 4),
            "test_macro_f1": round(float(macro_f1), 4),
            "f1_dropout": round(float(rep["Dropout"]["f1-score"]), 4),
            "f1_enrolled": round(float(rep["Enrolled"]["f1-score"]), 4),
            "f1_graduate": round(float(rep["Graduate"]["f1-score"]), 4),
            "confusion_matrix": cm,
        }
        results.append(row)
        fitted_models[name] = clf
        print(
            f"{name:32s} | CV5-F1={row['cv5_macro_f1']:.4f} | "
            f"Acc={row['test_accuracy']*100:.2f}% | MacroF1={row['test_macro_f1']:.4f} | "
            f"Drop={row['f1_dropout']:.3f} Enr={row['f1_enrolled']:.3f} Grad={row['f1_graduate']:.3f}"
        )

    # Chọn Logistic Regression (L2, C=0.5) làm mô hình triển khai vì vừa cao nhất/ổn định vừa có hệ số giải thích trực quan
    # (hoặc nếu Logistic Regression Balanced có CV5-F1 cao hơn rõ rệt)
    lr_std = next(r for r in results if r["model"] == "Logistic Regression (L2, C=0.5)")
    lr_bal = next(r for r in results if r["model"] == "Logistic Regression (Balanced)")
    chosen_name = "Logistic Regression (Balanced)" if lr_bal["cv5_macro_f1"] > lr_std["cv5_macro_f1"] + 0.01 else "Logistic Regression (L2, C=0.5)"
    chosen_model = fitted_models[chosen_name]

    # Refit trên toàn bộ tập dữ liệu (hoặc giữ bản fit trên train — fit trên train giúp tái lập kết quả test nhất quán)
    print(f"\n>>> MÔ HÌNH ĐƯỢC CHỌN ĐÓNG GÓI: {chosen_name}")
    feat_names = preprocessor.get_feature_names_out()
    for idx, cls in enumerate(chosen_model.classes_):
        coefs = chosen_model.coef_[idx]
        top_pos = np.argsort(coefs)[-4:][::-1]
        print(f"  Top + cho {cls}: {[(feat_names[i], round(float(coefs[i]), 3)) for i in top_pos]}")

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": chosen_model,
            "chosen_name": chosen_name,
            "benchmark": results,
        },
        MODEL_OUT,
    )
    BENCHMARK_OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu mô hình mới tại: {MODEL_OUT} ({MODEL_OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
