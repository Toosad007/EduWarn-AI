from pathlib import Path

import io
import joblib
import numpy as np
import pandas as pd


MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "student_dropout_model.pkl")


def load_prediction_model():
    """Tải bộ tiền xử lý và mô hình đã huấn luyện."""
    data = joblib.load(MODEL_PATH)
    return data["preprocessor"], data["model"]


def get_template(preprocessor):
    """
    Tạo file mẫu có đầy đủ các cột mà mô hình yêu cầu.
    Người dùng sẽ điền dữ liệu sinh viên vào file này.
    """
    columns = list(preprocessor.feature_names_in_)

    template = pd.DataFrame(columns=["Ma_sinh_vien"] + columns)

    return template


def read_uploaded_file(uploaded_file):
    """Đọc dữ liệu từ file CSV hoặc Excel."""
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Chỉ hỗ trợ file CSV hoặc Excel.")


def predict_batch(df, preprocessor, model):
    """
    Kiểm tra dữ liệu, tiền xử lý và dự đoán hàng loạt.
    Trả về bảng kết quả và số liệu kiểm tra.
    """

    expected_columns = list(preprocessor.feature_names_in_)
    input_df = df.copy()

    # Loại bỏ khoảng trắng ở tên cột
    input_df.columns = input_df.columns.astype(str).str.strip()

    # Chuyển ô rỗng hoặc chỉ chứa khoảng trắng thành NaN.
    # SimpleImputer trong preprocessor sẽ xử lý các giá trị này.
    input_df = input_df.replace(r"^\s*$", np.nan, regex=True)

    # Kiểm tra cột trùng tên
    if input_df.columns.duplicated().any():
        raise ValueError("File có tên cột bị trùng. Hãy kiểm tra lại.")

    # Nhận diện cột mã sinh viên nếu có
    id_column = None

    for col in ["Ma_sinh_vien", "Mã sinh viên", "Student_ID", "Student ID"]:
        if col in input_df.columns:
            id_column = col
            break

    if id_column:
        student_ids = input_df[id_column].copy()
    else:
        student_ids = pd.Series(
            [f"SV{i:03d}" for i in range(1, len(input_df) + 1)]
        )

    # Kiểm tra các cột bắt buộc
    missing_columns = [
        col for col in expected_columns
        if col not in input_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "File đang thiếu các cột bắt buộc:\n- "
            + "\n- ".join(missing_columns)
        )

    # Chỉ lấy đúng các cột mà mô hình đã được huấn luyện
    model_input = input_df[expected_columns].copy()

    if model_input.empty:
        raise ValueError("File không có dữ liệu sinh viên.")

    # Lấy danh sách đặc trưng phân loại và số
    categorical_features = list(preprocessor.transformers_[0][2])
    numerical_features = list(preprocessor.transformers_[1][2])

    # Chuyển các cột số về kiểu numeric.
    # Ô trống được giữ là NaN để SimpleImputer tự điền.
    for col in numerical_features:
        original = model_input[col]
        converted = pd.to_numeric(original, errors="coerce")

        # Chỉ báo lỗi khi ô có dữ liệu nhưng không chuyển được sang số.
        invalid = original.notna() & converted.isna()
        if invalid.any():
            raise ValueError(
                f"Cột '{col}' có dữ liệu không phải số. "
                "Hãy kiểm tra lại file đầu vào."
            )

        model_input[col] = converted

    # Chuyển các cột phân loại về đúng kiểu dữ liệu lúc huấn luyện.
    # cat là Pipeline; OneHotEncoder nằm bên trong named_steps.
    categorical_pipeline = preprocessor.named_transformers_["cat"]
    encoder = categorical_pipeline.named_steps["encoder"]

    for i, col in enumerate(categorical_features):
        categories = encoder.categories_[i]

        if np.issubdtype(categories.dtype, np.number):
            original = model_input[col]
            converted = pd.to_numeric(original, errors="coerce")

            invalid = original.notna() & converted.isna()
            if invalid.any():
                raise ValueError(
                    f"Cột '{col}' có mã phân loại không hợp lệ."
                )

            model_input[col] = converted

    # Kiểm tra dữ liệu số có chứa giá trị vô hạn không.
    # NaN được phép vì SimpleImputer sẽ xử lý ở bước tiền xử lý.
    numeric_values = model_input[numerical_features].to_numpy(dtype=float)
    if np.isinf(numeric_values).any():
        raise ValueError("Dữ liệu số chứa giá trị vô hạn.")

    # Tiền xử lý và dự đoán
    processed_data = preprocessor.transform(model_input)

    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)

    # Nhận diện cột họ tên sinh viên nếu có
    name_column = None
    for col in ["Ho_va_ten", "Họ và tên", "Ho_ten", "Full_Name", "Name"]:
        if col in input_df.columns:
            name_column = col
            break

    # Tạo bảng kết quả
    result_dict = {"Ma_sinh_vien": student_ids.values}
    if name_column:
        result_dict["Ho_va_ten"] = input_df[name_column].fillna("").values
    result_dict["Ket_qua_du_doan"] = predictions
    result = pd.DataFrame(result_dict)

    for i, class_name in enumerate(model.classes_):
        result[f"{class_name}_(%)"] = np.round(
            probabilities[:, i] * 100, 2
        )

    return result


def export_excel(result_df):
    """Xuất kết quả dự đoán thành file Excel."""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result_df.to_excel(
            writer,
            index=False,
            sheet_name="Ket_qua_du_doan"
        )

    output.seek(0)
    return output.getvalue()