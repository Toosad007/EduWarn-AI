import os
import io
import uuid
import unicodedata
from pathlib import Path

# Đảm bảo thư mục làm việc luôn là thư mục chứa file flask_app.py này
os.chdir(Path(__file__).resolve().parent)

import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, Response

from modules.prediction import (
    load_prediction_model,
    validate_student_data,
    predict_student,
    explain_prediction,
    get_student_guidance,
    build_student_profile_dict,
    display_category,
    code_descriptions,
    labels,
)
from modules.batch_prediction import (
    get_template,
    read_uploaded_file,
    predict_batch,
    export_excel,
)
from modules.rag_chatbot import (
    check_ollama,
    resolve_chat_model,
    build_knowledge_base,
    get_collection,
    retrieve_context,
    generate_answer,
    generate_advice,
    get_direct_faq_answer,
    get_smart_fallback_answer,
    _clean_latex_symbols,
    DOCUMENT_FOLDER,
    CHAT_MODEL,
    EMBED_MODEL,
    AVAILABLE_CHAT_MODELS,
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'eduwarn-ai-secret-key-2026'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Bộ nhớ tạm lưu kết quả dự đoán hàng loạt để tải Excel
batch_cache = {}
_student_roster_cache = None


def _fold_vi(text):
    """Chuẩn hóa tiếng Việt không dấu, chữ thường để tìm kiếm nhanh."""
    s = unicodedata.normalize("NFKD", str(text or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.replace("đ", "d").replace("Đ", "d").lower().strip()


def load_student_roster():
    """Đọc danh sách sinh viên có sẵn (CSV/Excel) một lần để tra cứu nhanh và cho AI trả lời."""
    global _student_roster_cache
    if _student_roster_cache is not None:
        return _student_roster_cache

    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "data" / "danh_sach_100_sinh_vien_day_du.csv"
    xlsx_path = base_dir / "data" / "danh_sach_100_sinh_vien_day_du.xlsx"
    df = None
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif xlsx_path.exists():
        df = pd.read_excel(xlsx_path)

    roster = []
    if df is not None and not df.empty:
        expected_cols = list(preprocessor.feature_names_in_)
        for _, row in df.iterrows():
            sid = str(row.get("Ma_sinh_vien", "")).strip()
            sname = str(row.get("Ho_va_ten", "")).strip()
            features = {}
            select_labels = {}
            for col in expected_cols:
                val = row.get(col)
                if pd.isna(val):
                    features[col] = None
                elif isinstance(val, (int, float)):
                    features[col] = int(val) if float(val).is_integer() else round(float(val), 2)
                else:
                    features[col] = val
                if col in code_descriptions and features[col] is not None:
                    select_labels[col] = display_category(col, features[col])

            course_name = select_labels.get("Course", "Chưa rõ ngành")
            grade20 = features.get("Curricular units 1st sem (grade)")
            gpa4 = round(float(grade20) / 5.0, 2) if grade20 is not None else None
            has_debt = (features.get("Debtor") == 1) or (features.get("Tuition fees up to date") == 0)

            roster.append({
                "student_id": sid,
                "student_name": sname,
                "course_name": course_name,
                "enrolled": features.get("Curricular units 1st sem (enrolled)"),
                "approved": features.get("Curricular units 1st sem (approved)"),
                "gpa4": gpa4,
                "has_debt": has_debt,
                "features": features,
                "select_labels": select_labels,
                "_search_key": f"{_fold_vi(sid)} {_fold_vi(sname)} {_fold_vi(course_name)}",
                "_name_folded": _fold_vi(sname),
            })

    _student_roster_cache = roster
    return roster


def find_students_in_query(question, max_hits=2):
    """Tìm sinh viên được nhắc tên hoặc MSSV trong câu hỏi Chatbot để cấp ngữ cảnh chi tiết cho AI."""
    q_folded = _fold_vi(question)
    if not q_folded:
        return []
    hits = []
    for st in load_student_roster():
        sid = st["student_id"]
        name_f = st["_name_folded"]
        if (sid and sid in q_folded) or (len(name_f) >= 5 and name_f in q_folded):
            hits.append(st)
            if len(hits) >= max_hits:
                break
    return hits


def format_student_rag_context(st):
    """Chạy mô hình ML cho sinh viên tra cứu được và tạo đoạn ngữ cảnh đầy đủ cho AI trả lời."""
    prediction, probs = predict_student(preprocessor, model, st["features"])
    prob_map = {str(lbl): round(float(p) * 100, 1) for lbl, p in zip(model.classes_, probs)}
    exps = explain_prediction(preprocessor, model, st["features"], target_class=prediction, top_n=5)
    profile = build_student_profile_dict(
        st["features"],
        student_name=st["student_name"],
        student_id=st["student_id"],
        student_class=st["course_name"],
        explanations=exps,
    )
    lines = [
        f"[HỒ SƠ SINH VIÊN THỰC TẾ TRONG HỆ THỐNG: {st['student_name']} - MSSV {st['student_id']}]",
        f"- Kết quả dự đoán mô hình AI: {prediction} (Bỏ học: {prob_map.get('Dropout', 0)}%, Đang học: {prob_map.get('Enrolled', 0)}%, Tốt nghiệp: {prob_map.get('Graduate', 0)}%)",
    ]
    for k, v in profile.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)

# Tải mô hình một lần khi khởi động ứng dụng
print("Đang nạp mô hình Machine Learning...")
preprocessor, model = load_prediction_model()
print("Mô hình đã sẵn sàng!")


# ==================================================
# 1. CÁC ROUTE GIAO DIỆN CHÍNH
# ==================================================

@app.route("/")
def index():
    return render_template("index.html", active_page="index")


@app.route("/du-doan-ca-nhan")
def predict_single():
    return render_template("predict_single.html", active_page="predict_single")


@app.route("/du-doan-hang-loat")
def predict_batch_view():
    return render_template("predict_batch.html", active_page="predict_batch")


@app.route("/chatbot")
def chatbot_view():
    DOCUMENT_FOLDER.mkdir(parents=True, exist_ok=True)
    documents = [
        f.name for f in DOCUMENT_FOLDER.iterdir()
        if f.is_file() and f.suffix.lower() in [".pdf", ".docx", ".txt"]
    ]
    total_chunks = 0
    try:
        col = get_collection()
        total_chunks = col.count()
    except Exception as e:
        print(f"Lỗi lấy thông tin ChromaDB: {e}")

    return render_template(
        "chatbot.html",
        active_page="chatbot",
        documents=sorted(documents),
        total_chunks=total_chunks,
        chat_model=CHAT_MODEL,
        embed_model=EMBED_MODEL,
        available_models=AVAILABLE_CHAT_MODELS,
    )


# ==================================================
# 2. CÁC API DỰ ĐOÁN & XỬ LÝ DỮ LIỆU
# ==================================================

@app.route("/api/predict-single", methods=["POST"])
def api_predict_single():
    """API dự đoán cho một sinh viên."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Không nhận được dữ liệu."}), 400

        errors = validate_student_data(data, labels)
        if errors:
            return jsonify({"status": "error", "message": "; ".join(errors)}), 400

        prediction, probabilities = predict_student(preprocessor, model, data)
        explanations = explain_prediction(preprocessor, model, data, target_class=prediction, top_n=5)
        title, explanation, concerns, actions, close_result = get_student_guidance(
            prediction, probabilities, data
        )

        prob_map = {
            str(label): round(float(prob) * 100, 2)
            for label, prob in zip(model.classes_, probabilities)
        }

        return jsonify({
            "status": "success",
            "prediction": str(prediction),
            "probabilities": prob_map,
            "explanations": explanations,
            "title": title,
            "explanation": explanation,
            "concerns": concerns,
            "actions": actions,
            "close_result": bool(close_result),
            "input_values": data
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/generate-advice", methods=["POST"])
def api_generate_advice():
    """API gọi Ollama sinh tư vấn chuyên sâu và mẫu tin nhắn với đầy đủ thông tin sinh viên."""
    try:
        req = request.get_json()
        prediction = req.get("prediction")
        probabilities = req.get("probabilities", {})
        input_values = req.get("input_values", {})
        explanations = req.get("explanations") or explain_prediction(
            preprocessor, model, input_values, target_class=prediction, top_n=5
        )
        student_name = (req.get("student_name") or "").strip()
        student_id = (req.get("student_id") or "").strip()
        student_class = (req.get("student_class") or "").strip()

        selected_fields = build_student_profile_dict(
            input_values,
            student_name=student_name,
            student_id=student_id,
            student_class=student_class,
            explanations=explanations,
        )

        retrieval_query = (
            "Quy chế cảnh báo học vụ, quy trình hỗ trợ sinh viên nợ môn, "
            "chính sách gia hạn nộp học phí và tư vấn học tập."
        )

        contexts, sources = retrieve_context(retrieval_query, top_k=3)
        advice_generator = generate_advice(
            prediction, probabilities, selected_fields, contexts
        )
        full_advice = _clean_latex_symbols("".join(advice_generator))
        if student_name:
            full_advice = full_advice.replace("[Tên sinh viên]", student_name)

        return jsonify({
            "status": "success",
            "advice": full_advice,
            "sources": sources
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict-batch", methods=["POST"])
def api_predict_batch():
    """API tải lên file Excel/CSV và dự đoán hàng loạt."""
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "Chưa chọn file tải lên."}), 400

        uploaded_file = request.files["file"]
        if uploaded_file.filename == "":
            return jsonify({"status": "error", "message": "Tên file không hợp lệ."}), 400

        filename = uploaded_file.filename.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return jsonify({"status": "error", "message": "Chỉ hỗ trợ file CSV hoặc Excel (.xlsx, .xls)."}), 400

        result_df = predict_batch(df, preprocessor, model)

        token = str(uuid.uuid4())
        batch_cache[token] = result_df

        records = result_df.to_dict(orient="records")
        return jsonify({
            "status": "success",
            "token": token,
            "data": records
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ==================================================
# 3. TẢI FILE MẪU VÀ XUẤT BÁO CÁO EXCEL
# ==================================================

@app.route("/download/template-csv")
def download_template_csv():
    template_df = get_template(preprocessor)
    csv_bytes = template_df.to_csv(index=False).encode("utf-8-sig")
    return send_file(
        io.BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name="file_mau_du_doan.csv"
    )


@app.route("/download/template-excel")
def download_template_excel():
    template_df = get_template(preprocessor)
    excel_bytes = export_excel(template_df)
    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="file_mau_du_doan.xlsx"
    )


@app.route("/export/batch-excel")
def export_batch_excel():
    token = request.args.get("token")
    if not token or token not in batch_cache:
        return "File kết quả không tồn tại hoặc đã hết hạn.", 404

    result_df = batch_cache[token]
    excel_bytes = export_excel(result_df)
    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="Ket_qua_du_doan_hang_loat.xlsx"
    )


# ==================================================
# 4. API CHATBOT AI VÀ KHO TRI THỨC
# ==================================================

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Hỏi đáp với chatbot AI RAG (Gemma 4 12B + bge-m3 + trí nhớ hội thoại + Fallback)."""
    question = ""
    try:
        data = request.get_json() or {}
        question = data.get("message", "").strip()
        history = data.get("history", [])
        selected_model = resolve_chat_model(data.get("model"))
        if not question:
            return jsonify({"status": "error", "message": "Câu hỏi không được rỗng."}), 400

        # Chặn nhanh các chuỗi vô nghĩa / chỉ toàn ký tự đặc biệt hoặc 1 ký tự đơn lẻ
        alnum_chars = [c for c in question if c.isalnum()]
        if len(alnum_chars) < 2:
            return jsonify({
                "status": "success",
                "answer": "Mình chưa hiểu rõ câu hỏi của bạn. Bạn vui lòng nhập rõ hơn về vấn đề học vụ cần hỗ trợ (ví dụ: quy chế cảnh báo học vụ, học bổng/GPA, nợ học phí, bảo lưu, chuyển ngành, hoặc tên/MSSV sinh viên cần tra cứu) nhé!",
                "sources": [],
                "model_used": "Hỏi lại để làm rõ"
            })

        # Kiểm tra xem người dùng có đang hỏi về sinh viên cụ thể (theo tên hoặc MSSV) không
        matched_students = find_students_in_query(question)
        student_contexts = [format_student_rag_context(st) for st in matched_students]

        # ƯU TIÊN 1: Dùng Ollama LLM + Hybrid RAG (+ hồ sơ sinh viên thực tế nếu có)
        ready, msg = check_ollama(selected_model)
        if ready:
            contexts, sources = retrieve_context(question, top_k=5)
            if student_contexts:
                contexts = student_contexts + contexts
                sources = ["danh_sach_100_sinh_vien_day_du.xlsx"] + sources
            answer_generator = generate_answer(
                question, contexts, chat_history=history, model_name=selected_model
            )
            full_answer = _clean_latex_symbols("".join(answer_generator).strip())
            if full_answer:
                return jsonify({
                    "status": "success",
                    "answer": full_answer,
                    "sources": sources,
                    "model_used": selected_model
                })

        # Nếu có tra cứu đúng sinh viên nhưng Ollama không chạy -> trả thẳng bảng hồ sơ chi tiết
        if student_contexts:
            return jsonify({
                "status": "success",
                "answer": "\n\n---\n\n".join(student_contexts),
                "sources": ["danh_sach_100_sinh_vien_day_du.xlsx"],
                "model_used": "Tra cứu hồ sơ sinh viên trực tiếp"
            })

        # ƯU TIÊN 2 (Dự phòng khi Ollama chưa chạy hoặc trả lời từ chối): Direct FAQ / Smart Fallback
        direct_ans, direct_srcs = get_direct_faq_answer(question)
        if direct_ans:
            return jsonify({
                "status": "success",
                "answer": direct_ans,
                "sources": direct_srcs,
                "model_used": "Tra cứu nhanh (Offline FAQ)"
            })

        answer, sources = get_smart_fallback_answer(question)
        return jsonify({
            "status": "success",
            "answer": answer,
            "sources": sources,
            "model_used": "Tra cứu tài liệu (Fallback)"
        })

    except Exception:
        direct_ans, direct_srcs = get_direct_faq_answer(question)
        if direct_ans:
            return jsonify({"status": "success", "answer": direct_ans, "sources": direct_srcs})
        answer, sources = get_smart_fallback_answer(question)
        return jsonify({"status": "success", "answer": answer, "sources": sources})


@app.route("/api/rebuild-knowledge", methods=["POST"])
def api_rebuild_knowledge():
    """Cập nhật lại kho tri thức ChromaDB."""
    try:
        result = build_knowledge_base()
        return jsonify({
            "status": "success",
            "document_count": result.get("document_count", 0),
            "chunk_count": result.get("chunk_count", 0)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    # Chạy trên port 5000 với chế độ tự động tải lại khi chỉnh sửa code / HTML / CSS
    print("=" * 60)
    print("Khởi chạy ứng dụng Flask (Auto-Reload) tại: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=True)
