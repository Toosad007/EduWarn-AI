
import hashlib
from pathlib import Path
import json
import requests

# Bọc kiểm tra chromadb để tránh sập toàn bộ ứng dụng nếu môi trường thiếu thư viện
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except Exception:
    chromadb = None
    CHROMADB_AVAILABLE = False

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document
except Exception:
    Document = None


# ==================================================
# 1. CẤU HÌNH
# ==================================================

OLLAMA_URL = "http://localhost:11434"

CHAT_MODEL = "qwen3.5:4b"
EMBED_MODEL = "bge-m3:latest"
PREFERRED_CHAT_MODELS = ["qwen3.5:4b", "qwen2.5:3b", "gemma4:12b"]
AVAILABLE_CHAT_MODELS = PREFERRED_CHAT_MODELS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENT_FOLDER = PROJECT_ROOT / "knowledge_base" / "documents"
CHROMA_FOLDER = PROJECT_ROOT / "knowledge_base" / "chroma_db"

COLLECTION_NAME = "student_dropout_knowledge"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# ==================================================
# 2. KẾT NỐI CHROMADB
# ==================================================

def get_collection():
    """Kết nối hoặc tạo collection lưu vector."""
    if not CHROMADB_AVAILABLE or chromadb is None:
        raise RuntimeError("Chưa cài đặt thư viện chromadb trong môi trường Python hiện tại.")

    CHROMA_FOLDER.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_FOLDER)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Knowledge base for student dropout project"}
    )

    return collection


# ==================================================
# 3. KIỂM TRA OLLAMA
# ==================================================

def resolve_chat_model(preferred=None):
    """Tự động nhận diện model chat tốt nhất đang có trong Ollama (ưu tiên qwen3.5:4b)."""
    candidates = ([preferred] if preferred else []) + PREFERRED_CHAT_MODELS
    try:
        res = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        res.raise_for_status()
        installed = [m["name"] for m in res.json().get("models", [])]
        for cand in candidates:
            for name in installed:
                if name == cand or name.startswith(cand + ":"):
                    return name
        # Nếu không khớp danh sách ưu tiên, lấy model chat đầu tiên không phải embed
        for name in installed:
            if "embed" not in name.lower() and "bge" not in name.lower():
                return name
    except Exception:
        pass
    return preferred or CHAT_MODEL


def check_ollama(chat_model=None):
    """Kiểm tra Ollama và hai mô hình cần thiết."""
    if not CHROMADB_AVAILABLE:
        return False, "Chưa cài đặt thư viện chromadb. Vui lòng chạy ứng dụng bằng môi trường ảo (.venv)."

    target_chat = chat_model or CHAT_MODEL
    target_embed_base = EMBED_MODEL.split(":")[0]

    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=5
        )
        response.raise_for_status()

        models = response.json().get("models", [])
        model_names = [item["name"] for item in models]

        chat_ready = any(
            name == target_chat or name.startswith(target_chat + ":")
            for name in model_names
        )
        embedding_ready = any(
            name == EMBED_MODEL or name == target_embed_base or name.startswith(target_embed_base + ":")
            for name in model_names
        )

        if not chat_ready or not embedding_ready:
            missing = []
            if not chat_ready:
                missing.append(target_chat)
            if not embedding_ready:
                missing.append(EMBED_MODEL)
            return False, "Chưa tìm thấy mô hình: " + ", ".join(missing)

        return True, f"Ollama sẵn sàng ({target_chat} + {EMBED_MODEL})."

    except requests.RequestException as e:
        return False, f"Không thể kết nối Ollama: {e}"


# ==================================================
# 4. ĐỌC TÀI LIỆU
# ==================================================

def read_txt(file_path):
    return Path(file_path).read_text(
        encoding="utf-8",
        errors="ignore"
    )


def read_pdf(file_path):
    reader = PdfReader(str(file_path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append(
                f"[Trang {page_number}]\n{text}"
            )

    return "\n\n".join(pages)


def read_docx(file_path):
    document = Document(str(file_path))

    paragraphs = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n".join(paragraphs)


def load_documents():
    """Đọc tất cả tài liệu được hỗ trợ trong thư mục."""

    DOCUMENT_FOLDER.mkdir(parents=True, exist_ok=True)

    documents = []

    for file_path in DOCUMENT_FOLDER.iterdir():

        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        try:
            if extension == ".txt":
                content = read_txt(file_path)

            elif extension == ".pdf":
                content = read_pdf(file_path)

            elif extension == ".docx":
                content = read_docx(file_path)

            else:
                continue

            if content.strip():
                documents.append({
                    "name": file_path.name,
                    "content": content
                })

        except Exception as e:
            raise ValueError(
                f"Không thể đọc file {file_path.name}: {e}"
            )

    return documents


# ==================================================
# 5. CHIA TÀI LIỆU THÀNH CÁC ĐOẠN
# ==================================================

def split_text(text):
    """Chia văn bản thông minh theo đoạn/mục (Paragraph-aware Chunking), giữ nguyên ngữ cảnh."""
    raw_paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    if not raw_paragraphs:
        return []

    doc_title = raw_paragraphs[0].splitlines()[0].strip() if raw_paragraphs else ""
    chunks = []
    current = ""

    for para in raw_paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(para) <= CHUNK_SIZE:
                prefix = f"[{doc_title}]\n" if doc_title and not para.startswith(doc_title) else ""
                current = f"{prefix}{para}".strip()
            else:
                # Nếu một đoạn đơn lẻ dài hơn CHUNK_SIZE, cắt theo dòng/câu thay vì cắt ngang từ
                lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
                sub = f"[{doc_title}]\n" if doc_title else ""
                for ln in lines:
                    if len(sub) + len(ln) + 1 <= CHUNK_SIZE:
                        sub = f"{sub}\n{ln}".strip()
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = f"[{doc_title}]\n{ln}".strip() if doc_title else ln
                current = sub

    if current:
        chunks.append(current)

    return chunks


# ==================================================
# 6. TẠO EMBEDDING BẰNG OLLAMA
# ==================================================

def create_embeddings(texts):
    """Chuyển danh sách văn bản thành vector."""

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts,
            "keep_alive": "10m"
        },
        timeout=180
    )

    response.raise_for_status()

    embeddings = response.json().get("embeddings")

    if not embeddings or len(embeddings) != len(texts):
        raise ValueError("Ollama không trả về embedding hợp lệ.")

    return embeddings


# ==================================================
# 7. XÂY DỰNG KHO TRI THỨC
# ==================================================

def _compute_documents_signature():
    """Tạo chữ ký (tên file, dung lượng, thời gian sửa) để tự phát hiện thay đổi trong kho tài liệu."""
    if not DOCUMENT_FOLDER.exists():
        return ""
    entries = []
    for path in sorted(DOCUMENT_FOLDER.iterdir()):
        if path.is_file() and path.suffix.lower() in {".txt", ".pdf", ".docx"}:
            st = path.stat()
            entries.append(f"{path.name}|{st.st_size}|{int(st.st_mtime)}")
    return "\n".join(entries)


def build_knowledge_base():
    """
    Đọc tài liệu, chia đoạn, tạo embedding
    và lưu vào ChromaDB.
    """

    documents = load_documents()

    if not documents:
        raise ValueError(
            "Chưa có tài liệu PDF, DOCX hoặc TXT "
            "trong thư mục knowledge_base/documents."
        )

    all_chunks = []
    all_metadata = []
    all_ids = []

    for document in documents:

        chunks = split_text(document["content"])

        for index, chunk in enumerate(chunks):

            chunk_id = hashlib.sha256(
                f"{document['name']}_{index}_{chunk}".encode("utf-8")
            ).hexdigest()

            all_chunks.append(chunk)

            all_metadata.append({
                "source": document["name"],
                "chunk_index": index
            })

            all_ids.append(chunk_id)

    if not all_chunks:
        raise ValueError("Không tìm thấy nội dung văn bản trong tài liệu.")

    collection = get_collection()

    # Xóa dữ liệu cũ để tránh trùng lặp khi xây dựng lại
    client = chromadb.PersistentClient(
        path=str(CHROMA_FOLDER)
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Student dropout RAG knowledge base"}
    )

    # Chia batch để tránh gửi quá nhiều văn bản cùng lúc
    batch_size = 32

    for start in range(0, len(all_chunks), batch_size):

        end = start + batch_size

        batch_texts = all_chunks[start:end]

        embeddings = create_embeddings(batch_texts)

        collection.add(
            ids=all_ids[start:end],
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=all_metadata[start:end]
        )

    # Lưu chữ ký hiện tại của thư mục documents để tự nhận biết lần sau có thay đổi hay không
    try:
        (CHROMA_FOLDER / ".kb_signature").write_text(_compute_documents_signature(), encoding="utf-8")
    except Exception:
        pass

    return {
        "document_count": len(documents),
        "chunk_count": len(all_chunks)
    }


# ==================================================
# 8. TÌM KIẾM TÀI LIỆU LIÊN QUAN
# ==================================================

def retrieve_context(question, top_k=5):
    """Tìm các đoạn tài liệu liên quan bằng Hybrid Retrieval (Vector ChromaDB + Keyword Boost)."""
    def _fallback_read():
        contexts, sources = [], []
        if DOCUMENT_FOLDER.exists():
            for doc_file in DOCUMENT_FOLDER.glob("*.txt"):
                try:
                    contexts.append(doc_file.read_text(encoding="utf-8"))
                    sources.append(doc_file.name)
                except Exception:
                    pass
        return contexts[:top_k], sources[:top_k]

    if not CHROMADB_AVAILABLE:
        return _fallback_read()

    try:
        collection = get_collection()
        sig_file = CHROMA_FOLDER / ".kb_signature"
        current_sig = _compute_documents_signature()
        saved_sig = sig_file.read_text(encoding="utf-8") if sig_file.exists() else ""

        # Tự động nạp lại ChromaDB nếu kho rỗng HOẶC có file mới / file bị sửa trong knowledge_base/documents
        if collection.count() == 0 or (current_sig and current_sig != saved_sig):
            try:
                build_knowledge_base()
                collection = get_collection()
            except Exception:
                pass

        total = collection.count()
        if total == 0:
            return _fallback_read()

        question_embedding = create_embeddings([question])[0]
        fetch_k = min(max(top_k * 2, 8), total)
        result = collection.query(
            query_embeddings=[question_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"]
        )

        # Kết hợp điểm Vector (distance càng nhỏ càng tốt) + điểm khớp từ khóa tiếng Việt
        q_tokens = [w.strip("?.,!:;()[]{}\"'") for w in question.lower().split() if len(w.strip("?.,!:;()[]{}\"'")) >= 2]
        scored = []
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            doc_lower = doc.lower()
            kw_hits = sum(1 for tok in q_tokens if tok in doc_lower)
            kw_ratio = kw_hits / max(len(q_tokens), 1)
            vec_sim = 1.0 / (1.0 + max(float(dist), 0.0))
            final_score = 0.72 * vec_sim + 0.28 * kw_ratio
            scored.append((final_score, doc, meta.get("source", "Không rõ nguồn")))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_items = scored[:top_k]

        contexts = [item[1] for item in top_items]
        sources = []
        for item in top_items:
            if item[2] not in sources:
                sources.append(item[2])

        return contexts, sources
    except Exception:
        return _fallback_read()


# ==================================================
# 9. GỬI CÂU HỎI ĐẾN LLM (GEMMA 4 / QWEN)
# ==================================================


def _clean_latex_symbols(text):
    """Chuyển các ký hiệu LaTeX thô ($\ge$, $\approx$, \%) sang Unicode sạch để hiển thị đẹp trên web."""
    if not text:
        return text
    import re
    replacements = [
        (r"\$\\geq?\$", "≥"),
        (r"\$\\leq?\$", "≤"),
        (r"\$\\approx\$", "≈"),
        (r"\$\\times\$", "×"),
        (r"\\geq?", "≥"),
        (r"\\leq?", "≤"),
        (r"\\approx", "≈"),
        (r"\\times", "×"),
        (r"\\%", "%"),
    ]
    out = text
    for pat, rep in replacements:
        out = re.sub(pat, rep, out)
    # Gỡ dấu $ bọc công thức ngắn như $< 1.0/4.0$ hoặc $50%$
    out = re.sub(r"\$([^$\n]{1,30})\$", r"\1", out)
    return out


def generate_answer(question, contexts, chat_history=None, model_name=None):
    """Tạo câu trả lời chuyên sâu từ LLM cục bộ kết hợp ngữ cảnh RAG và lịch sử hội thoại."""
    selected_model = model_name if model_name in AVAILABLE_CHAT_MODELS else CHAT_MODEL

    context_text = "\n\n".join(
        [
            f"[Tài liệu {i + 1}]\n{context}"
            for i, context in enumerate(contexts)
        ]
    )

    system_prompt = f"""Bạn là Chuyên gia Cố vấn Học vụ & Gia sư Tư vấn Học tập của hệ thống EduWarn AI (môn Trí tuệ nhân tạo ứng dụng – Trường Đại học Văn Lang).

VAI TRÒ & PHONG CÁCH:
- Giải đáp chính xác mọi thắc mắc về đồ án EduWarn AI (mô hình Logistic Regression, 30 thuộc tính, chống Data Leakage bỏ Học kỳ 2, Macro F1-Score, kiến trúc Flask & RAG).
- Tra cứu và hướng dẫn chi tiết về quy chế cảnh báo học vụ, nợ học phí, gia hạn học phí, học bổng khuyến khích, bảo lưu và chuyển ngành.
- Tư vấn học tập thực chiến, tận tâm và cụ thể cho sinh viên: lập kế hoạch trả nợ môn, nâng GPA, quản lý thời gian khi đi làm thêm/học tối, phương pháp học hiệu quả (Active Recall, Spaced Repetition, Pomodoro, Feynman) và hỗ trợ tâm lý học đường.

QUY TẮC TRẢ LỜI:
1. Ưu tiên trích xuất và tổng hợp đầy đủ từ TÀI LIỆU THAM KHẢO dưới đây; nếu câu hỏi mang tính tư vấn học tập mở rộng, kết hợp thêm kiến thức sư phạm chuẩn mực để đưa ra lộ trình hành động cụ thể, thiết thực.
2. Trình bày bằng tiếng Việt tự nhiên, rõ ràng với tiêu đề ngắn và gạch đầu dòng mạch lạc. KHÔNG dùng cú pháp LaTeX (như `$\\ge$`, `$\\le$`, `$\\approx$`, `\\%`); hãy viết trực tiếp ký tự `≥`, `≤`, `≈`, `%`.
3. Không bịa đặt con số kỹ thuật trái với tài liệu (luôn nhớ: đồ án giữ Học kỳ 1, loại bỏ 6 thuộc tính Học kỳ 2).
4. GIỚI HẠN PHẠM VI (BẮT BUỘC): Bạn CHỈ là Trợ lý AI Cố vấn Học vụ. Nếu người dùng hỏi bất kỳ chủ đề nào NGOÀI phạm vi học vụ, học tập đại học, quy chế đào tạo, học phí/học bổng, tâm lý/định hướng sinh viên, hồ sơ sinh viên hoặc đồ án EduWarn AI (ví dụ: địa điểm du lịch, lịch sử, thời tiết, thể thao, giải trí, nấu ăn, tin tức xã hội...), TUYỆT ĐỐI KHÔNG tự bịa câu trả lời và KHÔNG cố ghép tài liệu không liên quan. Hãy trả lời lịch sự, ngắn gọn trong 2 câu:
   "Xin lỗi, tôi là Trợ lý AI Cố vấn Học vụ của hệ thống EduWarn AI nên không có thông tin về chủ đề này. Tôi chỉ hỗ trợ giải đáp về quy chế đào tạo, học tập đại học, học phí/học bổng, hồ sơ sinh viên và kiến thức đồ án EduWarn AI."
5. XỬ LÝ CÂU HỎI MƠ HỒ / SAI CHÍNH TẢ / VU VƠ:
   - Nếu câu hỏi viết tắt hoặc sai chính tả nhẹ nhưng vẫn hiểu được ý học vụ (VD: "hoc bong bao nhiu", "no 2 mon lam sao"), hãy tự hiểu đúng ý và trả lời bình thường.
   - Nếu câu hỏi quá ngắn, vô nghĩa, vu vơ hoặc không đủ rõ ý để trả lời (VD: "asdf", "???", "alo", "điểm sao", "cái kia thế nào"), ĐỪNG đoán mò — hãy lịch sự yêu cầu người dùng hỏi rõ hơn và gợi ý 2–3 ví dụ cụ thể (quy chế cảnh báo học vụ, học bổng/GPA, nợ học phí, bảo lưu/chuyển ngành, hoặc tên/MSSV sinh viên).

TÀI LIỆU THAM KHẢO:
{context_text}
"""
    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        for turn in chat_history[-6:]:
            role = turn.get("role")
            content = str(turn.get("content", "")).strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": selected_model,
                "messages": messages,
                "stream": True,
                "think": False,
                "keep_alive": "20m",
                "options": {
                    "temperature": 0.25,
                    "num_predict": 2048,
                    "num_ctx": 8192
                }
            },
            stream=True,
            timeout=(15, 180)
        )
        response.raise_for_status()

        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(data["error"])
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
        finally:
            response.close()
    except Exception:
        fallback_ans, _ = get_smart_fallback_answer(question)
        yield fallback_ans


def get_direct_faq_answer(question):
    """Trả về câu trả lời trực tiếp, đầy đủ 100% cho các câu hỏi gợi ý nhanh và chủ đề phổ biến."""
    if not question:
        return None, []

    q = question.lower().strip()

    # 1. Câu hỏi về học bổng khuyến khích
    if any(k in q for k in ["học bổng", "hoc bong", "khen thưởng"]):
        return (
            "### 🏆 Tiêu Chuẩn & Chính Sách Xét Cấp Học Bổng Khuyến Khích:\n\n"
            "- **Điều kiện tham gia xét:** Sinh viên đăng ký đủ khối lượng tín chỉ tối thiểu trong học kỳ, không có học phần nào phải thi lại/học lại và điểm rèn luyện phải đạt từ loại Tốt trở lên.\n\n"
            "- **Các mức học bổng cụ thể:**\n"
            "  + **Loại Xuất Sắc:** Điểm trung bình GPA ≥ 3.6/4.0 (hoặc ≥ 16.0/20.0). Mức khen thưởng cao nhất kèm giấy khen của Hiệu trưởng.\n"
            "  + **Loại Giỏi:** Điểm trung bình GPA ≥ 3.2/4.0 (hoặc ≥ 14.0/20.0).\n"
            "  + **Loại Khá:** Điểm trung bình GPA ≥ 2.5/4.0 (hoặc ≥ 12.0/20.0).\n\n"
            "- **Quỹ hỗ trợ sinh viên vượt khó:** Đối với sinh viên có hoàn cảnh khó khăn (thuộc diện di dời, gia đình khó khăn), Nhà trường có các suất học bổng tài trợ doanh nghiệp và hỗ trợ thủ tục vay vốn ưu đãi từ Ngân hàng Chính sách Xã hội.",
            ["chinh_sach_hoc_phi_va_hoc_bong.txt"]
        )

    # 2. Câu hỏi về việc loại bỏ Học kỳ 2 / Data Leakage
    if any(k in q for k in ["học kỳ 2", "hoc ky 2", "kỳ 2", "loại bỏ", "loai bo", "leakage", "rò rỉ"]):
        return (
            "### 📌 Lý Do Đồ Án Loại Bỏ Thuộc Tính Học Kỳ 2 (Curricular Units 2nd Sem):\n\n"
            "1. **Mục tiêu cảnh báo sớm (Early Intervention):** Mục đích cốt lõi của đề tài là nhận diện nguy cơ sinh viên thôi học ngay sau khi kết thúc **Học kỳ 1 năm nhất**, giúp Nhà trường và Cố vấn học tập có đủ thời gian can thiệp trước khi sinh viên quyết định bỏ học. Nếu chờ đến khi có kết quả Học kỳ 2 thì sinh viên đã nghỉ học từ trước, làm mất đi ý nghĩa cảnh báo sớm.\n"
            "2. **Chống rò rỉ dữ liệu (Prevent Data Leakage):** Kết quả Học kỳ 2 (như không đăng ký môn, số môn đạt = 0) là hệ quả trực tiếp của hành vi đã bỏ học. Đưa dữ liệu này vào sẽ khiến mô hình 'học thuộc quá khứ' và phóng đại độ chính xác ảo thay vì dự báo thực tế.\n\n"
            "*(Hệ thống giữ lại trọn vẹn 6 chỉ số học tập của Học kỳ 1 để đánh giá năng lực thích ứng ban đầu của sinh viên)*",
            ["gioi_thieu_do_an.txt"]
        )

    # 3. Câu hỏi về thuật toán Machine Learning & Mô hình
    if any(k in q for k in ["thuật toán", "thuat toan", "mô hình", "mo hinh", "logistic", "f1", "accuracy", "độ chính xác"]):
        return (
            "### ⚙️ Thuật Toán Machine Learning & Hiệu Năng Của Đồ Án:\n\n"
            "- **Mô hình triển khai chính thức:** **Logistic Regression (Hồi quy Logistic đa lớp)** kết hợp tiền xử lý `ColumnTransformer` (`OneHotEncoder` cho biến danh mục và `StandardScaler` cho biến số).\n"
            "- **Các thuật toán đối chứng đã thử nghiệm:** Random Forest, Support Vector Machine (SVM), Decision Tree, K-Nearest Neighbors (KNN).\n"
            "- **Vì sao chọn Logistic Regression?**\n"
            "  + **Tính giải thích cao (Explainability):** Dễ dàng diễn giải trọng số từng biến số cho Cố vấn học tập và sinh viên hiểu vì sao bị cảnh báo.\n"
            "  + **Xuất xác suất tin cậy:** Trả về trực tiếp % xác suất cho cả 3 nhóm: **Dropout, Enrolled, Graduate**.\n"
            "  + **Độ chính xác:** Đạt Accuracy ≈ 77% và Macro F1-score ≈ 0.70, hoạt động ổn định và suy luận tức thì.",
            ["gioi_thieu_do_an.txt"]
        )

    # 4. Câu hỏi về cảnh báo học vụ và buộc thôi học
    if any(k in q for k in ["cảnh báo", "canh bao", "buộc thôi học", "buoc thoi hoc", "quy chế", "quy che", "mức 1", "mức 2", "mức 3"]):
        return (
            "### ⚠️ Quy Định Về Các Mức Cảnh Báo Học Vụ & Buộc Thôi Học:\n\n"
            "- **Cảnh báo Mức 1:** Áp dụng khi điểm GPA học kỳ < 1.0/4.0 (hoặc < 10.0/20.0), hoặc số tín chỉ tích lũy đạt dưới 50% số tín chỉ đã đăng ký trong kỳ.\n"
            "- **Cảnh báo Mức 2:** Bị cảnh báo Mức 1 trong 2 học kỳ liên tiếp, hoặc điểm CPA tích lũy tụt dưới chuẩn năm đào tạo.\n"
            "- **Cảnh báo Mức 3 (Báo động đỏ):** Bị giới hạn đăng ký tối đa 14 tín chỉ trong kỳ tiếp theo và bắt buộc phải tham gia các buổi tư vấn định kỳ cùng Cố vấn học tập.\n"
            "- **Buộc thôi học:** Bị cảnh báo Mức 3 trong 2 kỳ liên tiếp, hoặc điểm tích lũy cả năm không đạt chuẩn tối thiểu, hoặc vượt quá thời gian đào tạo tối đa cho phép.",
            ["quy_che_canh_bao_hoc_vu.txt"]
        )

    # 5. Câu hỏi về nợ học phí và xử lý tài chính (tránh nhầm với 'nợ môn')
    if any(k in q for k in ["học phí", "hoc phi", "nợ học phí", "no hoc phi", "tiền học"]):
        return (
            "### 💰 Cách Xử Lý Khi Sinh Viên Nợ Học Phí:\n\n"
            "1. **Làm đơn xin gia hạn:** Sinh viên gặp khó khăn kinh tế nộp **Đơn xin gia hạn nộp học phí** trước hạn chót 7 ngày (có xác nhận hoàn cảnh từ gia đình/địa phương). Được chia đợt nộp hoặc gia hạn đến trước tuần thi kết thúc môn.\n"
            "2. **Hỗ trợ từ Nhà trường:** Kết nối với Ban Cố vấn học tập để xin trợ cấp từ Quỹ hỗ trợ sinh viên khó khăn hoặc hướng dẫn thủ tục vay vốn sinh viên ưu đãi.\n"
            "3. **Hệ quả nếu để nợ quá hạn:** Sẽ bị tạm khóa tài khoản đăng ký môn kỳ tiếp theo, không được dự thi và không được xét học bổng khuyến khích.",
            ["chinh_sach_hoc_phi_va_hoc_bong.txt", "so_tay_co_van_hoc_tap.txt"]
        )

    # 6. Câu hỏi về các yếu tố ảnh hưởng đến bỏ học
    if any(k in q for k in ["yếu tố", "yeu to", "ảnh hưởng", "anh huong", "quan trọng", "nguyên nhân"]):
        return (
            "### 📊 Các Yếu Tố Ảnh Hưởng Lớn Nhất Đến Nguy Cơ Bỏ Học (Dropout):\n\n"
            "1. **Tình trạng học phí (Debtor & Tuition fees up to date):** Sinh viên nợ học phí có nguy cơ bỏ học cao hơn gấp 3 lần bình thường do gánh nặng tài chính.\n"
            "2. **Số học phần đạt ở kỳ 1 (Curricular units 1st sem approved):** Sinh viên nợ từ 2 môn trở lên ở kỳ 1 có tỷ lệ chán nản và thôi học tăng vọt.\n"
            "3. **Điểm trung bình kỳ 1 (Curricular units 1st sem grade):** Phản ánh năng lực tiếp thu và sự hòa nhập với môi trường đại học.\n"
            "4. **Tuổi khi nhập học (Age at enrollment):** Sinh viên lớn tuổi thường có nhiều áp lực cuộc sống và công việc làm thêm chi phối việc học.",
            ["gioi_thieu_do_an.txt"]
        )

    # 7. Mẫu tin nhắn hẹn gặp sinh viên nguy cơ cao
    if any(k in q for k in ["mẫu tin nhắn", "mau tin nhan", "hẹn gặp", "hen gap", "thư mời", "tin nhắn"]):
        return (
            "### ✉️ Mẫu Tin Nhắn Cố Vấn Học Tập Hẹn Gặp Sinh Viên Nguy Cơ Cao:\n\n"
            "> *\"Chào [Tên sinh viên], thầy/cô là cố vấn học tập của em. Thầy/cô nhận thấy trong học kỳ vừa qua em đang gặp một số thử thách trong việc hoàn thành các môn học và sắp xếp lộ trình học tập. Thầy/cô rất mong được gặp gỡ và trò chuyện cùng em tại văn phòng cố vấn vào [Thời gian] để cùng tìm giải pháp tháo gỡ khó khăn, giúp em an tâm tiếp tục việc học. Em phản hồi lại thầy/cô nhé!\"*",
            ["so_tay_co_van_hoc_tap.txt"]
        )

    # 8. Câu hỏi về bảo lưu kết quả / tạm dừng học
    if any(k in q for k in ["bảo lưu", "bao luu", "tạm dừng", "tam dung", "nghỉ học tạm thời"]):
        return (
            "### 📋 Quy Định Về Bảo Lưu Kết Quả & Nghỉ Học Tạm Thời:\n\n"
            "- **Đối tượng được phép:** Sinh viên có lý do chính đáng như ốm đau dài ngày (có xác nhận bệnh viện), hoàn cảnh gia đình đặc biệt khó khăn hoặc thi hành nghĩa vụ quân sự.\n"
            "- **Thời hạn bảo lưu:** Tối đa không quá 4 học kỳ chính trong toàn khóa học.\n"
            "- **Quyền lợi:** Toàn bộ số tín chỉ và điểm số các học phần đã đạt sẽ được giữ nguyên khi sinh viên quay trở lại học tập.",
            ["quy_che_canh_bao_hoc_vu.txt"]
        )

    return None, []


def get_smart_fallback_answer(question):
    """Trích xuất câu trả lời thông minh từ kho tài liệu khi Ollama ngoại tuyến hoặc chưa mở."""
    q_lower = question.lower()

    # 1. Câu hỏi về loại bỏ học kỳ 2 / data leakage
    if any(k in q_lower for k in ["học kỳ 2", "hoc ky 2", "loại bỏ", "loai bo", "leakage", "rò rỉ"]):
        return (
            "### 📌 Lý do đồ án loại bỏ các thuộc tính Học kỳ 2:\n\n"
            "1. **Mục tiêu cảnh báo sớm (Early Warning):** Mục đích cốt lõi của đồ án là phát hiện sinh viên có nguy cơ bỏ học ngay sau khi kết thúc Học kỳ 1 năm nhất để Cố vấn học tập và Nhà trường kịp thời can thiệp trước khi quá muộn. Nếu chờ đến khi có kết quả Học kỳ 2 thì sinh viên đã có thể bỏ học từ trước, làm giảm đáng kể giá trị thực tế của hệ thống cảnh báo.\n"
            "2. **Chống rò rỉ dữ liệu (Prevent Data Leakage):** Kết quả học tập ở kỳ 2 phản ánh trực tiếp hành vi sắp bỏ học (như không đăng ký môn, điểm thi bằng 0). Đưa thông tin kỳ 2 vào sẽ khiến mô hình 'học thuộc' dữ liệu tương lai thay vì học cách dự báo từ sớm.\n\n"
            "*(Lưu ý: Mô hình vẫn giữ lại đầy đủ 6 thuộc tính của Học kỳ 1 để đánh giá chính xác năng lực thích nghi ban đầu)*",
            ["gioi_thieu_do_an.txt"]
        )

    # 2. Câu hỏi về thuật toán Machine Learning
    if any(k in q_lower for k in ["thuật toán", "thuat toan", "mô hình", "mo hinh", "machine learning", "logistic"]):
        return (
            "### ⚙️ Thuật toán Machine Learning trong đồ án EduWarn AI:\n\n"
            "- **Mô hình triển khai chính thức:** **Logistic Regression (Hồi quy Logistic đa lớp)** kết hợp tiền xử lý `ColumnTransformer` (`OneHotEncoder` cho biến phân loại và `StandardScaler` cho biến số liên tục).\n"
            "- **Các thuật toán đối chứng đã thử nghiệm:** Random Forest, Support Vector Machine (SVM), Decision Tree, K-Nearest Neighbors (KNN).\n"
            "- **Lý do lựa chọn Logistic Regression:**\n"
            "  + **Tính giải thích cao (Explainability):** Dễ dàng diễn giải trọng số của từng yếu tố rủi ro cho Cố vấn học tập.\n"
            "  + **Xác suất phân lớp rõ ràng:** Xuất trực tiếp % xác suất của 3 nhóm (Dropout, Enrolled, Graduate).\n"
            "  + **Hiệu năng thực tế:** Đạt Accuracy ~77% và Macro F1-score ~0.70, tốc độ dự đoán tức thì và không bị overfitting.",
            ["gioi_thieu_do_an.txt"]
        )

    # 3. Câu hỏi về nợ học phí và xử lý tài chính (tránh nhầm với 'nợ môn')
    if any(k in q_lower for k in ["học phí", "hoc phi", "nợ học phí", "tiền học"]):
        return (
            "### 💰 Quy định và hướng dẫn xử lý khi sinh viên nợ học phí:\n\n"
            "1. **Chính sách xin gia hạn học phí:** Sinh viên gặp khó khăn kinh tế đột xuất có thể làm **Đơn xin gia hạn nộp học phí** gửi Phòng Kế hoạch - Tài chính và Cố vấn học tập trước hạn chót ít nhất 7 ngày. Sinh viên được chia đợt nộp hoặc gia hạn tối đa đến trước tuần thi kết thúc học phần.\n"
            "2. **Hỗ trợ tài chính & Vay vốn:** Nhà trường có Quỹ hỗ trợ sinh viên vượt khó, các gói học bổng doanh nghiệp và hỗ trợ thủ tục vay vốn ưu đãi từ Ngân hàng Chính sách Xã hội.\n"
            "3. **Hệ quả nếu nợ quá hạn không giải quyết:** Sinh viên bị tạm khóa đăng ký môn kỳ mới, không được cấp bảng điểm hoặc xét tốt nghiệp.",
            ["chinh_sach_hoc_phi_va_hoc_bong.txt", "so_tay_co_van_hoc_tap.txt"]
        )

    # 4. Câu hỏi về cảnh báo học vụ và buộc thôi học
    if any(k in q_lower for k in ["cảnh báo", "canh bao", "buộc thôi học", "buoc thoi hoc", "quy chế", "quy che"]):
        return (
            "### ⚠️ Quy định các mức Cảnh báo học vụ & Buộc thôi học:\n\n"
            "- **Cảnh báo Mức 1:** Điểm trung bình học kỳ (GPA) dưới 1.0/4 (hoặc dưới 10/20) hoặc tích lũy dưới 50% số tín chỉ đăng ký.\n"
            "- **Cảnh báo Mức 2:** Bị cảnh báo Mức 1 trong 2 học kỳ liên tiếp hoặc điểm tích lũy CPA dưới chuẩn năm học.\n"
            "- **Cảnh báo Mức 3 (Nguy cơ cực cao):** Bị giới hạn đăng ký tối đa 14 tín chỉ trong kỳ tiếp theo và bắt buộc phải tham gia các buổi tư vấn định kỳ cùng Cố vấn học tập.\n"
            "- **Buộc thôi học:** Bị cảnh báo Mức 3 trong 2 kỳ liên tiếp, hoặc điểm tích lũy cả năm không đạt chuẩn tối thiểu, hoặc vượt quá thời gian đào tạo tối đa.",
            ["quy_che_canh_bao_hoc_vu.txt"]
        )

    # 5. Câu hỏi về học bổng
    if any(k in q_lower for k in ["học bổng", "hoc bong", "khen thưởng"]):
        return (
            "### 🏆 Tiêu chuẩn xét Học bổng khuyến khích học tập:\n\n"
            "- **Điều kiện:** Đăng ký đủ tín chỉ tối thiểu theo quy định kỳ học, không có môn thi lại, điểm rèn luyện từ loại Tốt trở lên.\n"
            "- **Mức học bổng:**\n"
            "  + **Xuất sắc:** Điểm trung bình GPA >= 3.6/4 (hoặc >= 16/20).\n"
            "  + **Giỏi:** Điểm trung bình GPA >= 3.2/4 (hoặc >= 14/20).\n"
            "  + **Khá:** Điểm trung bình GPA >= 2.5/4 (hoặc >= 12/20).",
            ["chinh_sach_hoc_phi_va_hoc_bong.txt"]
        )

    # 6. Các yếu tố quan trọng ảnh hưởng bỏ học
    if any(k in q_lower for k in ["yếu tố", "yeu to", "ảnh hưởng", "anh huong", "quan trọng"]):
        return (
            "### 📊 Các yếu tố rủi ro ảnh hưởng lớn nhất đến nguy cơ bỏ học (Dropout):\n\n"
            "1. **Tình trạng nợ học phí & Đã hoàn thành học phí:** Yếu tố kinh tế gia đình đóng vai trò quyết định, sinh viên nợ học phí có tỷ lệ bỏ học cao hơn gấp 3 lần.\n"
            "2. **Số môn đạt / nợ ở kỳ 1:** Sinh viên nợ trên 2 môn kỳ 1 có xác suất bỏ học tăng vọt do mất động lực và áp lực học lại đè nặng.\n"
            "3. **Điểm trung bình kỳ 1:** Phản ánh trực tiếp khả năng thích nghi với phương pháp học tập đại học.\n"
            "4. **Độ tuổi khi nhập học:** Sinh viên nhập học lớn tuổi thường có gánh nặng việc làm, mưu sinh và gia đình nhiều hơn.",
            ["gioi_thieu_do_an.txt"]
        )

    # 7. Mẫu tin nhắn hẹn gặp sinh viên
    if any(k in q_lower for k in ["mẫu tin nhắn", "mau tin nhan", "hẹn gặp", "hen gap", "tin nhắn"]):
        return (
            "### ✉️ Mẫu tin nhắn Cố vấn học tập hẹn gặp sinh viên nguy cơ cao:\n\n"
            "> *\"Chào [Tên sinh viên], thầy/cô là cố vấn học tập của em. Thầy/cô nhận thấy trong học kỳ vừa qua em đang gặp một số thử thách trong việc hoàn thành các môn học và sắp xếp lộ trình học tập. Thầy/cô rất mong được gặp gỡ và trò chuyện cùng em tại văn phòng cố vấn vào [Thời gian] để cùng tìm giải pháp tháo gỡ khó khăn, giúp em an tâm tiếp tục việc học. Em phản hồi lại thầy/cô nhé!\"*",
            ["so_tay_co_van_hoc_tap.txt"]
        )

    # Nếu câu hỏi khác, tìm kiếm đoạn khớp nguyên từ trong kho tài liệu txt (tránh khớp nhầm câu ngoài phạm vi)
    import re
    from pathlib import Path
    doc_folder = Path(__file__).resolve().parent.parent / "knowledge_base" / "documents"
    best_text = ""
    best_score = 0
    matched_source = "gioi_thieu_do_an.txt"
    stopwords = {"ban", "bạn", "cho", "toi", "tôi", "biet", "biết", "cua", "của", "cac", "các", "nhung", "những", "nhu", "như", "the", "thế", "nao", "nào", "la", "là", "gi", "gì", "co", "có", "khong", "không", "ve", "về", "va", "và", "hay", "giup", "giúp", "duoc", "được", "nay", "này"}
    words = [w for w in re.findall(r"[^\W\d_]+", q_lower) if len(w) > 2 and w not in stopwords]
    if doc_folder.exists() and len(words) >= 2:
        for doc_file in doc_folder.glob("*.txt"):
            try:
                content = doc_file.read_text(encoding="utf-8")
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                for p in paragraphs:
                    p_low = p.lower()
                    score = sum(1 for w in words if re.search(rf"(?<!\w){re.escape(w)}(?!\w)", p_low))
                    if score > best_score or (score == best_score and score > 0 and len(p) > len(best_text)):
                        best_score = score
                        best_text = p
                        matched_source = doc_file.name
            except Exception:
                pass

    if best_text and best_score >= 3:
        return (best_text, [matched_source])

    return (
        "Xin lỗi, tôi là **Trợ lý AI Cố vấn Học vụ** của hệ thống EduWarn AI nên không có thông tin về chủ đề này. Tôi chỉ hỗ trợ giải đáp về **quy chế đào tạo, cảnh báo học vụ, học phí/học bổng, bảo lưu, chuyển ngành, phương pháp học tập, hồ sơ sinh viên và kiến thức đồ án EduWarn AI**.",
        []
    )


def _get_fallback_advice(prediction, probabilities, student_data):
    """Tạo tư vấn mẫu chuyên sâu tự động khi Ollama chưa chạy hoặc ngoại tuyến."""
    prob_str = ", ".join(f"{k}: {v:.1f}%" for k, v in probabilities.items())
    tuition_note = str(student_data.get("Nghĩa vụ học phí", ""))
    has_debt = "Đang có nợ" in tuition_note or student_data.get("Có khoản nợ học phí") == "Có" or student_data.get("Đã đóng học phí đúng hạn") == "Không"
    student_name = str(student_data.get("Họ và tên sinh viên", "")).strip()
    greeting = f"Chào {student_name}" if student_name else "Chào em"

    if prediction == "Graduate":
        return f"""### 📌 1. Đánh giá tổng quan hồ sơ
Sinh viên{f' **{student_name}**' if student_name else ''} có tiến độ học tập rất khả quan và ổn định ({prob_str}). Các chỉ số học tập kỳ 1 cho thấy sinh viên có năng lực tốt và khả năng cao sẽ tốt nghiệp đúng hạn.

### 🎯 2. Hướng dẫn can thiệp dành cho Cố vấn học tập
- **Về học tập:** Khích lệ sinh viên tiếp tục phát huy kết quả hiện tại; tư vấn đăng ký các học phần nâng cao hoặc tham gia nghiên cứu khoa học, hoạt động ngoại khóa để tích lũy thêm điểm rèn luyện. Nếu có môn chưa đạt điểm tối đa, có thể hướng dẫn học cải thiện vào kỳ hè.
- **Về tài chính & Đời sống:** Ghi nhận sinh viên hoàn thành tốt nghĩa vụ học tập và học phí; giới thiệu các chương trình học bổng khuyến khích tài năng.
- **Kế hoạch theo dõi:** Lên lịch gặp định kỳ vào giữa học kỳ hoặc khi đăng ký học phần chuyên ngành.

### ✉️ 3. Mẫu tin nhắn gửi sinh viên
> *"{greeting}, thầy/cô là cố vấn học tập của em. Kết quả học tập và rèn luyện của em đang duy trì rất tốt. Em hãy tiếp tục giữ vững phong độ này trong kỳ tới nhé. Nếu cần tư vấn thêm về đăng ký môn chuyên ngành hay cơ hội học bổng, em có thể liên hệ thầy/cô bất cứ lúc nào!"*
"""
    elif prediction == "Enrolled":
        return f"""### 📌 1. Đánh giá tổng quan hồ sơ
Sinh viên{f' **{student_name}**' if student_name else ''} hiện đang trong diện cần duy trì và cải thiện học tập ({prob_str}). Kết quả học tập ở mức trung bình, cần xây dựng kế hoạch phân bổ thời gian hợp lý hơn để bắt kịp lộ trình đào tạo chuẩn.

### 🎯 2. Hướng dẫn can thiệp dành cho Cố vấn học tập
- **Về học tập:** Rà soát lại các môn chưa đạt hoặc điểm thấp trong kỳ 1; hướng dẫn ưu tiên đăng ký học lại sớm; tham gia các nhóm trợ giảng hoặc nhóm học tập để cải thiện GPA.
- **Về tài chính & Đời sống:** Tìm hiểu xem sinh viên có gặp trở ngại về phương tiện đi lại, việc làm thêm hay tài chính hay không để có phương án hỗ trợ.
- **Kế hoạch theo dõi:** Đặt lịch hẹn cố vấn học vụ sau đợt thi giữa kỳ để đánh giá mức độ tiến bộ.

### ✉️ 3. Mẫu tin nhắn gửi sinh viên
> *"{greeting}, thầy/cô là cố vấn học tập của em. Thầy/cô muốn hẹn gặp em vào tuần tới để cùng rà soát tiến độ học tập và lên kế hoạch đăng ký môn cho kỳ tiếp theo nhằm đạt kết quả tốt nhất. Em xem sắp xếp thời gian phản hồi lại thầy/cô nhé!"*
"""
    else:  # Dropout
        financial_advice = "- **Về tài chính & Đời sống:** Hướng dẫn làm đơn xin gia hạn nộp học phí hoặc tiếp cận quỹ hỗ trợ sinh viên khó khăn." if has_debt else "- **Về tài chính & Đời sống:** Tình hình học phí hiện tại ổn định. Cần tập trung hỗ trợ động lực và tháo gỡ khó khăn về mặt phương pháp học tập."

        return f"""### 📌 1. Đánh giá tổng quan hồ sơ
Hệ thống ghi nhận hồ sơ sinh viên{f' **{student_name}**' if student_name else ''} thuộc nhóm có nguy cơ thôi học/bỏ học đáng chú ý ({prob_str}). Cần có sự đồng hành và can thiệp kịp thời từ phía Cố vấn học tập và Khoa.

### 🎯 2. Hướng dẫn can thiệp dành cho Cố vấn học tập
- **Về học tập:** Tìm hiểu cụ thể nguyên nhân vắng học hoặc sa sút học lực; tư vấn giảm bớt tải tín chỉ nếu quá sức; hướng dẫn lộ trình học lại từng bước hoặc thủ tục bảo lưu tạm thời nếu sinh viên gặp biến cố cá nhân.
{financial_advice}
- **Kế hoạch theo dõi:** Xếp lịch hẹn gặp trực tiếp trong tuần này; giữ liên lạc thường xuyên qua Zalo/Email và phối hợp cùng phụ huynh khi cần thiết.

### ✉️ 3. Mẫu tin nhắn gửi sinh viên
> *"{greeting}, thầy/cô là cố vấn học tập của em. Thầy/cô nhận thấy em đang gặp một số khó khăn trong quá trình học tập kỳ vừa qua. Thầy/cô rất muốn lắng nghe và hỗ trợ em tháo gỡ những trở ngại này. Em hãy liên hệ hoặc ghé văn phòng cố vấn để thầy trò mình cùng trao đổi nhé!"*
"""


def generate_advice(prediction, probabilities, student_data, contexts):
    """Sinh gợi ý hỗ trợ dựa trên kết quả mô hình, dữ liệu đã nhập và ngữ cảnh RAG."""

    probability_text = "\n".join(
        f"- {label}: {float(value):.2f}%"
        for label, value in probabilities.items()
    )
    student_text = "\n".join(
        f"- {key}: {value}" for key, value in student_data.items()
    ) or "Không có trường dữ liệu bổ sung được cung cấp."
    context_text = "\n\n".join(
        f"[Tài liệu {index + 1}]\n{context}"
        for index, context in enumerate(contexts)
    ) or "Không tìm thấy đoạn tài liệu liên quan."

    # Chỉ đạo đặc thù theo từng nhóm kết quả để tránh mô hình suy diễn sai lệch
    if prediction == "Graduate":
        category_instruction = """
QUY TẮC CỐT LÕI CHO HỒ SƠ GRADUATE (TỐT NGHIỆP):
1. Sinh viên có kết quả học tập KHẢ QUAN, đúng tiến độ tốt nghiệp (Graduate chiếm ưu thế lớn). Xác suất Dropout rất thấp.
2. TUYỆT ĐỐI KHÔNG ĐƯỢC NÓI sinh viên 'có nguy cơ bỏ học' hay nhắc đến việc 'xin gia hạn học phí' nếu sinh viên không nợ học phí!
3. Giữ thái độ biểu dương, khích lệ thành tích học tập tốt ở kỳ 1.
4. Hướng dẫn sinh viên đăng ký trả nợ hoặc cải thiện học phần chưa đạt ở kỳ hè/kỳ tiếp theo để nâng cao GPA, hướng tới tốt nghiệp loại Giỏi/Khá.
5. Mẫu tin nhắn gửi sinh viên: Thân mật, khen ngợi thành tích và hỏi thăm kế hoạch học tập kỳ tới một cách tích cực.
"""
    elif prediction == "Enrolled":
        category_instruction = """
QUY TẮC CỐT LÕI CHO HỒ SƠ ENROLLED (ĐANG THEO HỌC):
1. Sinh viên đang duy trì việc học ở mức trung bình.
2. Cần động viên sinh viên tập trung cải thiện điểm các môn còn yếu và tích lũy đủ tín chỉ để bắt kịp tiến độ tốt nghiệp.
3. Mẫu tin nhắn gửi sinh viên: Thân thiện, hẹn gặp trao đổi kế hoạch học tập kỳ tới.
"""
    else:
        category_instruction = """
QUY TẮC CỐT LÕI CHO HỒ SƠ DROPOUT (NGUY CƠ BỎ HỌC):
1. Sinh viên thuộc nhóm cần ưu tiên hỗ trợ đặc biệt do nguy cơ thôi học cao.
2. Phân tích đúng nguyên nhân thực tế có trong dữ liệu (nếu có nợ môn thì hướng dẫn trả nợ môn; nếu có nợ học phí thì hướng dẫn thủ tục gia hạn/học bổng; nếu không nợ học phí thì không được nhắc nợ học phí).
3. Mẫu tin nhắn hẹn gặp sinh viên: Thấu cảm, lắng nghe khó khăn, không phán xét, tạo sự an tâm.
"""

    system_prompt = f"""
Bạn là Chuyên gia Cố vấn Học vụ và Đào tạo Đại học tận tâm, giàu kinh nghiệm.
Dưới đây là kết quả phân tích hồ sơ học tập và tài chính của một sinh viên từ hệ thống cảnh báo học vụ.
Mục tiêu của bạn là đưa ra nhận định chính xác, khách quan, giàu tính nhân văn và thiết thực.

{category_instruction}

THÔNG TIN HỒ SƠ VÀ KẾT QUẢ DỰ ĐOÁN:
- Kết quả mô hình: {prediction}
- Tỷ lệ xác suất:
{probability_text}
- Một số chỉ số hồ sơ thực tế:
{student_text}

TÀI LIỆU QUY CHẾ VÀ CHÍNH SÁCH THAM KHẢO:
{context_text}

HƯỚNG DẪN CẤU TRÚC PHẢN HỒI:
### 📌 1. Đánh giá tổng quan hồ sơ
(Phân tích chi tiết, trung thực dựa trên ĐÚNG các số liệu thực tế của sinh viên: họ tên/MSSV nếu có, ngành học, tỷ lệ đạt môn, điểm TB thang 4.0/10, tình trạng học phí, hoàn cảnh ở trọ/gia đình và đối chiếu quy chế. Nếu học tốt thì khen ngợi, nếu có rủi ro mới cảnh báo; không bịa thông tin ngoài hồ sơ)

### 🎯 2. Hướng dẫn can thiệp dành cho Cố vấn học tập
- **Về học tập:** (Kế hoạch tích lũy tín chỉ, môn cần học lại hoặc học nâng cao dựa đúng vào số môn đạt/nợ và điểm TB hiện tại)
- **Về tài chính & Đời sống:** (Ghi nhận nếu học phí đã hoàn tất tốt, hoặc giải pháp nếu có nợ; lưu ý hoàn cảnh ở trọ/hỗ trợ đặc biệt nếu có)
- **Kế hoạch theo dõi:** (Mốc thời gian cố vấn trao đổi tiếp theo)

### ✉️ 3. Mẫu tin nhắn hẹn gặp sinh viên
> *"Chào {student_data.get('Họ và tên sinh viên') or '[Tên sinh viên]'}, thầy/cô là cố vấn học tập của em..."* (Đoạn tin nhắn chuẩn mực sư phạm, cá nhân hóa theo đúng tên và kết quả học tập thực tế của em)
"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Hãy phân tích hồ sơ này thật đầy đủ, chi tiết theo đúng số liệu của sinh viên và đưa ra gợi ý hỗ trợ phù hợp theo định dạng."}
                ],
                "stream": True,
                "think": False,
                "keep_alive": "20m",
                "options": {"temperature": 0.25, "num_predict": 2048, "num_ctx": 8192}
            },
            stream=True,
            timeout=(10, 240)
        )
        response.raise_for_status()

        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(data["error"])
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
        finally:
            response.close()
    except Exception:
        # Tự động chuyển sang phương án dự phòng chuẩn xác nếu Ollama chưa mở hoặc lỗi kết nối
        yield _get_fallback_advice(prediction, probabilities, student_data)
