"""Các hàm dùng chung cho dự đoán cá nhân của đồ án."""

from pathlib import Path
import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "student_dropout_model.pkl"


def load_prediction_model():
    """Tải preprocessor và Logistic Regression đã lưu."""
    data = joblib.load(MODEL_PATH)
    return data["preprocessor"], data["model"]


# ==================================================
# 2. TÊN CÁC TRƯỜNG BẰNG TIẾNG VIỆT
# ==================================================

labels = {
    "Marital Status": "Tình trạng hôn nhân",
    "Application mode": "Phương thức tuyển sinh",
    "Application order": "Thứ tự nguyện vọng",
    "Course": "Ngành học",
    "Daytime/evening attendance": "Hình thức học",
    "Previous qualification": "Trình độ học vấn trước đó",
    "Nacionality": "Quốc tịch",
    "Mother's qualification": "Trình độ học vấn của mẹ",
    "Father's qualification": "Trình độ học vấn của cha",
    "Mother's occupation": "Nghề nghiệp của mẹ",
    "Father's occupation": "Nghề nghiệp của cha",
    "Displaced": "Sinh viên thuộc diện di dời",
    "Educational special needs": "Nhu cầu giáo dục đặc biệt",
    "Debtor": "Có khoản nợ học phí",
    "Tuition fees up to date": "Đã đóng học phí đúng hạn",
    "Gender": "Giới tính",
    "Scholarship holder": "Có học bổng",
    "International": "Sinh viên quốc tế",

    "Previous qualification (grade)": "Điểm văn bằng trước đó",
    "Admission grade": "Điểm tuyển sinh",
    "Age at enrollment": "Tuổi khi nhập học",
    "Curricular units 1st sem (credited)": "Số học phần được công nhận",
    "Curricular units 1st sem (enrolled)": "Số học phần đăng ký",
    "Curricular units 1st sem (evaluations)": "Số học phần được đánh giá",
    "Curricular units 1st sem (approved)": "Số học phần đạt",
    "Curricular units 1st sem (grade)": "Điểm trung bình học kỳ 1",
    "Curricular units 1st sem (without evaluations)": "Số học phần không được đánh giá",
    "Unemployment rate": "Tỷ lệ thất nghiệp",
    "Inflation rate": "Tỷ lệ lạm phát",
    "GDP": "Tổng sản phẩm quốc nội (GDP)"
}


# ==================================================
# 3. GIẢI THÍCH MỘT SỐ MÃ PHÂN LOẠI
# ==================================================

code_descriptions = {
    "Marital Status": {
        1: "Độc thân",
        2: "Đã kết hôn",
        3: "Góa",
        4: "Đã ly hôn",
        5: "Sống chung không kết hôn",
        6: "Ly thân hợp pháp"
    },

    "Daytime/evening attendance": {
        1: "Học ban ngày",
        0: "Học buổi tối"
    },

    "Gender": {
        1: "Nam",
        0: "Nữ"
    },

    "Displaced": {
        1: "Có",
        0: "Không"
    },

    "Educational special needs": {
        1: "Có",
        0: "Không"
    },

    "Debtor": {
        1: "Có",
        0: "Không"
    },

    "Tuition fees up to date": {
        1: "Đã đóng đúng hạn",
        0: "Chưa đóng đúng hạn"
    },

    "Scholarship holder": {
        1: "Có học bổng",
        0: "Không có học bổng"
    },

    "International": {
        1: "Sinh viên quốc tế",
        0: "Không phải sinh viên quốc tế"
    },

    "Course": {
        33: "Công nghệ sản xuất nhiên liệu sinh học",
        171: "Thiết kế hoạt hình và đa phương tiện",
        8014: "Công tác xã hội (buổi tối)",
        9003: "Nông học",
        9070: "Thiết kế truyền thông",
        9085: "Điều dưỡng thú y",
        9119: "Kỹ thuật tin học",
        9130: "Chăn nuôi ngựa",
        9147: "Quản lý",
        9238: "Công tác xã hội",
        9254: "Du lịch",
        9500: "Điều dưỡng",
        9556: "Vệ sinh răng miệng",
        9670: "Quản lý quảng cáo và tiếp thị",
        9773: "Báo chí và truyền thông",
        9853: "Giáo dục cơ bản",
        9991: "Quản lý (buổi tối)"
    },

    "Previous qualification": {
        1: "Giáo dục trung học",
        2: "Giáo dục đại học - Cử nhân",
        3: "Giáo dục đại học - Bằng cấp",
        4: "Giáo dục đại học - Thạc sĩ",
        5: "Giáo dục đại học - Tiến sĩ",
        6: "Đã từng học đại học",
        9: "Chưa hoàn thành năm học thứ 12",
        10: "Chưa hoàn thành năm học thứ 11",
        12: "Khác - năm học thứ 11",
        14: "Năm học thứ 10",
        15: "Chưa hoàn thành năm học thứ 10",
        19: "Giáo dục cơ bản chu kỳ 3",
        38: "Giáo dục cơ bản chu kỳ 2",
        39: "Chứng chỉ chuyên môn công nghệ",
        40: "Giáo dục đại học - Cử nhân chu kỳ 1",
        42: "Khóa kỹ thuật chuyên nghiệp bậc cao",
        43: "Giáo dục đại học - Thạc sĩ chu kỳ 2"
    },

    "Nacionality": {
        1: "Bồ Đào Nha",
        2: "Đức",
        6: "Tây Ban Nha",
        11: "Ý",
        13: "Hà Lan",
        14: "Anh",
        17: "Litva",
        21: "Angola",
        22: "Cape Verde",
        24: "Guinea",
        25: "Mozambique",
        26: "São Tomé và Príncipe",
        32: "Thổ Nhĩ Kỳ",
        41: "Brazil",
        62: "Romania",
        100: "Moldova",
        101: "Mexico",
        103: "Ukraine",
        105: "Nga",
        108: "Cuba",
        109: "Colombia"
    }
}



# Bổ sung mô tả theo bảng thuộc tính chính thức của bộ dữ liệu UCI
code_descriptions.update({
    "Application mode": {
        1: "Đợt 1 - Diện tuyển sinh chung",
        2: "Theo Quy định số 612/93",
        5: "Đợt 1 - Diện đặc biệt (Azores)",
        7: "Người có bằng cấp thuộc chương trình đại học khác",
        10: "Theo Quy định số 854-B/99",
        15: "Sinh viên quốc tế (bậc cử nhân)",
        16: "Đợt 1 - Diện đặc biệt (Madeira)",
        17: "Đợt 2 - Diện tuyển sinh chung",
        18: "Đợt 3 - Diện tuyển sinh chung",
        26: "Theo Quy định 533-A/99 - Mục b2 (chương trình khác)",
        27: "Theo Quy định 533-A/99 - Mục b3 (cơ sở khác)",
        39: "Thí sinh trên 23 tuổi",
        42: "Chuyển trường",
        43: "Chuyển ngành học",
        44: "Có bằng chuyên môn công nghệ",
        51: "Chuyển trường hoặc chuyển ngành",
        53: "Có bằng khóa học chu kỳ ngắn",
        57: "Sinh viên quốc tế chuyển trường hoặc chuyển ngành"
    },
    "Application order": {
        0: "Nguyện vọng 1 - Ưu tiên cao nhất",
        1: "Nguyện vọng 2",
        2: "Nguyện vọng 3",
        3: "Nguyện vọng 4",
        4: "Nguyện vọng 5",
        5: "Nguyện vọng 6",
        6: "Nguyện vọng 7",
        7: "Nguyện vọng 8",
        8: "Nguyện vọng 9",
        9: "Nguyện vọng 10 - Ưu tiên thấp nhất"
    },
    "Mother's qualification": {
        1: "Trung học phổ thông - Lớp 12 hoặc tương đương",
        2: "Đại học - Cử nhân",
        3: "Giáo dục đại học - Bằng cấp",
        4: "Thạc sĩ",
        5: "Tiến sĩ",
        6: "Đã từng học đại học",
        9: "Chưa hoàn thành lớp 12",
        10: "Chưa hoàn thành lớp 11",
        11: "Lớp 7 (hệ giáo dục cũ)",
        12: "Khác - Lớp 11",
        14: "Lớp 10",
        18: "Khóa học thương mại tổng hợp",
        19: "Giáo dục cơ bản chu kỳ 3",
        22: "Khóa học kỹ thuật - chuyên môn",
        26: "Lớp 7",
        27: "Chu kỳ 2 chương trình trung học phổ thông",
        29: "Chưa hoàn thành lớp 9",
        30: "Lớp 8",
        34: "Không xác định",
        35: "Không biết đọc và viết",
        36: "Biết đọc nhưng chưa hoàn thành lớp 4",
        37: "Giáo dục cơ bản chu kỳ 1",
        38: "Giáo dục cơ bản chu kỳ 2",
        39: "Khóa chuyên môn công nghệ",
        40: "Giáo dục đại học - Cử nhân chu kỳ 1",
        41: "Khóa học chuyên sâu bậc cao",
        42: "Khóa kỹ thuật chuyên nghiệp bậc cao",
        43: "Thạc sĩ chu kỳ 2",
        44: "Tiến sĩ chu kỳ 3"
    },
    "Father's qualification": {
        1: "Trung học phổ thông - Lớp 12 hoặc tương đương",
        2: "Đại học - Cử nhân",
        3: "Giáo dục đại học - Bằng cấp",
        4: "Thạc sĩ",
        5: "Tiến sĩ",
        6: "Đã từng học đại học",
        9: "Chưa hoàn thành lớp 12",
        10: "Chưa hoàn thành lớp 11",
        11: "Lớp 7 (hệ giáo dục cũ)",
        12: "Khác - Lớp 11",
        13: "Năm thứ 2 trung học bổ túc",
        14: "Lớp 10",
        18: "Khóa học thương mại tổng hợp",
        19: "Giáo dục cơ bản chu kỳ 3",
        20: "Khóa học trung học bổ túc",
        22: "Khóa học kỹ thuật - chuyên môn",
        25: "Chưa hoàn thành khóa học trung học bổ túc",
        26: "Lớp 7",
        27: "Chu kỳ 2 chương trình trung học phổ thông",
        29: "Chưa hoàn thành lớp 9",
        30: "Lớp 8",
        31: "Khóa quản trị và thương mại tổng hợp",
        33: "Khóa kế toán và quản trị bổ túc",
        34: "Không xác định",
        35: "Không biết đọc và viết",
        36: "Biết đọc nhưng chưa hoàn thành lớp 4",
        37: "Giáo dục cơ bản chu kỳ 1",
        38: "Giáo dục cơ bản chu kỳ 2",
        39: "Khóa chuyên môn công nghệ",
        40: "Giáo dục đại học - Cử nhân chu kỳ 1",
        41: "Khóa học chuyên sâu bậc cao",
        42: "Khóa kỹ thuật chuyên nghiệp bậc cao",
        43: "Thạc sĩ chu kỳ 2",
        44: "Tiến sĩ chu kỳ 3"
    },
    "Mother's occupation": {
        0: "Sinh viên",
        1: "Lãnh đạo, quản lý và điều hành",
        2: "Chuyên gia hoạt động trí thức và khoa học",
        3: "Kỹ thuật viên và chuyên viên trình độ trung cấp",
        4: "Nhân viên hành chính",
        5: "Nhân viên dịch vụ cá nhân, an ninh và bán hàng",
        6: "Lao động nông nghiệp, thủy sản và lâm nghiệp",
        7: "Thợ thủ công, công nhân xây dựng và công nghiệp",
        8: "Nhân viên vận hành máy móc và lắp ráp",
        9: "Lao động phổ thông",
        10: "Lực lượng vũ trang",
        90: "Tình trạng nghề nghiệp khác",
        99: "Không có thông tin",
        122: "Chuyên gia y tế",
        123: "Giáo viên",
        125: "Chuyên gia công nghệ thông tin và truyền thông",
        131: "Kỹ thuật viên khoa học và kỹ thuật",
        132: "Kỹ thuật viên và chuyên viên y tế",
        134: "Kỹ thuật viên dịch vụ pháp lý, xã hội, thể thao và văn hóa",
        141: "Nhân viên văn phòng, thư ký và xử lý dữ liệu",
        143: "Nhân viên kế toán, thống kê và tài chính",
        144: "Nhân viên hỗ trợ hành chính khác",
        151: "Nhân viên dịch vụ cá nhân",
        152: "Nhân viên bán hàng",
        153: "Nhân viên chăm sóc cá nhân",
        171: "Thợ xây dựng lành nghề (trừ thợ điện)",
        173: "Thợ in, chế tác dụng cụ chính xác và thủ công mỹ nghệ",
        175: "Công nhân chế biến thực phẩm, gỗ, may mặc và sản xuất",
        191: "Nhân viên vệ sinh",
        192: "Lao động phổ thông trong nông nghiệp, chăn nuôi và thủy sản",
        193: "Lao động phổ thông trong khai khoáng, xây dựng, sản xuất và vận tải",
        194: "Nhân viên chuẩn bị bữa ăn"
    },
    "Father's occupation": {
        0: "Sinh viên",
        1: "Lãnh đạo, quản lý và điều hành",
        2: "Chuyên gia hoạt động trí thức và khoa học",
        3: "Kỹ thuật viên và chuyên viên trình độ trung cấp",
        4: "Nhân viên hành chính",
        5: "Nhân viên dịch vụ cá nhân, an ninh và bán hàng",
        6: "Lao động nông nghiệp, thủy sản và lâm nghiệp",
        7: "Thợ thủ công, công nhân xây dựng và công nghiệp",
        8: "Nhân viên vận hành máy móc và lắp ráp",
        9: "Lao động phổ thông",
        10: "Lực lượng vũ trang",
        90: "Tình trạng nghề nghiệp khác",
        99: "Không có thông tin",
        101: "Sĩ quan lực lượng vũ trang",
        102: "Hạ sĩ quan lực lượng vũ trang",
        103: "Nhân viên lực lượng vũ trang khác",
        112: "Giám đốc dịch vụ hành chính và thương mại",
        114: "Giám đốc khách sạn, nhà hàng, thương mại và dịch vụ",
        121: "Chuyên gia khoa học vật lý, toán học và kỹ thuật",
        122: "Chuyên gia y tế",
        123: "Giáo viên",
        124: "Chuyên gia tài chính, kế toán và quản trị",
        131: "Kỹ thuật viên khoa học và kỹ thuật",
        132: "Kỹ thuật viên và chuyên viên y tế",
        134: "Kỹ thuật viên dịch vụ pháp lý, xã hội, thể thao và văn hóa",
        135: "Kỹ thuật viên công nghệ thông tin và truyền thông",
        141: "Nhân viên văn phòng, thư ký và xử lý dữ liệu",
        143: "Nhân viên kế toán, thống kê và tài chính",
        144: "Nhân viên hỗ trợ hành chính khác",
        151: "Nhân viên dịch vụ cá nhân",
        152: "Nhân viên bán hàng",
        153: "Nhân viên chăm sóc cá nhân",
        154: "Nhân viên bảo vệ và an ninh",
        161: "Nông dân và lao động nông nghiệp có tay nghề",
        163: "Nông dân, người chăn nuôi, ngư dân và thợ săn tự cung tự cấp",
        171: "Thợ xây dựng lành nghề (trừ thợ điện)",
        172: "Thợ luyện kim và gia công kim loại",
        174: "Thợ điện và điện tử",
        175: "Công nhân chế biến thực phẩm, gỗ, may mặc và sản xuất",
        181: "Nhân viên vận hành nhà máy và máy móc cố định",
        182: "Công nhân lắp ráp",
        183: "Tài xế và nhân viên vận hành phương tiện",
        192: "Lao động phổ thông trong nông nghiệp, chăn nuôi và thủy sản",
        193: "Lao động phổ thông trong khai khoáng, xây dựng, sản xuất và vận tải",
        194: "Nhân viên chuẩn bị bữa ăn",
        195: "Người bán hàng rong và cung cấp dịch vụ đường phố"
    }
})

def display_category(feature, value):
    """Hiển thị tên mô tả và giữ lại mã gốc."""

    if value is None or pd.isna(value):
        return "Không rõ / Không có thông tin"

    try:
        key = int(value) if float(value).is_integer() else value
    except (ValueError, TypeError):
        key = value

    description = code_descriptions.get(feature, {}).get(key)

    if description:
        return description

    return f"{labels.get(feature, feature)} - mã {value}"


def get_category_options(feature, categorical_features, categories):
    """Lấy danh sách giá trị phân loại từ metadata của mô hình.

    Nhận categorical_features và categories từ trang gọi hàm, tránh phụ thuộc
    vào biến toàn cục không tồn tại trong module này.
    """
    if feature not in categorical_features:
        raise ValueError(f"'{feature}' không phải thuộc tính phân loại.")

    index = categorical_features.index(feature)
    options = categories[index].tolist()

    return [None] + options



def get_student_guidance(prediction, probabilities, values):
    """Tạo giải thích và gợi ý hành động dựa trên kết quả + dữ liệu đã nhập.

    Các dấu hiệu bên dưới là điểm cần kiểm tra thực tế, không phải nguyên nhân
    được mô hình chứng minh hay kết luận về cá nhân sinh viên.
    """
    concerns = []
    actions = []

    approved = values.get("Curricular units 1st sem (approved)")
    enrolled = values.get("Curricular units 1st sem (enrolled)")
    evaluations = values.get("Curricular units 1st sem (evaluations)")
    without_eval = values.get("Curricular units 1st sem (without evaluations)")
    grade = values.get("Curricular units 1st sem (grade)")
    debtor = values.get("Debtor")
    tuition_ok = values.get("Tuition fees up to date")

    # Chỉ nêu những thông tin quan sát được trực tiếp trong hồ sơ.
    if approved is not None and enrolled is not None and enrolled > 0:
        ratio = approved / enrolled
        if ratio < 0.5:
            concerns.append(f"Sinh viên mới đạt {approved:g}/{enrolled:g} học phần đã đăng ký. Đây là nội dung nên được cố vấn học tập kiểm tra.")
            actions.append("Rà soát từng học phần chưa đạt, xác định môn cần học lại hoặc học bổ sung và lập lịch học phù hợp.")
        elif ratio < 0.8:
            concerns.append(f"Sinh viên đạt {approved:g}/{enrolled:g} học phần đã đăng ký; vẫn còn một số học phần cần theo dõi.")
            actions.append("Ưu tiên hoàn thành các học phần còn thiếu và trao đổi với giảng viên nếu đang gặp khó khăn ở môn học cụ thể.")

    if grade is not None and grade < 10:
        concerns.append(f"Điểm trung bình học kỳ 1 đang ở mức {grade:g}/20 theo dữ liệu nhập.")
        actions.append("Trao đổi với giảng viên để xác định phần kiến thức còn yếu, tham gia giờ hỗ trợ học tập và theo dõi điểm ở kỳ tiếp theo.")

    if without_eval is not None and without_eval > 0:
        concerns.append(f"Có {without_eval:g} học phần chưa có kết quả đánh giá.")
        actions.append("Kiểm tra lý do chưa được đánh giá (vắng thi, thiếu bài, chưa hoàn tất yêu cầu...) và liên hệ phòng đào tạo/giảng viên để biết cách xử lý.")

    if debtor == 1 or tuition_ok == 0:
        concerns.append("Thông tin tài chính cho thấy có khoản nợ hoặc học phí chưa được cập nhật đúng hạn.")
        actions.append("Liên hệ bộ phận tài chính – công tác sinh viên để kiểm tra công nợ, hạn đóng và các chính sách gia hạn hoặc hỗ trợ học phí đang áp dụng.")

    if not concerns:
        concerns.append("Từ các trường dữ liệu học tập và tài chính đã nhập, hệ thống chưa phát hiện dấu hiệu cần lưu ý theo các tiêu chí kiểm tra đơn giản.")

    if prediction == "Dropout":
        title = "Cần ưu tiên trao đổi và hỗ trợ sớm"
        explanation = ("Mô hình xếp hồ sơ này vào nhóm Dropout. Đây là tín hiệu để nhà trường chủ động tìm hiểu tình hình, "
                       "không có nghĩa sinh viên chắc chắn sẽ bỏ học.")
        actions = actions + [
            "Cố vấn học tập nên trao đổi riêng với sinh viên để hiểu khó khăn thực tế về học tập, tài chính, thời gian hoặc hoàn cảnh cá nhân.",
            "Cùng sinh viên thống nhất một kế hoạch hỗ trợ ngắn hạn, có người phụ trách và thời điểm kiểm tra lại tiến độ."
        ]
    elif prediction == "Enrolled":
        title = "Đang theo học – nên theo dõi và cải thiện"
        explanation = ("Mô hình xếp hồ sơ vào nhóm Enrolled, tức nhóm sinh viên vẫn đang theo học trong dữ liệu. "
                       "Kết quả này không đồng nghĩa sinh viên đã đạt tiến độ học tập mong muốn.")
        actions = actions + [
            "Lập kế hoạch học kỳ tiếp theo theo số tín chỉ còn thiếu và khả năng học thực tế của sinh viên.",
            "Hẹn kiểm tra tiến độ giữa kỳ để kịp thời hỗ trợ nếu kết quả học tập hoặc việc tham gia học bị giảm."
        ]
    else:
        title = "Kết quả dự đoán tích cực"
        explanation = ("Mô hình xếp hồ sơ vào nhóm Graduate. Đây là kết quả tích cực theo mô hình, "
                       "nhưng vẫn cần đối chiếu với tình trạng hoàn thành chương trình thực tế.")
        actions = actions + [
            "Đối chiếu số tín chỉ, học phần bắt buộc và các điều kiện tốt nghiệp với dữ liệu chính thức của nhà trường.",
            "Nếu sinh viên vẫn còn học phần phải hoàn thành, tiếp tục theo dõi kế hoạch học tập cho đến khi đủ điều kiện tốt nghiệp."
        ]

    # Khử trùng lặp nhưng giữ nguyên thứ tự.
    actions = list(dict.fromkeys(actions))
    ordered_probs = sorted([float(x) for x in probabilities], reverse=True)
    close_result = len(ordered_probs) > 1 and (ordered_probs[0] - ordered_probs[1]) < 0.10
    return title, explanation, concerns, actions, close_result




def get_model_metadata(preprocessor):
    """Lấy danh sách thuộc tính và các giá trị phân loại từ preprocessor."""
    categorical_features = list(preprocessor.transformers_[0][2])
    numerical_features = list(preprocessor.transformers_[1][2])

    categorical_pipeline = preprocessor.named_transformers_["cat"]
    encoder = categorical_pipeline.named_steps["encoder"]
    categories = encoder.categories_

    return categorical_features, numerical_features, categories


def validate_student_data(values, labels):
    """Kiểm tra các trường số quan trọng trước khi dự đoán (cho phép None để SimpleImputer tự điền)."""
    errors = []

    age = values.get("Age at enrollment")
    if age is not None and not pd.isna(age) and age <= 0:
        errors.append("Tuổi nhập học phải lớn hơn 0.")

    prev_grade = values.get("Previous qualification (grade)")
    if prev_grade is not None and not pd.isna(prev_grade) and prev_grade < 0:
        errors.append("Điểm văn bằng trước đó không hợp lệ.")

    adm_grade = values.get("Admission grade")
    if adm_grade is not None and not pd.isna(adm_grade) and adm_grade < 0:
        errors.append("Điểm tuyển sinh không hợp lệ.")

    fields = [
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (without evaluations)",
        "Curricular units 1st sem (grade)",
    ]
    for feature in fields:
        value = values.get(feature)
        if value is not None and not pd.isna(value) and value < 0:
            errors.append(f"{labels.get(feature, feature)} không được âm.")

    enrolled = values.get("Curricular units 1st sem (enrolled)")
    approved = values.get("Curricular units 1st sem (approved)")
    if (
        enrolled is not None and approved is not None
        and not pd.isna(enrolled) and not pd.isna(approved)
        and approved > enrolled
    ):
        errors.append("Số môn đã qua (Approved) không thể lớn hơn Số môn đăng ký.")

    return errors


def predict_student(preprocessor, model, values):
    """Tiền xử lý một hồ sơ và trả về nhãn cùng xác suất dự đoán."""
    expected_columns = list(preprocessor.feature_names_in_)
    missing_columns = [col for col in expected_columns if col not in values]
    if missing_columns:
        raise ValueError("Thiếu trường dữ liệu đầu vào: " + ", ".join(missing_columns))

    input_data = pd.DataFrame(
        [[values[col] for col in expected_columns]],
        columns=expected_columns
    )
    input_data = input_data.replace({None: float("nan")})

    transformed = preprocessor.transform(input_data)
    prediction = model.predict(transformed)[0]
    probabilities = model.predict_proba(transformed)[0]

    if len(model.classes_) != len(probabilities):
        raise ValueError("Số nhóm kết quả và số xác suất không khớp với file mô hình.")

    return str(prediction), [float(value) for value in probabilities]


def explain_prediction(preprocessor, model, values, target_class=None, top_n=5):
    """Giải thích Top yếu tố tác động mạnh nhất theo đúng lớp được dự đoán cho hồ sơ này."""
    if not hasattr(model, "coef_"):
        return []

    expected_columns = list(preprocessor.feature_names_in_)
    input_data = pd.DataFrame(
        [[values.get(col) for col in expected_columns]],
        columns=expected_columns,
    ).replace({None: float("nan")})

    transformed = preprocessor.transform(input_data)[0]
    feat_names = preprocessor.get_feature_names_out()
    classes = list(model.classes_)
    cls_name = target_class if target_class in classes else ("Dropout" if "Dropout" in classes else classes[0])
    cls_idx = classes.index(cls_name)
    coefs = model.coef_[cls_idx]

    class_vi = {
        "Dropout": "Bỏ học",
        "Enrolled": "Đang học",
        "Graduate": "Tốt nghiệp",
    }.get(cls_name, cls_name)

    grouped = {}
    for fname, x_val, w_val in zip(feat_names, transformed, coefs):
        contrib = float(x_val * w_val)
        if abs(contrib) < 1e-4:
            continue
        raw = fname.split("__", 1)[1] if "__" in fname else fname
        base_col = next((col for col in expected_columns if raw == col or raw.startswith(col + "_")), raw)
        grouped[base_col] = grouped.get(base_col, 0.0) + contrib

    # Bỏ qua 3 biến vĩ mô ẩn để phần giải thích chỉ tập trung vào yếu tố của chính sinh viên
    macro_cols = {"Unemployment rate", "Inflation rate", "GDP"}
    ranked = sorted(
        [(c, s) for c, s in grouped.items() if c not in macro_cols],
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:top_n]

    explanations = []
    for col, score in ranked:
        raw_val = values.get(col)
        if raw_val is None or (isinstance(raw_val, float) and pd.isna(raw_val)):
            val_text = "Chưa rõ"
        elif col in code_descriptions:
            val_text = display_category(col, raw_val)
        else:
            val_text = f"{raw_val:g}"

        supports = score > 0
        explanations.append({
            "feature": labels.get(col, col),
            "value_text": val_text,
            "score": round(score, 3),
            "target_class": cls_name,
            "supports_prediction": supports,
            "impact_text": f"Đẩy về nhóm {class_vi}" if supports else f"Kéo khỏi nhóm {class_vi}",
        })
    return explanations


def build_student_profile_dict(values, student_name="", student_id="", student_class="", explanations=None):
    """Chuyển hồ sơ 30 trường sang bản tóm tắt tiếng Việt đầy đủ, chi tiết cho AI phân tích."""
    profile = {}
    if student_name:
        profile["Họ và tên sinh viên"] = student_name
    if student_id:
        profile["Mã sinh viên (MSSV)"] = str(student_id)
    if student_class:
        profile["Lớp / Khoa / Ngành hiển thị"] = student_class

    def _fmt(col):
        v = values.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if col in code_descriptions:
            return display_category(col, v)
        return f"{v:g}" if isinstance(v, (int, float)) else str(v)

    # 1. Thông tin cá nhân & Tuyển sinh
    for col in ["Gender", "Age at enrollment", "Course", "Daytime/evening attendance", "Application mode", "Application order", "Previous qualification"]:
        val = _fmt(col)
        if val is not None:
            profile[labels.get(col, col)] = val

    adm = values.get("Admission grade")
    if adm is not None and not pd.isna(adm):
        profile["Điểm trúng tuyển Đại học"] = f"{float(adm)/200*30:.2f}/30 ({float(adm):.1f}/200)"
    prev_g = values.get("Previous qualification (grade)")
    if prev_g is not None and not pd.isna(prev_g):
        profile["Điểm học bạ / Tốt nghiệp THPT"] = f"{float(prev_g)/20:.2f}/10 ({float(prev_g):.1f}/200)"

    # 2. Kết quả Học kỳ 1 & đối chiếu Quy chế
    enrolled = values.get("Curricular units 1st sem (enrolled)")
    approved = values.get("Curricular units 1st sem (approved)")
    grade = values.get("Curricular units 1st sem (grade)")
    evals = values.get("Curricular units 1st sem (evaluations)")
    credited = values.get("Curricular units 1st sem (credited)")
    no_eval = values.get("Curricular units 1st sem (without evaluations)")

    if enrolled is not None and not pd.isna(enrolled) and approved is not None and not pd.isna(approved):
        enr_i, app_i = int(enrolled), int(approved)
        pct = (app_i / enr_i * 100) if enr_i > 0 else 0
        failed_i = max(0, enr_i - app_i)
        profile["Tiến độ tín chỉ Học kỳ 1"] = f"Đạt {app_i}/{enr_i} môn ({pct:.0f}%), chưa đạt/nợ {failed_i} môn"
    elif enrolled is not None and not pd.isna(enrolled):
        profile["Số học phần đăng ký HK1"] = str(int(enrolled))

    if grade is not None and not pd.isna(grade):
        g20 = float(grade)
        profile["Điểm trung bình Học kỳ 1"] = f"{g20/5:.2f}/4.0 (≈ {g20/2:.2f}/10 | {g20:.1f}/20)"
    if evals is not None and not pd.isna(evals):
        profile["Số lượt dự thi / đánh giá HK1"] = str(int(evals))
    if credited is not None and not pd.isna(credited) and int(credited) > 0:
        profile["Số môn được miễn / công nhận"] = str(int(credited))
    if no_eval is not None and not pd.isna(no_eval) and int(no_eval) > 0:
        profile["Số môn bỏ thi / không có điểm đánh giá"] = str(int(no_eval))

    # 3. Tài chính & Học bổng
    debtor = values.get("Debtor")
    tuition = values.get("Tuition fees up to date")
    if debtor is None and tuition is None:
        profile["Nghĩa vụ học phí"] = "Chưa rõ thông tin học phí"
    elif debtor == 1 or tuition == 0:
        profile["Nghĩa vụ học phí"] = "Đang có nợ học phí hoặc quá hạn đóng"
    else:
        profile["Nghĩa vụ học phí"] = "Đã đóng đầy đủ đúng hạn, KHÔNG có nợ"

    sch = _fmt("Scholarship holder")
    if sch is not None:
        profile["Học bổng"] = sch

    # Đối chiếu quy chế TT 08/2021
    fail_high = (enrolled is not None and approved is not None and not pd.isna(enrolled) and not pd.isna(approved) and float(enrolled) > 0 and float(approved) / float(enrolled) < 0.5)
    low_gpa = (grade is not None and not pd.isna(grade) and float(grade) < 10.0)
    if fail_high or low_gpa:
        profile["Đối chiếu Quy chế (TT 08/2021)"] = "Thuộc diện Cảnh báo học vụ Mức 1 (đạt dưới 50% số môn hoặc GPA dưới 2.0/4.0)"
    elif grade is not None and not pd.isna(grade) and float(grade) >= 16.0 and enrolled and approved and float(approved) >= float(enrolled) and debtor != 1 and tuition != 0:
        profile["Đối chiếu Quy chế (TT 08/2021)"] = "Đủ điều kiện xét Học bổng Khuyến khích học tập"
    else:
        profile["Đối chiếu Quy chế (TT 08/2021)"] = "Chưa thuộc diện cảnh báo học vụ"

    # 4. Hoàn cảnh gia đình & Đời sống
    for col in ["Displaced", "Marital Status", "Educational special needs", "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]:
        val = _fmt(col)
        if val is not None:
            profile[labels.get(col, col)] = val

    # 5. Top yếu tố ảnh hưởng từ mô hình ML
    if explanations:
        top_factors = [
            f"{item.get('feature')} ({item.get('value_text', '')}): {item.get('impact_text', '')}"
            for item in explanations[:5]
        ]
        profile["Top yếu tố ảnh hưởng mạnh nhất theo mô hình AI"] = "; ".join(top_factors)

    return profile


def create_history_item(model, prediction, probabilities):
    """Tạo một dòng lịch sử dự đoán."""
    from datetime import datetime

    probability_map = {
        str(label): float(probabilities[index]) * 100
        for index, label in enumerate(model.classes_)
    }

    return {
        "Thời gian": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "Kết quả dự đoán": prediction,
        "Dropout (%)": round(probability_map.get("Dropout", 0), 2),
        "Enrolled (%)": round(probability_map.get("Enrolled", 0), 2),
        "Graduate (%)": round(probability_map.get("Graduate", 0), 2),
    }
