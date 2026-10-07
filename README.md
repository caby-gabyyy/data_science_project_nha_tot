# GUI Nhà Tốt — Định giá nhà & Phát hiện giá bất thường

Ứng dụng Streamlit cho project 1 (đồ án tốt nghiệp Data Science, TTTH ĐH KHTN), bố cục kiểu trang bất động sản:
danh sách nhà dạng lưới, trang chi tiết nhà (giá ước tính AI, đánh giá giá đăng, lịch sử giá khu vực, nhà tương tự,
ước tính trả góp), công cụ định giá và kiểm tra giá hàng loạt.

## Chạy trên máy

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cấu trúc

| File | Vai trò |
|---|---|
| `app.py` | Trang chủ · Tìm nhà · Chi tiết nhà · Định giá nhà · Kiểm tra giá hàng loạt · Dữ liệu & mô hình · Nhóm |
| `ui.py` | Thành phần HTML/CSS: thẻ nhà, khối giá, thanh so sánh giá, biểu đồ lịch sử giá, trả góp |
| `nhatot_core.py` | Hàm parse/clean, FEATURES, pipeline, bộ chấm điểm bất thường — chép từ notebook project_1 |
| `train_models.py` | Train lại XGBoost + AnomalyScorer từ `data/df_clean.parquet` → `models/nhatot_bundle.joblib` |
| `data/` | `df_clean.parquet`, `df_scored.parquet`, bảng so sánh, 2 file CSV mẫu để upload |
| `images/` | Biểu đồ EDA / model / anomaly xuất từ notebook |
| `.streamlit/config.toml` | Màu giao diện |

Nếu server không nạp được `models/nhatot_bundle.joblib` (lệch phiên bản thư viện), app tự chạy `train_models.py` (~20 giây) rồi nạp lại.
Dữ liệu không có ảnh nhà → app dùng hình minh hoạ.

## Deploy Streamlit Cloud

1. Tạo repo GitHub (public), upload **toàn bộ nội dung thư mục này**, kể cả thư mục `.streamlit` (không có file > 100MB).
2. share.streamlit.io → Create app → Deploy a public app from GitHub → chọn repo, branch `main`, main file `app.py`.
3. Advanced settings → Python **3.12** → Deploy.
