import ast
import os
from html import escape

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import nhatot_core as core
import csv_io
import ui

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(HERE, *a)

st.set_page_config(page_title="Nhà Tốt — Định giá & phát hiện giá bất thường", page_icon="🏠", layout="wide")
ui.inject_css()

TEAM = [
    {"ten": "Phạm Võ Khánh Thư", "email": "phamvokhanhthu@gmail.com",
     "viec": "EDA, mô hình scikit-learn, phát hiện bất thường"},
    {"ten": "Chế Quang Dương", "email": "duongdino8x@gmail.com",
     "viec": "Pipeline dữ liệu, mô hình PySpark, bảng so sánh, báo cáo"},
]


# ======================================================================
# Nạp model + dữ liệu (cache 1 lần cho mọi người dùng)
# ======================================================================
@st.cache_resource(show_spinner="Đang nạp model…")
def load_bundle():
    path = P("models", "nhatot_bundle.joblib")
    try:
        return joblib.load(path)
    except Exception:
        # Lệch phiên bản thư viện trên server -> train lại từ df_clean.parquet (~20s)
        import train_models
        train_models.main()
        return joblib.load(path)


@st.cache_data
def load_scored():
    return pd.read_parquet(P("data", "df_scored.parquet"))


B = load_bundle()
MODEL, SCORER, CH = B["model"], B["scorer"], B["choices"]
SC = load_scored()


@st.cache_resource(show_spinner=False)
def load_lexicon():
    """Từ điển âm tiết tiếng Việt (tiêu đề, mô tả, địa chỉ, loại hình…) để khôi phục chữ bị mất dấu trong file upload."""
    cols = ["tieu_de", "mo_ta", "dia_chi", "loai_hinh", "giay_to_phap_ly", "huong_cua_chinh",
            "tinh_trang_noi_that", "dac_diem"]
    return csv_io.build_lexicon(pd.concat([SC[c] for c in cols]).fillna(""))
QL = lambda q: core.QUAN_LABEL.get(q, q) if isinstance(q, str) else ""
RQ = B["resid_q"]          # (cũ) phân vị P10/P90 sai số tuyệt đối trên tập test
# Khoảng 80% theo TỈ LỆ giá thật / giá dự đoán (dự đoán out-of-fold của toàn bộ dữ liệu). Sai số tuyệt đối cố định
# phủ 96% nhà <5 tỷ nhưng chỉ 32% nhà >12 tỷ; dùng tỉ lệ thì đều hơn (83% / 84% / 76% / 65%), kiểm bằng chia đôi dữ liệu.
_ratio = (SC["gia_ban_ty"] / SC["y_pred"]).clip(lower=1e-3)
R10, R90 = float(_ratio.quantile(0.10)), float(_ratio.quantile(0.90))
FEAT_LABEL = {"loai_hinh": "Loại hình", "phuong": "Phường", "quan": "Quận", "dien_tich_m2": "Diện tích",
              "giay_to_phap_ly": "Pháp lý", "so_phong_ngu_n": "Số phòng ngủ", "so_wc_n": "Số WC",
              "tong_so_tang": "Số tầng", "chieu_ngang_m": "Chiều ngang", "chieu_dai_m": "Chiều dài",
              "bdg_mean": "Đơn giá khu vực (TB)", "bdg_std": "Biến động giá khu vực",
              "bdg_slope": "Xu hướng giá khu vực", "mo_ta_len": "Độ dài mô tả",
              "co_mat_tien": "Từ khoá mặt tiền", "co_hem_xh": "Từ khoá hẻm xe hơi"}
IMPORTANCE = B["importance"].rename(lambda f: FEAT_LABEL.get(f, f)).rename("mức quan trọng")

MENU = ["🏠 Trang chủ", "🔎 Tìm nhà", "🏡 Chi tiết nhà", "💰 Định giá nhà",
        "🚨 Kiểm tra giá hàng loạt", "📊 Dữ liệu & mô hình", "👥 Nhóm thực hiện"]
ss = st.session_state
ss.setdefault("nav", MENU[0])
ss.setdefault("listing", int(SC.index[0]))


def go(page, listing=None):
    ss.nav = page
    if listing is not None:
        ss.listing = int(listing)
        ss._toast = f"Đã mở: {SC.at[int(listing), 'tieu_de'][:70]}"


def predict(df):
    df = core.fill_bdg_from_area(df, B["bdg_table"])
    return df, np.clip(MODEL.predict(df[core.FEATURES]), 0.05, None)


def est_range(pred):
    return max(pred * R10, 0.1), pred * R90


def weights_widget(key):
    """Trọng số composite + top-k%, mặc định giống notebook."""
    with st.expander("⚙️ Tuỳ chỉnh trọng số composite & ngưỡng top-k%"):
        c = st.columns(5)
        w = {"resid":  c[0].slider("Residual-z", 0.0, 1.0, 0.40, 0.05, key=f"{key}_w1"),
             "minmax": c[1].slider("Min/Max", 0.0, 1.0, 0.20, 0.05, key=f"{key}_w2"),
             "pct":    c[2].slider("P10–P90", 0.0, 1.0, 0.20, 0.05, key=f"{key}_w3"),
             "ml":     c[3].slider("Isolation Forest", 0.0, 1.0, 0.20, 0.05, key=f"{key}_w4")}
        k = c[4].slider("Top-k%", 1, 20, 5, key=f"{key}_k") / 100
        s = sum(w.values())
        if s == 0:
            st.warning("Tổng trọng số = 0 → dùng lại mặc định.")
            return core.W_DEFAULT, k
        if abs(s - 1) > 1e-9:
            st.caption(f"Tổng trọng số = {s:.2f} → tự chuẩn hoá về 1.")
        return {kk: v / s for kk, v in w.items()}, k


SIGNALS = [("s_resid", "Residual-z", "giá lệch xa giá mô hình dự đoán"),
           ("s_minmax", "Min/Max khu vực", "đơn giá/m² vượt sàn/trần khu vực"),
           ("s_pct", "P10–P90 đơn giá", "đơn giá/m² nằm ngoài khoảng P10–P90"),
           ("s_ml", "Isolation Forest", "tổ hợp giá – diện tích – số phòng lạ")]


def price_check(r, thr):
    """Khối 'Đánh giá giá đăng': thanh so sánh + kết luận + 4 tín hiệu."""
    lo, hi = est_range(r["y_pred"])
    lech = r["gia_ban_ty"] / r["y_pred"] - 1
    if r["flag_bat_thuong"] == 1:
        verdict = f'<span class="z-badge z-bad">⚠ BẤT THƯỜNG — {escape(r["huong"])}</span>'
    elif lo <= r["gia_ban_ty"] <= hi:
        verdict = '<span class="z-badge z-good">✓ Nằm trong khoảng ước tính</span>'
    else:
        verdict = '<span class="z-badge z-hi">Lệch khoảng ước tính nhưng chưa tới ngưỡng bất thường</span>'
    sig = "".join(
        f'<div><span>{lab}</span><b style="color:{"#d1242f" if r[k] >= .67 else "#8a5a00" if r[k] >= .34 else "#11734b"}">'
        f'{r[k]:.2f}</b></div>' for k, lab, _ in SIGNALS)
    why = [txt for k, _, txt in SIGNALS if (r[k] >= 1 if k == "s_minmax" else r[k] >= 0.5)]
    return (f'{verdict}<div class="z-txt" style="margin-top:8px">Giá đăng <b>{ui.ty(r["gia_ban_ty"])}</b> '
            f'{"cao" if lech > 0 else "thấp"} hơn giá ước tính <b>{abs(lech):.0%}</b>. '
            + (("Tín hiệu vượt ngưỡng: " + "; ".join(why) + ".") if why else "Không tín hiệu nào vượt ngưỡng.")
            + "</div>" + ui.range_bar(lo, hi, r["y_pred"], r["gia_ban_ty"])
            + f'<div class="z-dl">{sig}<div><span>Composite score</span><b>{r["total_score"]:.3f}</b></div>'
            f'<div><span>Ngưỡng gắn cờ (mốc top 5% composite)</span><b>{thr:.3f}</b></div></div>')


def grid(rows, key, ncol=3):
    """Lưới thẻ nhà. key != None -> cả thẻ bấm được: 1 nút trong suốt phủ lên thẻ (CSS st-key-open_*)."""
    cols = st.columns(ncol)
    for n, (i, r) in enumerate(rows.iterrows()):
        with cols[n % ncol], st.container(key=f"lc_{key}_{i}" if key else None):
            ui.listing_card(r, int(i), QL(r.get("quan")))
            if key:
                st.button(f"Xem chi tiết: {r['tieu_de'][:60]}", key=f"open_{key}_{i}",
                          on_click=go, args=(MENU[2], i))


# ======================================================================
# Sidebar
# ======================================================================
st.sidebar.image(P("images", "nhatot.jpg"), width="stretch")
st.sidebar.radio("Menu", MENU, key="nav", label_visibility="collapsed")
st.sidebar.divider()
ui.team_sidebar(TEAM)
st.sidebar.caption("Đồ án tốt nghiệp Data Science — TTTH ĐH KHTN")
st.sidebar.markdown('<div class="z-note">Sản phẩm học tập, dùng dữ liệu tin đăng Nhà Tốt cho mục đích nghiên cứu — '
                    'không phải website chính thức của Nhà Tốt / Chợ Tốt.</div>', unsafe_allow_html=True)
choice = ss.nav
THR = SCORER.threshold()

# ----------------------------------------------------------------------
if choice == MENU[0]:
    ui.html('<div class="z-hero"><h1>Định giá nhà. Phát hiện giá bất thường.</h1>'
            '<p>7.926 tin rao bán nhà riêng tại Bình Thạnh · Gò Vấp · Phú Nhuận (TP.HCM) — '
            'mô hình XGBoost ước tính giá, 4 tín hiệu thống kê & ML gắn cờ tin quá rẻ / quá đắt.</p></div>')
    m = B["metrics_test"]
    ui.kpis([("Tin dùng để học", ui.vn(B["n_train"], 0)), ("Sai số tuyệt đối TB (MAE)", ui.ty(m["MAE"])),
             ("R² trên tập test", ui.vn(m["R2"], 3)),
             ("Tin bị gắn cờ", f"{int(SC.flag_bat_thuong.sum())} ({SC.flag_bat_thuong.mean():.0%})")])
    c = st.columns(3)
    for col, (icon, title, txt, page) in zip(c, [
            ("🔎", "Tìm nhà", "Lọc tin theo quận, phường, giá, số phòng — mỗi tin có nhãn giá tốt / bất thường.", MENU[1]),
            ("💰", "Định giá nhà", "Nhập thông tin căn nhà → giá ước tính AI, khoảng tham khảo và kiểm tra giá bạn định đăng.", MENU[3]),
            ("🚨", "Kiểm tra hàng loạt", "Upload file CSV tin đăng → chấm 4 tín hiệu bất thường cho từng tin.", MENU[4])]):
        with col, st.container(border=True, key=f"card_home_{title}"):
            ui.html(f'<div style="font-size:2rem">{icon}</div><div class="z-h2" style="margin-top:4px">{title}</div>'
                    f'<div class="z-txt">{txt}</div>')
            st.button(f"Mở {title.lower()} →", key=f"home_{page}", on_click=go, args=(page,), type="primary")
    ui.html('<div class="z-h2">Tin nghi vấn nhất</div>')
    grid(SC[SC.flag_bat_thuong == 1].sort_values("total_score", ascending=False).head(3), "home")

# ----------------------------------------------------------------------
elif choice == MENU[1]:
    f = st.columns([1.1, 1.1, 1.3, 1.6, 0.9, 1.1])
    quan = f[0].selectbox("Quận", ["Tất cả"] + CH["quan"], format_func=lambda q: "Tất cả quận" if q == "Tất cả" else QL(q))
    phuong = f[1].selectbox("Phường", ["Tất cả"] + (CH["phuong_by_quan"][quan] if quan != "Tất cả" else []),
                            disabled=quan == "Tất cả")
    loai = f[2].selectbox("Loại hình", ["Tất cả"] + CH["loai_hinh"])
    gia = f[3].slider("Giá (tỷ)", 0.0, 50.0, (0.0, 50.0), 0.5)
    pn = f[4].selectbox("PN tối thiểu", [0, 1, 2, 3, 4, 5], format_func=lambda x: "Bất kỳ" if x == 0 else f"{x}+")
    trang_thai = f[5].selectbox("Trạng thái giá", ["Tất cả", "Giá tốt", "Bất thường"])
    v = SC
    if quan != "Tất cả":
        v = v[v.quan == quan]
    if phuong != "Tất cả":
        v = v[v.phuong == phuong]
    if loai != "Tất cả":
        v = v[v.loai_hinh == loai]
    hi_cap = gia[1] if gia[1] < 50 else np.inf
    v = v[v.gia_ban_ty.between(gia[0], hi_cap)]
    if pn:
        v = v[v.so_phong_ngu_n >= pn]
    if trang_thai == "Giá tốt":
        v = v[(v.flag_bat_thuong == 0) & v.gia_ban_ty.between(v.y_pred * (1 - ui.SUSPECT), v.y_pred * 0.9)]
    elif trang_thai == "Bất thường":
        v = v[v.flag_bat_thuong == 1]

    h1, h2 = st.columns([3, 1])
    title = "Nhà riêng " + (f"{phuong}, " if phuong != "Tất cả" else "") + (f"Quận {QL(quan)}" if quan != "Tất cả" else "TP.HCM")
    h1.markdown(f'<div class="z-h2" style="margin-top:4px">{escape(title)}</div>'
                f'<div class="z-meta">{ui.vn(len(v), 0)} tin · giá trung vị '
                f'{ui.ty(v.gia_ban_ty.median()) if len(v) else "—"}</div>', unsafe_allow_html=True)
    sort = h2.selectbox("Sắp xếp", ["Giá tốt nhất (so với ước tính)", "Giá thấp → cao", "Giá cao → thấp",
                                    "Diện tích lớn nhất", "Nghi vấn nhất"])
    # "Giá tốt nhất": rẻ nhất so với giá ước tính, nhưng đẩy tin nghi vấn (bị gắn cờ / lệch ≥ 35%) xuống cuối
    key, asc = {"Giá tốt nhất (so với ước tính)": (["suspect", "ratio"], [True, True]),
                "Giá thấp → cao": ("gia_ban_ty", True), "Giá cao → thấp": ("gia_ban_ty", False),
                "Diện tích lớn nhất": ("dien_tich_m2", False), "Nghi vấn nhất": ("total_score", False)}[sort]
    ratio = v.gia_ban_ty / v.y_pred
    v = v.assign(ratio=ratio, suspect=(v.flag_bat_thuong == 1) | ((ratio - 1).abs() >= ui.SUSPECT)
                 ).sort_values(key, ascending=asc)

    PER = 12
    n_page = max(1, int(np.ceil(len(v) / PER)))
    pkey = f"pg_{quan}_{phuong}_{loai}_{gia}_{pn}_{trang_thai}_{sort}"
    page = ss.get(pkey, 1)
    if len(v):
        grid(v.iloc[(page - 1) * PER: page * PER], "s")
    else:
        st.info("Không có tin phù hợp bộ lọc.")
    if n_page > 1:
        st.number_input(f"Trang (1–{n_page})", 1, n_page, key=pkey)

# ----------------------------------------------------------------------
elif choice == MENU[2]:
    top = st.columns([1, 4])
    top[0].button("← Danh sách", on_click=go, args=(MENU[1],))
    opts = SC.index.tolist()
    sel = top[1].selectbox("Chọn tin", opts, index=opts.index(ss.listing), label_visibility="collapsed",
                           format_func=lambda i: f"{SC.at[i, 'tieu_de'][:90]} — {ui.ty(SC.at[i, 'gia_ban_ty'])}")
    ss.listing = int(sel)
    r = SC.loc[ss.listing]
    i = ss.listing
    try:
        hist = [float(x) for x in ast.literal_eval(r["bieu_do_gia"])]
    except Exception:
        hist = []

    # Dải ảnh: ảnh minh hoạ theo loại hình | Google Maps + biểu đồ lịch sử đơn giá khu vực
    g1, g2 = st.columns([2, 1], gap="small")
    with g1:
        ui.html(f'<div class="z-photo" style="height:404px;background-image:url({ui.house_img(r["loai_hinh"], seed=i)})">'
                f'<span class="cap">{escape(r["loai_hinh"])} · {ui.vn(r["dien_tich_m2"], 0)} m² · ảnh minh hoạ</span></div>')
    with g2:
        components.html(
            f'<div style="position:relative"><iframe src="{ui.map_url(r["dia_chi"])}" loading="lazy" '
            f'style="border:0;width:100%;height:190px;border-radius:12px" referrerpolicy="no-referrer-when-downgrade">'
            f'</iframe><span style="position:absolute;left:10px;bottom:12px;background:rgba(255,255,255,.95);'
            f'font:600 12px sans-serif;color:#2a2a33;border-radius:6px;padding:3px 9px">📍 {escape(str(r["phuong"]))}, '
            f'{escape(QL(r["quan"]))}</span></div>', height=194)
        if len(hist) >= 2:
            ch = hist[-1] / hist[0] - 1
            labels = [f"T-{len(hist) - 1 - k}" if k < len(hist) - 1 else "Nay" for k in range(len(hist))]
            ui.html(f'<div class="z-chart" style="height:194px"><div class="t">Đơn giá khu vực 12 tháng · '
                    f'<span style="color:{"#11734b" if ch >= 0 else "#d1242f"}">{ch:+.0%}</span></div>'
                    + ui.line_svg(hist, labels, w=420, h=158) + "</div>")
        else:
            ui.html('<div class="z-chart" style="height:194px"><div class="t">Đơn giá khu vực 12 tháng</div>'
                    '<div class="z-meta" style="margin-top:60px;text-align:center">Tin không có dữ liệu lịch sử giá</div></div>')

    main, side = st.columns([2.1, 1])
    with main:
        lo, hi = ui.vn(est_range(r["y_pred"])[0], 2), ui.ty(est_range(r["y_pred"])[1])
        c1, c2 = st.columns([1.5, 1])
        c1.markdown(f'<span class="z-status"><span class="z-dot"></span>Đang rao bán</span> {ui.price_status(r)}'
                    f'<div class="z-price">{ui.ty(r["gia_ban_ty"])}</div>'
                    f'<div class="z-addr">{escape(str(r["dia_chi"]).split(" (")[0])}</div>', unsafe_allow_html=True)
        c2.markdown('<div class="z-stats" style="margin-top:34px">'
                    + "".join(f'<div><b>{v}</b>{lab}</div>' for v, lab in [
                        (int(r["so_phong_ngu_n"]) if pd.notna(r["so_phong_ngu_n"]) else "—", "phòng ngủ"),
                        (int(r["so_wc_n"]) if pd.notna(r["so_wc_n"]) else "—", "WC"),
                        (ui.vn(r["dien_tich_m2"], 0), "m²")]) + "</div>", unsafe_allow_html=True)
        ui.html(f'<div class="z-est"><span>🤖 Giá ước tính AI: <b>{ui.ty(r["y_pred"])}</b></span>'
                f'<span>Khoảng 80%: <b>{lo} – {hi}</b></span>'
                f'<span>Trả góp ước tính: <b>{ui.vn(ui.monthly(r["gia_ban_ty"]), 0)} tr/tháng</b></span></div>')

        def fact(icon, label, val):
            return f'<div class="z-fact"><span class="i">{icon}</span><div><small>{label}</small>{escape(str(val))}</div></div>'
        ngang_dai = (f"{ui.vn(r['chieu_ngang_m'], 1)} × {ui.vn(r['chieu_dai_m'], 1)} m"
                     if pd.notna(r["chieu_ngang_m"]) and pd.notna(r["chieu_dai_m"]) else "—")
        ui.html('<div class="z-facts">'
                + fact("🏠", "Loại hình", r["loai_hinh"])
                + fact("📜", "Pháp lý", r["giay_to_phap_ly"] if isinstance(r["giay_to_phap_ly"], str) else "Chưa rõ")
                + fact("🏢", "Số tầng", int(r["tong_so_tang"]) if pd.notna(r["tong_so_tang"]) else "—")
                + fact("📐", "Ngang × dài", ngang_dai)
                + fact("🏷", "Đơn giá", f"{ui.vn(r['don_gia_trm2'], 0)} tr/m²")
                + fact("🛋", "Nội thất", r["tinh_trang_noi_that"] if isinstance(r["tinh_trang_noi_that"], str) else "Chưa rõ")
                + "</div>")

        raw = str(r["mo_ta"] or "")
        flat = " ".join(raw.split())
        full = "<br>".join(escape(x.strip()) for x in raw.splitlines() if x.strip())
        ui.html(f'<div class="z-h2">Điểm nổi bật</div><div class="z-h2" style="font-size:1.05rem;margin-top:0">'
                f'{escape(str(r["tieu_de"]))}</div><div class="z-txt">{escape(flat[:420])}'
                + (f'…<details><summary>Xem thêm</summary>{full}</details>' if len(flat) > 420 else "") + "</div>")

        ui.html('<div class="z-h2">Đánh giá giá đăng</div>')
        ui.card(price_check(r, THR))

        if len(hist) >= 2:
            st.caption(f"Đơn giá/m² khu vực {'tăng' if hist[-1] >= hist[0] else 'giảm'} "
                       f"{abs(hist[-1] / hist[0] - 1):.0%} trong 12 tháng · tin này {ui.vn(r['don_gia_trm2'], 0)} tr/m² "
                       f"so với mức hiện tại của khu vực {ui.vn(hist[-1], 0)} tr/m².")

        ui.html('<div class="z-h2">Thông tin chi tiết</div>')
        rows = [("Diện tích đất", f"{ui.vn(r['dien_tich_m2'], 1)} m²"),
                ("Diện tích sử dụng", f"{ui.vn(r['dt_su_dung_m2'], 1)} m²" if pd.notna(r["dt_su_dung_m2"]) else "—"),
                ("Hướng cửa chính", r["huong_cua_chinh"] if isinstance(r["huong_cua_chinh"], str) else "—"),
                ("Đặc điểm", r["dac_diem"] if isinstance(r["dac_diem"], str) else "—"),
                ("Phường", r["phuong"]), ("Quận", QL(r["quan"])),
                ("Đơn giá khu vực (TB 12 tháng)", f"{ui.vn(r['bdg_mean'], 0)} tr/m²" if pd.notna(r["bdg_mean"]) else "—"),
                ("Khung đơn giá khu vực", f"{ui.vn(r['san_khu_vuc'], 0)} – {ui.vn(r['tran_khu_vuc'], 0)} tr/m²")]
        ui.card('<div class="z-dl">' + "".join(f'<div><span>{a}</span><b>{escape(str(b))}</b></div>' for a, b in rows) + "</div>")

        ui.html('<div class="z-h2">Nhà tương tự gần đây</div>')
        same = SC[(SC.quan == r["quan"]) & (SC.index != i)]
        d = (np.abs(np.log(same.dien_tich_m2.clip(lower=5)) - np.log(max(r["dien_tich_m2"], 5)))
             + 0.15 * (same.so_phong_ngu_n.fillna(3) - (r["so_phong_ngu_n"] if pd.notna(r["so_phong_ngu_n"]) else 3)).abs()
             + 0.4 * (same.phuong != r["phuong"]) + 0.4 * (same.loai_hinh != r["loai_hinh"]))
        grid(same.loc[d.nsmallest(3).index], "sim")

    with side:
        with st.container(border=True, key="card_agent"):
            ui.html(f'<div class="z-agent"><div class="z-ava">NT</div><div><div class="z-meta">Tin đăng trên</div>'
                    f'<b>Nhà Tốt · Chợ Tốt</b><div class="z-meta">Mã tin #{i}</div></div></div>')
            st.button("💰 Định giá căn tương tự", type="primary", width="stretch", on_click=go, args=(MENU[3],))
            st.button("🔎 Xem nhà cùng khu vực", width="stretch", on_click=go, args=(MENU[1],))
        with st.container(border=True, key="card_loan"):
            ui.html('<div class="z-h2" style="margin-top:0;font-size:1.1rem">Ước tính trả góp</div>')
            down = st.slider("Trả trước (%)", 10, 90, 30, 5, key="m_down")
            rate = st.number_input("Lãi suất (%/năm)", 1.0, 20.0, 10.0, 0.5, key="m_rate")
            years = st.selectbox("Thời hạn vay", [10, 15, 20, 25, 30], index=2, key="m_years",
                                 format_func=lambda y: f"{y} năm")
            pay = ui.monthly(r["gia_ban_ty"], down / 100, rate / 100, years)
            ui.html(f'<div class="z-kpi">Mỗi tháng<b>{ui.vn(pay, 1)} triệu</b></div>'
                    f'<div class="z-meta">Vay {ui.ty(r["gia_ban_ty"] * (1 - down / 100))} · tổng lãi '
                    f'{ui.ty(pay * years * 12 / 1000 - r["gia_ban_ty"] * (1 - down / 100))}</div>')

# ----------------------------------------------------------------------
elif choice == MENU[3]:
    ui.html('<div class="z-h2" style="font-size:1.9rem;margin-top:0">Căn nhà của bạn đáng giá bao nhiêu?</div>'
            '<div class="z-meta">Nhập thông tin → mô hình XGBoost ước tính giá. Nhập thêm giá bạn định đăng để kiểm tra '
            'có bị xem là bất thường không.</div>')
    st.write("")
    tab1, tab2 = st.tabs(["Định giá 1 căn", "Định giá từ file CSV / Excel"])
    with tab1:
        left, right = st.columns([1.15, 1])
        with left, st.container(border=True, key="card_form"):
            c1, c2 = st.columns(2)
            quan = c1.selectbox("Quận", CH["quan"], format_func=QL)
            phuong = c2.selectbox("Phường", CH["phuong_by_quan"][quan])
            with st.form("form_predict", border=False):
                c1, c2 = st.columns(2)
                loai = c1.selectbox("Loại hình", CH["loai_hinh"])
                phap_ly = c2.selectbox("Giấy tờ pháp lý", CH["giay_to_phap_ly"])
                c1, c2, c3 = st.columns(3)
                dt = c1.number_input("Diện tích (m²)", 10.0, 1500.0, 50.0, 1.0)
                ngang = c2.number_input("Ngang (m)", 1.0, 50.0, 4.0, 0.1)
                dai = c3.number_input("Dài (m)", 1.0, 100.0, 12.5, 0.1)
                tang = c1.number_input("Số tầng", 1, 30, 3)
                pn = c2.number_input("Phòng ngủ", 1, 20, 3)
                wc = c3.number_input("WC", 1, 20, 3)
                c1, c2 = st.columns(2)
                mat_tien = c1.checkbox("Mặt tiền / MT kinh doanh")
                hem_xh = c2.checkbox("Hẻm xe hơi")
                mo_ta = st.text_area("Mô tả (tuỳ chọn)", placeholder="Nhà hẻm xe hơi, gần chợ, sổ hồng chính chủ…")
                gia_dang = st.number_input("Giá bạn định đăng (tỷ) — để 0 nếu chỉ cần ước tính", 0.0, 1000.0, 0.0, 0.1)
                ok = st.form_submit_button("Ước tính giá", type="primary")
        with right:
            if ok:
                mt = mat_tien or bool(pd.Series([mo_ta]).str.contains(core.KEY_MAT_TIEN, case=False).iloc[0])
                hx = hem_xh or bool(pd.Series([mo_ta]).str.contains(core.KEY_HEM_XH, case=False).iloc[0])
                row = pd.DataFrame([{
                    "quan": quan, "phuong": phuong, "loai_hinh": loai, "giay_to_phap_ly": phap_ly,
                    "dien_tich_m2": dt, "chieu_ngang_m": ngang, "chieu_dai_m": dai, "tong_so_tang": tang,
                    "so_phong_ngu_n": pn, "so_wc_n": wc, "co_mat_tien": int(mt), "co_hem_xh": int(hx),
                    "mo_ta_len": max(len(mo_ta), 1) if mo_ta else 360,   # trống -> trung vị độ dài mô tả
                    "bdg_mean": np.nan, "bdg_std": np.nan, "bdg_slope": np.nan,
                    "gia_ban_ty": gia_dang if gia_dang > 0 else np.nan}])
                row, yp = predict(row)
                pred = float(yp[0])
                lo, hi = est_range(pred)
                ui.card(f'<div class="z-meta">🤖 Giá ước tính AI · {ui.vn(dt, 0)} m² · {escape(loai)} · '
                        f'{escape(phuong)}, {QL(quan)}</div><div class="z-price">{ui.ty(pred)}</div>'
                        f'<div class="z-txt">Khoảng tham khảo 80%: <b>{ui.vn(lo, 2)} – {ui.ty(hi)}</b> · '
                        f'đơn giá <b>{ui.vn(pred * 1000 / dt, 1)} tr/m²</b> · trả góp ~'
                        f'<b>{ui.vn(ui.monthly(pred), 0)} tr/tháng</b></div>'
                        + ui.range_bar(lo, hi, pred, gia_dang if gia_dang > 0 else None)
                        + '<div class="z-meta">Khoảng 80% = giá ước tính × (P10–P90 của tỉ lệ giá thật / giá dự đoán, out-of-fold). Là khoảng tham khảo: phủ ~84% nhà dưới 8 tỷ, ~76% nhà 8–12 tỷ, ~65% nhà trên 12 tỷ.</div>')
                if gia_dang > 0:
                    row["don_gia_trm2"] = gia_dang * 1000 / dt
                    out, thr = SCORER.score(row, yp)
                    rr = pd.concat([row, out], axis=1).iloc[0]
                    ui.html('<div class="z-h2">Kiểm tra giá bạn định đăng</div>')
                    ui.card(price_check(rr, thr))
            else:
                ui.card('<div class="z-h2" style="margin-top:0">Cách ước tính</div><div class="z-txt">'
                        'Mô hình <b>XGBoost</b> học từ 7.878 tin rao bán (6.302 train / 1.576 test) với 16 đặc trưng: diện tích, số phòng, '
                        'số tầng, kích thước, loại hình, pháp lý, phường/quận, lịch sử đơn giá khu vực 12 tháng, '
                        'độ dài mô tả và từ khoá "mặt tiền" / "hẻm xe hơi".<br><br>Trên tập test: sai số trung bình '
                        f'<b>{ui.ty(B["metrics_test"]["MAE"])}</b>, R² <b>{ui.vn(B["metrics_test"]["R2"], 3)}</b>.</div>')
                ui.html('<div class="z-h2" style="font-size:1.05rem">Đặc trưng quan trọng nhất</div>')
                st.bar_chart(IMPORTANCE.head(8), horizontal=True, sort=False, color=ui.ACCENT, height=260)

    with tab2:
        st.markdown("Upload file CSV theo **định dạng tin đăng Nhà Tốt** (các cột `dien_tich`, `dia_chi`, "
                    "`loai_hinh`, `so_phong_ngu`…). File **không cần** cột giá.")
        d1, d2 = st.columns(2)
        with open(P("data", "sample_du_doan.xlsx"), "rb") as f:
            d1.download_button("⬇️ File mẫu Excel", f, "sample_du_doan.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with open(P("data", "sample_du_doan.csv"), "rb") as f:
            d2.download_button("⬇️ File mẫu CSV", f, "sample_du_doan.csv", "text/csv")
        up = st.file_uploader("Chọn file CSV hoặc Excel (.xlsx)", type=["csv", "xlsx"], key="up_pred")
        if up is not None:
            try:
                raw = csv_io.load_upload(st, up, load_lexicon())
                df = core.clean(raw, require_target=False) if core.is_raw(raw) else raw.copy()
                miss = [c for c in core.FEATURES if c not in df.columns]
                if miss:
                    st.error(f"Thiếu cột: {miss}")
                    st.stop()
                if df["quan"].isna().any():
                    st.warning(f"{int(df['quan'].isna().sum())} dòng không nhận ra quận từ địa chỉ — "
                               "model vẫn dự đoán nhưng kém chính xác hơn.")
                df, yp = predict(df)
                df["gia_du_doan_ty"] = np.round(yp, 2)
                df["don_gia_du_doan_trm2"] = np.round(yp * 1000 / df["dien_tich_m2"], 1)
                ui.kpis([("Số căn", str(len(df))), ("Giá ước tính trung vị", ui.ty(float(np.median(yp)))),
                         ("Thấp nhất", ui.ty(float(yp.min()))), ("Cao nhất", ui.ty(float(yp.max())))])
                show = [c for c in ["tieu_de", "dia_chi", "loai_hinh", "dien_tich_m2", "so_phong_ngu_n",
                                    "tong_so_tang", "gia_du_doan_ty", "don_gia_du_doan_trm2"] if c in df.columns]
                st.dataframe(df[show], width="stretch")
                st.download_button("⬇️ Tải kết quả", df[show].to_csv(index=False).encode("utf-8-sig"),
                                   "ket_qua_du_doan_gia.csv", "text/csv")
            except Exception as e:
                st.error(f"Không xử lý được file: {e}")

# ----------------------------------------------------------------------
elif choice == MENU[4]:
    ui.html('<div class="z-h2" style="font-size:1.9rem;margin-top:0">Kiểm tra giá hàng loạt</div>'
            '<div class="z-meta">Upload danh sách tin có giá → mỗi tin được chấm 4 tín hiệu, cộng có trọng số thành '
            'composite score và so với 7.926 tin tham chiếu; top-5% bị gắn cờ.</div>')
    ui.card('<div class="z-facts" style="margin:0;grid-template-columns:repeat(4,1fr)">'
            + "".join(f'<div class="z-fact"><div><small>{lab}</small>{txt}</div></div>' for _, lab, txt in SIGNALS)
            + "</div>")
    d1, d2 = st.columns(2)
    with open(P("data", "sample_co_gia.xlsx"), "rb") as f:
        d1.download_button("⬇️ File mẫu Excel", f, "sample_co_gia.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with open(P("data", "sample_co_gia.csv"), "rb") as f:
        d2.download_button("⬇️ File mẫu CSV", f, "sample_co_gia.csv", "text/csv")
    up = st.file_uploader("Chọn file CSV hoặc Excel (.xlsx) có cột `gia_ban`", type=["csv", "xlsx"], key="up_anom")
    w, k = weights_widget("up")
    if up is not None:
        try:
            raw = csv_io.load_upload(st, up, load_lexicon())
            df = core.clean(raw, require_target=True) if core.is_raw(raw) else raw.copy()
            if "gia_ban_ty" not in df.columns or df["gia_ban_ty"].isna().all():
                st.error("File phải có cột giá `gia_ban` (vd '3,85 tỷ').")
                st.stop()
            df, yp = predict(df)
            out, thr = SCORER.score(df, yp, w, k)
            res = pd.concat([df, out], axis=1)
            n = int(res["flag_bat_thuong"].sum())
            hc = res.loc[res.flag_bat_thuong == 1, "huong"].value_counts()
            ui.kpis([("Số tin", str(len(res))), ("Bị gắn cờ", str(n)),
                     ("Quá rẻ / quá đắt", f"{hc.get('QUÁ RẺ', 0)} / {hc.get('QUÁ ĐẮT', 0)}"),
                     ("Ngưỡng composite", f"{thr:.3f}")])
            st.markdown('<div class="z-h2">Tin nghi vấn nhất</div>', unsafe_allow_html=True)
            grid(res.sort_values("total_score", ascending=False).head(6), None)
            with st.expander("Bảng đầy đủ"):
                cols = ["tieu_de", "dia_chi", "loai_hinh", "dien_tich_m2", "gia_ban_ty", "y_pred", "huong",
                        "s_resid", "s_minmax", "s_pct", "s_ml", "total_score", "flag_bat_thuong"]
                st.dataframe(res[[c for c in cols if c in res.columns]].sort_values("total_score", ascending=False),
                             width="stretch", column_config={
                                 "gia_ban_ty": st.column_config.NumberColumn("Giá đăng (tỷ)", format="%.2f"),
                                 "y_pred": st.column_config.NumberColumn("Giá ước tính (tỷ)", format="%.2f"),
                                 "total_score": st.column_config.ProgressColumn("Composite", min_value=0, max_value=1, format="%.3f"),
                                 "flag_bat_thuong": st.column_config.CheckboxColumn("Bất thường")})
            st.download_button("⬇️ Tải kết quả", res.to_csv(index=False).encode("utf-8-sig"),
                               "ket_qua_bat_thuong.csv", "text/csv")
        except Exception as e:
            st.error(f"Không xử lý được file: {e}")

# ----------------------------------------------------------------------
elif choice == MENU[5]:
    ui.html('<div class="z-h2" style="font-size:1.9rem;margin-top:0">Dữ liệu & kết quả mô hình</div>'
            '<div class="z-meta">EDA · so sánh các mô hình scikit-learn (PySpark chạy trong notebook) · 4 phương pháp phát hiện bất thường</div>')
    t1, t2, t3 = st.tabs(["Khám phá dữ liệu", "So sánh mô hình", "Phát hiện bất thường"])
    with t1:
        st.image(P("images", "eda_target.png"), caption="Phân phối giá bán — lệch phải mạnh")
        st.image(P("images", "eda_categorical.png"), caption="Biến phân loại")
        st.image(P("images", "eda_corr.png"), caption="Tương quan giữa các biến số")
        st.image(P("images", "wordcloud_mo_ta.png"), caption="Wordcloud cột mô tả")
    with t2:
        st.markdown("**XGBoost** cho RMSE thấp nhất → chọn để triển khai. Đánh giá trên 20% dữ liệu giữ lại (tách ngẫu nhiên). "
                    "Lưu ý: bảng chỉ có mô hình scikit-learn; kết quả PySpark nằm trong notebook.")
        st.caption("Mô hình nên được so với mốc đơn giản (ví dụ trung vị giá theo quận/phường/loại hình) để thấy giá trị tăng thêm. "
                   "Dữ liệu 7.926 tin: 7.878 tin dùng để học (sau khi loại giá cực đoan), "
                   "6.302 train / 1.576 test.")
        st.dataframe(pd.read_csv(P("data", "bang_so_sanh_model.csv")).round(3), width="stretch", hide_index=True)
        st.image(P("images", "bang_so_sanh_model.png"))
        st.image(P("images", "model_sklearn.png"), caption="So sánh model scikit-learn + dự đoán vs thực tế")
        st.subheader("Đặc trưng quan trọng nhất (XGBoost, theo gain)")
        st.caption("Tính theo gain và cộng dồn các cột one-hot nên đề cao biến phân loại (loại hình, phường, quận). Kiểm bằng permutation importance trên tập test thì diện tích quan trọng nhất. Chỉ đọc là thứ hạng tương đối, không phải quan hệ nhân quả.")
        st.bar_chart(IMPORTANCE.head(12), horizontal=True, sort=False, color=ui.ACCENT)
    with t3:
        st.image(P("images", "anomaly_overview.png"), caption="Phân phối composite score và tin bị gắn cờ")
        st.image(P("images", "anomaly_jaccard.png"), caption="Mức độ đồng thuận giữa 4 tín hiệu")
        st.dataframe(pd.read_csv(P("data", "bang_uu_nhuoc_anomaly.csv")), width="stretch", hide_index=True)

# ----------------------------------------------------------------------
else:
    ui.html('<div class="z-h2" style="font-size:1.9rem;margin-top:0">Nhóm thực hiện</div>'
            '<div class="z-meta">Đồ án tốt nghiệp Data Science — Trung tâm Tin học, ĐH KHTN TP.HCM</div>')
    st.write("")
    c = st.columns(2)
    for col, m in zip(c, TEAM):
        ini = "".join(w[0] for w in m["ten"].split()[-2:]).upper()
        with col:
            ui.card(f'<div class="z-agent"><div class="z-ava" style="width:56px;height:56px">{escape(ini)}</div><div>'
                    f'<b style="font-size:1.15rem">{escape(m["ten"])}</b><div class="z-meta">📧 {escape(m["email"])}</div>'
                    f'</div></div><div class="z-txt" style="margin-top:10px"><b>Phụ trách:</b> {escape(m["viec"])}</div>')

# ======================================================================
# Phản hồi điều hướng: thông báo "Đã mở …" + cuộn lên đầu trang khi đổi trang / đổi tin
# ======================================================================
if "_toast" in ss:
    st.toast(ss.pop("_toast"), icon="🏡")
view = (ss.nav, ss.listing if ss.nav == MENU[2] else None)
if ss.get("_last_view") != view:
    ss._last_view = view
    ss._scroll_n = ss.get("_scroll_n", 0) + 1     # nội dung khác nhau mỗi lần -> script chạy lại
    components.html(
        f"<script>/* {ss._scroll_n} */ const d = window.parent.document;"
        "for (const s of ['[data-testid=\"stMain\"]', '[data-testid=\"stAppViewContainer\"]', 'section.main']) {"
        "  const el = d.querySelector(s); if (el) el.scrollTo({top: 0, behavior: 'smooth'}); }"
        "window.parent.scrollTo({top: 0, behavior: 'smooth'});</script>", height=0)
