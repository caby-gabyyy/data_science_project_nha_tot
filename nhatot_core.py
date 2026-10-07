"""Hàm dùng chung cho GUI Nhà Tốt — chép NGUYÊN logic từ notebook project_1
(Project1_Price_Prediction_NhaTot.ipynb, Mục 3–4, 6–7, 10) để app và notebook
cho ra cùng một kết quả."""
import ast
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# ---- FEATURES đã khoá ở Mục 6 notebook ---------------------------------
# dien_tich_dat_m2 bị bỏ vì trùng 100% với dien_tich_m2.
# TUYỆT ĐỐI KHÔNG thêm don_gia_trm2 / gia_m2_trm2 (rò rỉ nhãn).
TARGET = "gia_ban_ty"
NUM_FEATURES = ["dien_tich_m2", "so_phong_ngu_n", "so_wc_n",
                "tong_so_tang", "chieu_ngang_m", "chieu_dai_m",
                "bdg_mean", "bdg_std", "bdg_slope",
                "mo_ta_len", "co_mat_tien", "co_hem_xh"]
CAT_FEATURES = ["loai_hinh", "giay_to_phap_ly", "phuong", "quan"]
FEATURES = NUM_FEATURES + CAT_FEATURES

QUAN_LABEL = {"binh-thanh": "Bình Thạnh", "go-vap": "Gò Vấp", "phu-nhuan": "Phú Nhuận"}

# ======================================================================
# 1. Parse chuỗi số tiếng Việt (Mục 3 notebook)
# ======================================================================
def to_num_comma(s):
    # Tiền tệ VN: '3,85' -> 3.85 ; '1.070' -> 1070 (chấm = phân cách nghìn)
    if pd.isna(s):
        return np.nan
    x = re.sub(r"[^0-9,.]", "", str(s)).replace(".", "").replace(",", ".")
    try:
        return float(x) if x not in ("", ".") else np.nan
    except ValueError:
        return np.nan


def to_num_dot(s):
    # Đo lường: '59.4 m²' -> 59.4 ; '4.5 m' -> 4.5 ; '2 phòng' -> 2 (chấm = thập phân)
    if pd.isna(s):
        return np.nan
    m = re.search(r"\d+(?:[.,]\d+)?", str(s).replace(",", "."))
    return float(m.group()) if m else np.nan


def gia_ban_ty(s):
    # Tổng giá -> tỷ VND. '9,79 tỷ' -> 9.79 ; dòng ghi 'triệu' -> chia 1000
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.number)):
        return float(s)
    v = to_num_comma(s)
    t = str(s)
    return v / 1000 if ("triệu" in t and "tỷ" not in t) else v


def don_gia_trm2(s):
    # Đơn giá -> triệu/m². '106,94 triệu/m²' -> 106.94 ; 'x tỷ/m²' -> *1000
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.number)):
        return float(s)
    v = to_num_comma(s)
    return v * 1000 if "tỷ/m²" in str(s) else v


def bdg_feats(s):
    # '[134.04, 125.8, ...]' -> (mean, std, slope)
    try:
        a = np.asarray(ast.literal_eval(s), dtype=float)
        if a.size < 2:
            return (np.nan, np.nan, np.nan)
        slope = np.polyfit(np.arange(len(a)), a, 1)[0]
        return (float(a.mean()), float(a.std()), float(slope))
    except Exception:
        return (np.nan, np.nan, np.nan)


# ======================================================================
# 2. Làm sạch (Mục 4 notebook)
# ======================================================================
DROP_COLS = ["ma_can", "ten_phan_khu_lo", "dien_thoai"]

BOUNDS = {
    "dien_tich_m2":     (5,    2000),
    "dien_tich_dat_m2": (5,    2000),
    "dt_su_dung_m2":    (5,    5000),
    "chieu_ngang_m":    (1,      50),
    "chieu_dai_m":      (1,     200),
    "so_phong_ngu_n":   (0,      20),
    "so_wc_n":          (0,      20),
    "tong_so_tang":     (0,      30),
    "gia_ban_ty":       (0.05, 1000),
    "don_gia_trm2":     (1,   10000),
}

KEY_MAT_TIEN = r"mặt tiền|mt kinh doanh"
KEY_HEM_XH   = r"hẻm xe hơi|hẻm ô tô|hẻm oto|xe hơi"

# Cột thô tối thiểu của file crawl Nhà Tốt (quan-*.csv)
RAW_COLS = ["tieu_de", "gia_ban", "don_gia", "dien_tich", "dia_chi", "mo_ta",
            "loai_hinh", "dien_tich_dat", "dien_tich_su_dung", "gia_m2",
            "giay_to_phap_ly", "so_phong_ngu", "so_phong_ve_sinh", "tong_so_tang",
            "dac_diem", "chieu_ngang", "chieu_dai", "bieu_do_gia"]


def infer_quan(dia_chi):
    """'..., Quận Bình Thạnh, ...' -> 'binh-thanh' (khớp nhãn quận lúc train)."""
    s = str(dia_chi).lower()
    if "bình thạnh" in s:
        return "binh-thanh"
    if "gò vấp" in s:
        return "go-vap"
    if "phú nhuận" in s:
        return "phu-nhuan"
    return np.nan


def clean(df, require_target=True):
    """Giống clean() trong notebook. require_target=False dùng cho file
    cần DỰ ĐOÁN giá (không có cột gia_ban) — khi đó không bỏ dòng, không khử trùng."""
    df = df.copy()
    for c in RAW_COLS:
        if c not in df.columns:
            df[c] = np.nan
    if "quan" not in df.columns:
        df["quan"] = df["dia_chi"].apply(infer_quan)

    df["gia_ban_ty"]       = df["gia_ban"].apply(gia_ban_ty)
    df["don_gia_trm2"]     = df["don_gia"].apply(don_gia_trm2)
    df["gia_m2_trm2"]      = df["gia_m2"].apply(don_gia_trm2)
    df["dien_tich_m2"]     = df["dien_tich"].apply(to_num_dot)
    df["dien_tich_dat_m2"] = df["dien_tich_dat"].apply(to_num_dot)
    df["dt_su_dung_m2"]    = df["dien_tich_su_dung"].apply(to_num_dot)
    df["so_phong_ngu_n"]   = df["so_phong_ngu"].apply(to_num_dot)
    df["so_wc_n"]          = df["so_phong_ve_sinh"].apply(to_num_dot)
    df["chieu_ngang_m"]    = df["chieu_ngang"].apply(to_num_dot)
    df["chieu_dai_m"]      = df["chieu_dai"].apply(to_num_dot)
    df["tong_so_tang"]     = pd.to_numeric(df["tong_so_tang"], errors="coerce")

    df[["bdg_mean", "bdg_std", "bdg_slope"]] = pd.DataFrame(
        df["bieu_do_gia"].apply(bdg_feats).tolist(), index=df.index)

    df["phuong"] = (df["dia_chi"].astype(str)
                    .str.extract(r"(Phường\s+\d+|Phường\s+[A-Za-zÀ-ỹ\s]+?)[,\)]")[0]
                    .str.replace(r"\s+", " ", regex=True).str.strip())

    mo_ta = df["mo_ta"].fillna("").astype(str)
    df["mo_ta_len"]   = mo_ta.str.len()
    df["co_mat_tien"] = mo_ta.str.contains(KEY_MAT_TIEN, case=False, regex=True).astype(int)
    df["co_hem_xh"]   = mo_ta.str.contains(KEY_HEM_XH,   case=False, regex=True).astype(int)

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    if require_target:
        df = df[df["gia_ban_ty"].notna()]
        dup_keys = [c for c in ["tieu_de", "gia_ban", "dien_tich", "dia_chi"] if c in df.columns]
        df = df.drop_duplicates(subset=dup_keys, keep="first")

    for col, (lo, hi) in BOUNDS.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        df.loc[bad, col] = np.nan

    # Đơn giá thiếu nhưng có tổng giá + diện tích -> tự tính (dùng cho anomaly)
    miss = df["don_gia_trm2"].isna() & df["gia_ban_ty"].notna() & df["dien_tich_m2"].notna()
    df.loc[miss, "don_gia_trm2"] = df.loc[miss, "gia_ban_ty"] * 1000 / df.loc[miss, "dien_tich_m2"]

    return df.reset_index(drop=True)


def is_raw(df):
    """File thô (chuỗi '3,85 tỷ', '36 m²'...) hay file đã có cột số sạch?"""
    return not set(NUM_FEATURES).issubset(df.columns)


def fill_bdg_from_area(df, bdg_table):
    """Tin thiếu lịch sử giá -> lấy trung vị lịch sử giá của (quận, phường),
    rồi của quận. Đây cũng là cách form nhập tay có được bdg_*."""
    df = df.copy()
    for c in ["bdg_mean", "bdg_std", "bdg_slope"]:
        by_p = df.set_index(["quan", "phuong"]).index.map(
            lambda k: bdg_table["phuong"].get(k, {}).get(c, np.nan))
        by_q = df["quan"].map(lambda q: bdg_table["quan"].get(q, {}).get(c, np.nan))
        df[c] = df[c].fillna(pd.Series(by_p, index=df.index)).fillna(by_q)
    return df


def make_bdg_table(df_ref):
    cols = ["bdg_mean", "bdg_std", "bdg_slope"]
    by_p = df_ref.groupby(["quan", "phuong"])[cols].median()
    by_q = df_ref.groupby("quan")[cols].median()
    return {"phuong": {k: v.to_dict() for k, v in by_p.iterrows()},
            "quan":   {k: v.to_dict() for k, v in by_q.iterrows()}}


# ======================================================================
# 3. Pipeline model (Mục 7 notebook)
# ======================================================================
def make_pre():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), NUM_FEATURES),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh",  OneHotEncoder(handle_unknown="ignore",
                                                sparse_output=False))]), CAT_FEATURES),
    ])


def make_xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                        colsample_bytree=0.8, n_jobs=-1, random_state=RANDOM_STATE)


# ======================================================================
# 4. Phát hiện bất thường (Mục 10 notebook)
#    Tham chiếu (ngưỡng, phân vị, IsolationForest) được "học" trên df_clean
#    rồi ÁP DỤNG cho tin mới — tin mới không làm xê dịch ngưỡng.
# ======================================================================
W_DEFAULT = {"resid": 0.40, "minmax": 0.20, "pct": 0.20, "ml": 0.20}
K_PERCENT = 0.05
MIN_N, Q_LO, Q_HI = 20, 0.01, 0.99
ISO_COLS = ["gia_ban_ty", "don_gia_trm2", "dien_tich_m2",
            "so_phong_ngu_n", "so_wc_n", "tong_so_tang", "bdg_mean"]
COL_DG = "don_gia_trm2"


class AnomalyScorer:
    def fit(self, an, y_pred_oof):
        """an = df_clean, y_pred_oof = dự đoán out-of-fold (cross_val_predict 5-fold)."""
        resid = an[TARGET].values - y_pred_oof
        self.resid_mu, self.resid_sd = float(np.mean(resid)), float(np.std(resid, ddof=1))

        # s2 — sàn/trần theo (quận, phường, loại hình) -> (quận, loại hình) -> toàn bộ
        self.bounds = {}
        for keys in (["quan", "phuong", "loai_hinh"], ["quan", "loai_hinh"]):
            g = an.groupby(keys)[COL_DG]
            t = pd.DataFrame({"lo": g.quantile(Q_LO), "hi": g.quantile(Q_HI), "n": g.count()})
            t = t[t["n"] >= MIN_N]
            self.bounds[tuple(keys)] = {k: (r.lo, r.hi) for k, r in t.iterrows()}
        self.global_bound = (float(an[COL_DG].quantile(Q_LO)), float(an[COL_DG].quantile(Q_HI)))

        # s3 — P10–P90 đơn giá/m², log1p khoảng cách, min-max theo tập tham chiếu
        self.p10, self.p90 = float(an[COL_DG].quantile(0.10)), float(an[COL_DG].quantile(0.90))
        d = self._pct_dist(an[COL_DG])
        self.d_min, self.d_max = float(np.nanmin(d)), float(np.nanmax(d))

        # s4 — IsolationForest
        self.iso_imp = SimpleImputer(strategy="median").fit(an[ISO_COLS])
        Xa = self.iso_imp.transform(an[ISO_COLS])
        self.iso = IsolationForest(n_estimators=300, contamination=0.05,
                                   random_state=RANDOM_STATE, n_jobs=-1).fit(Xa)
        sc = self.iso.decision_function(Xa)
        self.sc_min, self.sc_max = float(sc.min()), float(sc.max())

        # Điểm từng tín hiệu trên tập tham chiếu -> dùng tính lại ngưỡng khi đổi trọng số/k
        self.ref_signals = self.signals(an, y_pred_oof)[["s_resid", "s_minmax", "s_pct", "s_ml"]]
        return self

    def _pct_dist(self, dg):
        d = np.maximum(0, np.maximum(self.p10 - dg, dg - self.p90))
        return np.log1p(d)

    def _bound_row(self, r):
        for keys, table in self.bounds.items():
            k = tuple(r[c] for c in keys)
            if k in table:
                return table[k]
        return self.global_bound

    def signals(self, df, y_pred):
        out = pd.DataFrame(index=df.index)
        out["y_pred"] = y_pred
        out["resid"]  = df[TARGET].values - y_pred
        z = (out["resid"] - self.resid_mu) / self.resid_sd
        out["z_resid"] = z
        out["s_resid"] = (z.abs().clip(upper=3) / 3).fillna(0)
        out["huong"]   = np.where(out["resid"] > 0, "QUÁ ĐẮT", "QUÁ RẺ")

        b = df.apply(self._bound_row, axis=1, result_type="expand")
        out["san_khu_vuc"], out["tran_khu_vuc"] = b[0].values, b[1].values
        dg = df[COL_DG]
        out["s_minmax"] = ((dg < out["san_khu_vuc"]) | (dg > out["tran_khu_vuc"])).astype(float)
        out.loc[dg.isna(), "s_minmax"] = 0.0

        d = self._pct_dist(dg)
        out["s_pct"] = ((d - self.d_min) / (self.d_max - self.d_min + 1e-9)).clip(0, 1).fillna(0)

        sc = self.iso.decision_function(self.iso_imp.transform(df[ISO_COLS]))
        out["s_ml"] = np.clip((self.sc_max - sc) / (self.sc_max - self.sc_min), 0, 1)
        return out

    def threshold(self, w=W_DEFAULT, k=K_PERCENT):
        s = self.ref_signals
        total = (w["resid"] * s["s_resid"] + w["minmax"] * s["s_minmax"]
                 + w["pct"] * s["s_pct"] + w["ml"] * s["s_ml"])
        return float(total.quantile(1 - k))

    def score(self, df, y_pred, w=W_DEFAULT, k=K_PERCENT):
        out = self.signals(df, y_pred)
        out["total_score"] = (w["resid"] * out["s_resid"] + w["minmax"] * out["s_minmax"]
                              + w["pct"] * out["s_pct"] + w["ml"] * out["s_ml"])
        thr = self.threshold(w, k)
        out["flag_bat_thuong"] = (out["total_score"] >= thr).astype(int)
        return out, thr
