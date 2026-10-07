"""Huấn luyện lại model cho GUI Nhà Tốt từ data/df_clean.parquet (sinh ở Mục 4
notebook project_1) và lưu vào models/nhatot_bundle.joblib.

Chạy:  python train_models.py
"""
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

import nhatot_core as core

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "df_clean.parquet")
OUT  = os.path.join(HERE, "models", "nhatot_bundle.joblib")


def metrics(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
            "R2": r2_score(y_true, y_pred),
            "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)}


def main():
    t0 = time.time()
    df_clean = pd.read_parquet(DATA)
    print(f"df_clean: {df_clean.shape}")

    # ---- BT1: cắt 0,5% đuôi giá cực đoan CHỈ cho bài toán dự đoán (Mục 6) ----
    p_lo, p_hi = df_clean[core.TARGET].quantile(0.001), df_clean[core.TARGET].quantile(0.995)
    df_model = df_clean[df_clean[core.TARGET].between(p_lo, p_hi)].reset_index(drop=True)
    X, y = df_model[core.FEATURES], df_model[core.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=core.RANDOM_STATE)

    pipe = Pipeline([("pre", core.make_pre()), ("model", core.make_xgb())]).fit(X_tr, y_tr)
    m = metrics(y_te.values, pipe.predict(X_te))
    print("XGBoost test:", {k: round(v, 3) for k, v in m.items()})

    # Model triển khai: fit lại trên TOÀN BỘ df_model
    final = Pipeline([("pre", core.make_pre()), ("model", core.make_xgb())]).fit(X, y)

    # Feature importance gộp về feature gốc
    names = final.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(final.named_steps["model"].feature_importances_, index=names)

    def root(f):
        f = f.split("__", 1)[-1]
        return next((c for c in core.CAT_FEATURES if f.startswith(c + "_")), f)
    imp_root = imp.groupby(imp.index.map(root)).sum().sort_values(ascending=False)

    # ---- BT2: anomaly — y_pred out-of-fold trên toàn bộ df_clean (Mục 10) ----
    y_oof = cross_val_predict(
        Pipeline([("pre", core.make_pre()), ("model", core.make_xgb())]),
        df_clean[core.FEATURES], df_clean[core.TARGET], cv=5, n_jobs=-1)
    scorer = core.AnomalyScorer().fit(df_clean, y_oof)
    scored, thr = scorer.score(df_clean, y_oof)
    print(f"Anomaly: ngưỡng {thr:.3f} · gắn cờ {int(scored.flag_bat_thuong.sum())} tin")

    # ---- Danh mục cho form nhập tay ----
    phuong_by_quan = (df_clean.groupby("quan")["phuong"]
                      .apply(lambda s: sorted(s.dropna().unique(),
                                              key=lambda p: (not p.split()[-1].isdigit(),
                                                             int(p.split()[-1]) if p.split()[-1].isdigit() else 0, p)))
                      .to_dict())
    bundle = {
        "model": final,
        "scorer": scorer,
        "bdg_table": core.make_bdg_table(df_clean),
        "metrics_test": m,
        "resid_q": {"p10": float(np.quantile(y_te.values - pipe.predict(X_te), 0.10)),
                    "p90": float(np.quantile(y_te.values - pipe.predict(X_te), 0.90))},
        "importance": imp_root,
        "choices": {
            "quan": sorted(df_clean["quan"].dropna().unique()),
            "phuong_by_quan": phuong_by_quan,
            "loai_hinh": df_clean["loai_hinh"].value_counts().index.tolist(),
            "giay_to_phap_ly": df_clean["giay_to_phap_ly"].value_counts().index.tolist(),
        },
        "n_train": len(df_model),
        "cut": (float(p_lo), float(p_hi)),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    joblib.dump(bundle, OUT, compress=3)

    # Kết quả chấm điểm toàn bộ dữ liệu -> trang "Tra cứu tin bất thường"
    keep = ["tieu_de", "dia_chi", "quan", "phuong", "loai_hinh", "giay_to_phap_ly", "dien_tich_m2",
            "dt_su_dung_m2", "so_phong_ngu_n", "so_wc_n", "tong_so_tang", "chieu_ngang_m", "chieu_dai_m",
            "huong_cua_chinh", "tinh_trang_noi_that", "dac_diem", "bieu_do_gia", "bdg_mean", "bdg_slope",
            "gia_ban_ty", "don_gia_trm2", "mo_ta"]
    pd.concat([df_clean[keep], scored], axis=1).to_parquet(
        os.path.join(HERE, "data", "df_scored.parquet"), index=False)

    print(f"Đã lưu {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB) trong {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
