"""Thành phần giao diện (HTML/CSS) cho GUI Nhà Tốt — bố cục trang bất động sản:
thẻ nhà dạng lưới, trang chi tiết (giá, ước tính AI, thông tin nhanh), thanh so sánh giá."""
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

BLUE = "#0d5bdd"

CSS = """
<style>
:root{--ink:#2a2a33;--muted:#596173;--line:#d9dbe1;--soft:#f3f4f7;--blue:#0d5bdd;--blue-soft:#e9f0fd}
.block-container,[data-testid="stMainBlockContainer"]{padding-top:4.2rem;max-width:1240px}
.z-card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:14px}
.z-hero{border-radius:14px;padding:34px 34px;color:#fff;margin-bottom:16px;
  background:linear-gradient(120deg,#0b3a8c 0%,#0d5bdd 55%,#4b8df8 100%)}
.z-hero h1{color:#fff;font-size:2.1rem;margin:0 0 6px 0;line-height:1.2}
.z-hero p{color:#e6eeff;margin:0;font-size:1.02rem}
.z-gal{display:grid;grid-template-columns:2fr 1fr;gap:6px;border-radius:12px;overflow:hidden;height:300px;margin-bottom:16px}
.z-gal .r{display:grid;grid-template-rows:1fr 1fr;gap:6px}
.z-tile{position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden}
.z-tile .cap{position:absolute;left:12px;bottom:10px;background:rgba(255,255,255,.92);color:var(--ink);
  font-size:.8rem;font-weight:600;border-radius:6px;padding:3px 9px}
.z-status{display:inline-flex;align-items:center;gap:6px;font-size:.85rem;font-weight:600;color:var(--ink);
  background:var(--soft);border-radius:999px;padding:3px 11px}
.z-dot{width:8px;height:8px;border-radius:50%;background:#d1242f;display:inline-block}
.z-price{font-size:2.3rem;font-weight:700;color:var(--ink);line-height:1.1;margin-top:8px}
.z-addr{font-size:1.05rem;color:var(--ink);margin-top:4px}
.z-stats{display:flex;gap:28px;justify-content:flex-end}
.z-stats div{font-size:.95rem;color:var(--muted)}
.z-stats b{display:block;font-size:1.7rem;color:var(--ink);line-height:1.1}
.z-est{display:inline-flex;gap:14px;flex-wrap:wrap;background:var(--blue-soft);border-radius:8px;padding:8px 14px;
  margin-top:12px;font-size:.95rem;color:var(--ink)}
.z-est b{color:var(--ink)}
.z-facts{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}
.z-fact{background:var(--soft);border-radius:8px;padding:10px 12px;font-size:.92rem;color:var(--ink);display:flex;gap:10px;align-items:center}
.z-fact .i{font-size:1.2rem}
.z-fact small{display:block;color:var(--muted);font-size:.75rem}
.z-h2{font-size:1.35rem;font-weight:700;color:var(--ink);margin:18px 0 8px}
.z-txt{color:var(--ink);font-size:.97rem;line-height:1.6}
.z-txt summary{color:var(--blue);cursor:pointer;font-weight:600;margin-top:6px}
.z-meta{color:var(--muted);font-size:.84rem}
.z-badge{display:inline-block;font-size:.75rem;font-weight:600;border-radius:4px;padding:2px 8px}
.z-good{background:#e3f5ec;color:#11734b}.z-hi{background:#fff3d6;color:#8a5a00}.z-mid{background:#eef0f4;color:#3d4452}
.z-bad{background:#fde8e8;color:#b42318}
.z-lcard{background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-bottom:6px}
.z-lcard .img{height:150px;position:relative;display:flex;align-items:center;justify-content:center}
.z-lcard .img .z-badge{position:absolute;left:10px;top:10px}
.z-lcard .bd{padding:10px 12px}
.z-lcard .p{font-size:1.3rem;font-weight:700;color:var(--ink)}
.z-lcard .s{font-size:.9rem;color:var(--ink);margin:2px 0}
.z-lcard .a{font-size:.85rem;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.z-range{margin:16px 0 6px}
.z-range .trk{position:relative;height:10px;background:var(--soft);border-radius:999px}
.z-range .band{position:absolute;top:0;height:10px;background:#9bbcf5;border-radius:999px}
.z-range .mk{position:absolute;top:-7px;width:4px;height:24px;border-radius:2px;transform:translateX(-2px)}
.z-range .lbl{position:relative;height:38px;font-size:.8rem;color:var(--muted)}
.z-range .lbl span{position:absolute;transform:translateX(-50%);white-space:nowrap;top:6px;text-align:center}
.z-dl{display:grid;grid-template-columns:1fr 1fr;gap:4px 28px;font-size:.93rem;color:var(--ink)}
.z-dl div{display:flex;justify-content:space-between;border-bottom:1px solid var(--soft);padding:6px 0}
.z-dl span{color:var(--muted)}
.z-agent{display:flex;gap:12px;align-items:center}
.z-ava{width:48px;height:48px;border-radius:50%;background:var(--blue-soft);color:var(--blue);display:flex;
  align-items:center;justify-content:center;font-weight:700;flex:none}
.z-kpi{font-size:.85rem;color:var(--muted)}
.z-kpi b{display:block;font-size:1.4rem;color:var(--ink);margin-top:2px}
.z-team{font-size:.85rem;line-height:1.45}
div[data-testid="stButton"] button[kind="primary"]{background:var(--blue);border-color:var(--blue)}
</style>
"""

GRADS = [("#dfe9fb", "#b9cff5"), ("#e6f2ea", "#bfe0cb"), ("#fbefe0", "#f3d3a8"),
         ("#efe7fa", "#d5c2f1"), ("#e3f3f6", "#b7dfe7"), ("#f6e6e6", "#ebc2c2")]
LOAI_ICON = {"Nhà ngõ, hẻm": "🏠", "Nhà mặt phố, mặt tiền": "🏪", "Nhà phố liền kề": "🏘️", "Nhà biệt thự": "🏡"}


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def vn(x, d=1):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def ty(x, d=2):
    return f"{vn(x, d)} tỷ"


def html(s):
    """Gộp 1 dòng — tránh Markdown hiểu '- …' trong text là danh sách."""
    st.markdown(" ".join(str(s).split("\n")), unsafe_allow_html=True)


def card(s):
    html(f'<div class="z-card">{s}</div>')


def house_svg(seed, size=120, floors=2):
    """Hình minh hoạ căn nhà (dữ liệu không có ảnh thật)."""
    roof = ["#c2553d", "#3f5f8f", "#5b7a4a", "#8a5a2b", "#6b4c9a", "#2f7a86"][seed % 6]
    wall = ["#fbfaf7", "#fff8ee", "#f4f7fb"][seed % 3]
    floors = int(min(max(floors, 1), 4))
    h = 34 + 22 * floors
    win = "".join(f'<rect x="{x}" y="{100 - 22 * f}" width="14" height="12" rx="2" fill="#9cc3f0"/>'
                  for f in range(1, floors + 1) for x in (44, 92) if not (f == 1 and x == 44))
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 150 130" xmlns="http://www.w3.org/2000/svg">'
            f'<rect x="30" y="{122 - h}" width="90" height="{h}" fill="{wall}" stroke="#c9ccd3"/>'
            f'<polygon points="22,{124 - h} 75,{94 - h} 128,{124 - h}" fill="{roof}"/>'
            f'<rect x="44" y="96" width="16" height="26" rx="2" fill="{roof}" opacity=".85"/>{win}'
            f'<rect x="0" y="122" width="150" height="8" fill="#9fb98b"/></svg>')


def tile(seed, inner, cap="", height=None):
    a, b = GRADS[seed % len(GRADS)]
    hs = f"height:{height}px;" if height else ""
    capt = f'<span class="cap">{escape(cap)}</span>' if cap else ""
    return f'<div class="z-tile" style="{hs}background:linear-gradient(160deg,{a},{b})">{inner}{capt}</div>'


def price_status(r):
    """Nhãn trạng thái giá: theo cờ bất thường trước, sau đó theo % lệch so với giá ước tính."""
    if r.get("flag_bat_thuong") == 1:
        return f'<span class="z-badge z-bad">⚠ Nghi {escape(str(r["huong"]).lower())}</span>'
    lech = r["gia_ban_ty"] / r["y_pred"] - 1
    if abs(lech) >= SUSPECT:      # lệch quá xa nhưng composite chưa tới ngưỡng -> không gọi là "giá tốt"
        return f'<span class="z-badge z-hi">Cần kiểm tra ({lech:+.0%})</span>'
    if lech <= -0.10:
        return '<span class="z-badge z-good">✓ Giá tốt</span>'
    if lech >= 0.10:
        return '<span class="z-badge z-hi">Cao hơn ước tính</span>'
    return '<span class="z-badge z-mid">Sát giá ước tính</span>'


SUSPECT = 0.35


def line_svg(values, labels, unit="tr/m²", w=640, h=220):
    """Biểu đồ đường SVG đơn giản (trục y theo min–max của chuỗi, có lưới)."""
    v = np.asarray(values, dtype=float)
    lo, hi = v.min(), v.max()
    pad = (hi - lo) * 0.15 or hi * 0.05 or 1
    lo, hi = lo - pad, hi + pad
    L, R, T, Bm = 52, 12, 14, 30
    x = lambda k: L + k * (w - L - R) / max(len(v) - 1, 1)
    y = lambda val: T + (hi - val) / (hi - lo) * (h - T - Bm)
    grid = ""
    for g in np.linspace(lo, hi, 4):
        grid += (f'<line x1="{L}" x2="{w - R}" y1="{y(g):.1f}" y2="{y(g):.1f}" stroke="#e6e8ee"/>'
                 f'<text x="{L - 8}" y="{y(g) + 4:.1f}" text-anchor="end" font-size="11" fill="#596173">{vn(g, 0)}</text>')
    pts = " ".join(f"{x(k):.1f},{y(val):.1f}" for k, val in enumerate(v))
    area = f"{x(0):.1f},{h - Bm} {pts} {x(len(v) - 1):.1f},{h - Bm}"
    last = len(v) - 1
    xl = "".join(f'<text x="{x(k):.1f}" y="{h - 10}" text-anchor="{"end" if k == last else "middle"}" '
                 f'font-size="11" fill="#596173">{escape(lb)}</text>'
                 for k, lb in enumerate(labels) if (k % 2 == 0 and k < last - 1) or k == last)
    dots = (f'<circle cx="{x(len(v) - 1):.1f}" cy="{y(v[-1]):.1f}" r="4.5" fill="{BLUE}"/>'
            f'<text x="{x(len(v) - 1) - 6:.1f}" y="{y(v[-1]) - 9:.1f}" text-anchor="end" font-size="12" '
            f'font-weight="700" fill="{BLUE}">{vn(v[-1], 0)} {unit}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg">{grid}'
            f'<polygon points="{area}" fill="{BLUE}" opacity=".08"/>'
            f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>{dots}{xl}</svg>')


def specs(r):
    out = []
    if pd.notna(r.get("so_phong_ngu_n")):
        out.append(f"<b>{int(r['so_phong_ngu_n'])}</b> PN")
    if pd.notna(r.get("so_wc_n")):
        out.append(f"<b>{int(r['so_wc_n'])}</b> WC")
    if pd.notna(r.get("dien_tich_m2")):
        out.append(f"<b>{vn(r['dien_tich_m2'], 0)}</b> m²")
    return " | ".join(out)


def listing_card(r, idx, quan_label):
    floors = r.get("tong_so_tang") if pd.notna(r.get("tong_so_tang")) else 2
    addr = ", ".join(x for x in [r.get("phuong"), quan_label] if isinstance(x, str) and x)
    html(f'<div class="z-lcard"><div class="img" style="background:linear-gradient(160deg,'
         f'{GRADS[idx % 6][0]},{GRADS[idx % 6][1]})">{price_status(r)}{house_svg(idx, 110, floors)}</div>'
         f'<div class="bd"><div class="p">{ty(r["gia_ban_ty"])}</div>'
         f'<div class="s">{specs(r)} · {escape(str(r["loai_hinh"]))}</div>'
         f'<div class="a" title="{escape(str(r["tieu_de"]))}">{escape(str(r["tieu_de"]))}</div>'
         f'<div class="a">📍 {escape(addr)}</div></div></div>')


def range_bar(low, high, est, price=None):
    """Thanh so sánh: dải giá ước tính (xanh nhạt), mốc ước tính (xanh), mốc giá đăng (đỏ/xanh lá)."""
    pts = [low, high, est] + ([price] if price else [])
    lo, hi = min(pts), max(pts)
    pad = (hi - lo) * 0.12 + 1e-9
    lo, hi = max(lo - pad, 0), hi + pad
    pos = lambda v: (v - lo) / (hi - lo) * 100
    mk = f'<span class="mk" style="left:{pos(est):.1f}%;background:{BLUE}"></span>'
    lab = (f'<span style="left:{pos(low):.1f}%">{vn(low, 2)}</span>'
           f'<span style="left:{pos(high):.1f}%">{vn(high, 2)}</span>')
    if price:
        inside = low <= price <= high
        color = "#11734b" if inside else "#d1242f"
        mk += f'<span class="mk" style="left:{pos(price):.1f}%;background:{color}"></span>'
        lab += (f'<span style="left:{pos(price):.1f}%;top:20px;color:{color};font-weight:600">'
                f'Giá đăng {vn(price, 2)}</span>')
    lab += f'<span style="left:{pos(est):.1f}%;top:20px;color:{BLUE};font-weight:600">Ước tính {vn(est, 2)}</span>'
    return (f'<div class="z-range"><div class="trk"><span class="band" style="left:{pos(low):.1f}%;'
            f'width:{pos(high) - pos(low):.1f}%"></span>{mk}</div><div class="lbl">{lab}</div></div>')


def kpis(items):
    cols = "".join(f'<div class="z-kpi">{escape(k)}<b>{v}</b></div>' for k, v in items)
    card(f'<div style="display:grid;grid-template-columns:repeat({len(items)},1fr);gap:12px">{cols}</div>')


def monthly(price_ty, down=0.3, rate=0.10, years=20):
    """Trả góp hằng tháng (triệu VND) — dư nợ giảm dần đều theo annuity."""
    loan = price_ty * 1000 * (1 - down)
    r, n = rate / 12, years * 12
    return loan * r / (1 - (1 + r) ** -n) if r > 0 else loan / n


def team_sidebar(team):
    st.sidebar.markdown(
        '<div class="z-team"><b>Nhóm thực hiện</b><br>'
        + "".join(f'{escape(m["ten"])}<br><span style="color:#596173">{escape(m["email"])}</span><br>'
                  for m in team) + "</div>", unsafe_allow_html=True)
