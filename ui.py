"""Thành phần giao diện (HTML/CSS) cho GUI Nhà Tốt — bố cục trang bất động sản:
thẻ nhà dạng lưới, trang chi tiết (giá, ước tính AI, thông tin nhanh), thanh so sánh giá."""
import os
import re
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

# Bộ màu theo nhận diện nhatot.com (cam chủ đạo, giá hồng đỏ, nền xám nhạt, font Reddit Sans)
ACCENT = "#fa6819"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Reddit+Sans:wght@400;500;600;700;800&display=swap');
:root{--ink:#222222;--muted:#8c8c8c;--muted2:#595959;--line:#e8e8e8;--soft:#f4f4f4;--accent:#fa6819;
  --accent2:#ff8800;--accent-soft:#fff1e3;--price:#f0325e;--link:#306bd9}
html,body,.stApp,.stApp p,.stApp label,.stApp input,.stApp textarea,.stApp button,.stApp li,.stApp h1,.stApp h2,
.stApp h3,.stApp td,.stApp th,.stApp span:not([data-testid="stIconMaterial"]){font-family:"Reddit Sans",sans-serif}
[data-testid="stIconMaterial"]{font-family:"Material Symbols Rounded" !important}
.stApp{background:#fff}
[class*="st-key-card_"]{background:#fff;border-radius:12px}
section[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line)}
.block-container,[data-testid="stMainBlockContainer"]{padding-top:4.2rem;max-width:1240px}
.stApp a{color:var(--link)}
.z-card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:14px}
.z-hero{border-radius:16px;padding:34px 34px;color:#fff;margin-bottom:16px;
  background:linear-gradient(115deg,#ff8800 0%,#fa6819 60%,#f2542d 100%)}
.z-hero h1{color:#fff;font-size:2.2rem;margin:0 0 6px 0;line-height:1.2;font-weight:800}
.z-hero p{color:#fff3e8;margin:0;font-size:1.02rem}
.z-photo{position:relative;border-radius:12px;overflow:hidden;background-size:cover;background-position:center}
.z-chart{position:relative;border-radius:12px;background:#fff;border:1px solid var(--line);padding:10px 12px 4px}
.z-chart .t{font-size:.85rem;font-weight:700;color:var(--ink)}
.z-tile{position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden}
.z-tile .cap,.z-photo .cap{position:absolute;left:12px;bottom:10px;background:rgba(34,34,34,.72);color:#fff;
  font-size:.8rem;font-weight:600;border-radius:6px;padding:3px 9px}
.z-status{display:inline-flex;align-items:center;gap:6px;font-size:.85rem;font-weight:600;color:var(--ink);
  background:#fff;border:1px solid var(--line);border-radius:999px;padding:3px 11px}
.z-dot{width:8px;height:8px;border-radius:50%;background:var(--accent);display:inline-block}
.z-price{font-size:2.3rem;font-weight:800;color:var(--price);line-height:1.1;margin-top:8px}
.z-addr{font-size:1.05rem;color:var(--ink);margin-top:4px}
.z-stats{display:flex;gap:28px;justify-content:flex-end}
.z-stats div{font-size:.95rem;color:var(--muted)}
.z-stats b{display:block;font-size:1.7rem;color:var(--ink);line-height:1.1}
.z-est{display:inline-flex;gap:14px;flex-wrap:wrap;background:var(--accent-soft);border-radius:10px;padding:8px 14px;
  margin-top:12px;font-size:.95rem;color:var(--ink)}
.z-est b{color:var(--ink)}
.z-facts{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}
.z-fact{background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:.92rem;color:var(--ink);
  display:flex;gap:10px;align-items:center}
.z-fact .i{font-size:1.2rem}
.z-fact small{display:block;color:var(--muted);font-size:.75rem}
.z-h2{font-size:1.35rem;font-weight:700;color:var(--ink);margin:18px 0 8px}
.z-txt{color:var(--ink);font-size:.97rem;line-height:1.6}
.z-txt summary{color:var(--link);cursor:pointer;font-weight:600;margin-top:6px}
.z-meta{color:var(--muted);font-size:.84rem}
.z-badge{display:inline-block;font-size:.75rem;font-weight:600;border-radius:4px;padding:2px 8px}
.z-good{background:#e3f5ec;color:#11734b}.z-hi{background:var(--accent-soft);color:#b04a0c}.z-mid{background:#eeeeee;color:var(--muted2)}
.z-bad{background:#fde8ee;color:#c4123f}
.z-lcard{background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden;margin-bottom:14px}
.z-lcard .img{height:160px;position:relative;background-size:cover;background-position:center}
.z-lcard .img .z-badge{position:absolute;left:10px;top:10px}
.z-lcard .bd{padding:10px 12px}
.z-lcard .p{font-size:1.3rem;font-weight:700;color:var(--price)}
.z-lcard .s{font-size:.9rem;color:var(--muted2);margin:2px 0}
.z-lcard .a{font-size:.85rem;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.z-lcard .t{font-size:.95rem;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.z-range{margin:16px 0 6px}
.z-range .trk{position:relative;height:10px;background:var(--soft);border-radius:999px}
.z-range .band{position:absolute;top:0;height:10px;background:#ffc9a3;border-radius:999px}
.z-range .mk{position:absolute;top:-7px;width:4px;height:24px;border-radius:2px;transform:translateX(-2px)}
.z-range .lbl{position:relative;height:38px;font-size:.8rem;color:var(--muted)}
.z-range .lbl span{position:absolute;transform:translateX(-50%);white-space:nowrap;top:6px;text-align:center}
.z-dl{display:grid;grid-template-columns:1fr 1fr;gap:4px 28px;font-size:.93rem;color:var(--ink)}
.z-dl div{display:flex;justify-content:space-between;border-bottom:1px solid var(--soft);padding:6px 0}
.z-dl span{color:var(--muted)}
.z-agent{display:flex;gap:12px;align-items:center}
.z-ava{width:48px;height:48px;border-radius:50%;background:var(--accent-soft);color:var(--accent);display:flex;
  align-items:center;justify-content:center;font-weight:700;flex:none}
.z-kpi{font-size:.85rem;color:var(--muted)}
.z-kpi b{display:block;font-size:1.4rem;color:var(--ink);margin-top:2px}
.z-team{font-size:.85rem;line-height:1.45}
.z-note{font-size:.75rem;color:var(--muted);line-height:1.4}
/* Phản hồi khi đang tải: thanh cam chạy ở đầu trang + nội dung cũ mờ đi */
.stApp[data-test-script-state="running"]::before{content:"";position:fixed;top:0;left:0;height:4px;width:40%;
  z-index:1000000;border-radius:0 4px 4px 0;background:linear-gradient(90deg,#ff8800,#fa6819);
  animation:z-load 0.9s ease-in-out infinite}
@keyframes z-load{0%{left:-40%}100%{left:100%}}
.stApp[data-test-script-state="running"] [data-stale="true"]{opacity:.45;transition:opacity .15s}
/* Cả thẻ nhà bấm được: nút trong suốt phủ kín thẻ + hiệu ứng nổi khi rê chuột, lún xuống khi nhấn */
[class*="st-key-lc_"]{position:relative}
[class*="st-key-lc_"] .z-lcard{transition:box-shadow .15s ease,transform .15s ease}
[class*="st-key-lc_"]:hover .z-lcard{box-shadow:0 6px 18px rgba(0,0,0,.12);transform:translateY(-2px)}
[class*="st-key-lc_"]:hover .z-lcard .t{color:var(--accent)}
[class*="st-key-lc_"]:active .z-lcard{transform:scale(.97);box-shadow:0 2px 6px rgba(250,104,25,.35);
  border-color:var(--accent)}
[class*="st-key-open_"]{position:absolute !important;inset:0;z-index:5;width:100% !important;margin:0 !important}
[class*="st-key-open_"] div[data-testid="stButton"],[class*="st-key-open_"] button{width:100% !important;
  height:100% !important;min-height:100%;opacity:0;cursor:pointer;padding:0;border:0}
/* Nút bo tròn kiểu viên thuốc như nhatot.com */
div[data-testid="stButton"] button,div[data-testid="stFormSubmitButton"] button,
div[data-testid="stDownloadButton"] button{border-radius:999px;font-weight:600}
div[data-testid="stButton"] button[kind="primary"],div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]{
  background:var(--accent);border-color:var(--accent);color:#fff}
div[data-testid="stButton"] button[kind="primary"]:hover{background:#e85a0e;border-color:#e85a0e}
div[data-testid="stButton"] button[kind="tertiary"]{color:var(--accent)}
div[data-testid="stTabs"] button[aria-selected="true"]{color:var(--accent)}
</style>
"""
# Ảnh minh hoạ theo loại hình (dữ liệu không có ảnh thật) — phục vụ tĩnh từ thư mục static/
HOUSE_IMG = {"Nhà ngõ, hẻm": "ngo_hem", "Nhà phố liền kề": "lien_ke",
             "Nhà mặt phố, mặt tiền": "mat_tien", "Nhà biệt thự": "biet_thu"}


_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _variants(slug):
    """Các ảnh của 1 loại hình: slug.jpg, slug_1.jpg, slug_2.jpg… (bỏ bản thu nhỏ _sm)."""
    files = sorted(f[:-4] for f in os.listdir(_STATIC)
                   if f.endswith(".jpg") and not f.endswith("_sm.jpg")
                   and (f == f"{slug}.jpg" or re.fullmatch(rf"{slug}_\d+\.jpg", f)))
    return files or [slug]


VARIANTS = {slug: _variants(slug) for slug in HOUSE_IMG.values()}


def house_img(loai, small=False, seed=0):
    """Ảnh theo loại hình; nhiều ảnh cùng loại thì xoay vòng theo mã tin (seed) — mỗi tin luôn cùng 1 ảnh."""
    opts = VARIANTS[HOUSE_IMG.get(loai, "ngo_hem")]
    return f'app/static/{opts[int(seed) % len(opts)]}{"_sm" if small else ""}.jpg'


def map_url(dia_chi):
    """Google Maps nhúng (không cần API key) theo địa chỉ cũ: bỏ phần '(Phường …, TP mới)'."""
    from urllib.parse import quote
    q = str(dia_chi).split(" (")[0]
    return f"https://maps.google.com/maps?q={quote(q)}&z=16&hl=vi&output=embed"


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
    dots = (f'<circle cx="{x(len(v) - 1):.1f}" cy="{y(v[-1]):.1f}" r="4.5" fill="{ACCENT}"/>'
            f'<text x="{x(len(v) - 1) - 6:.1f}" y="{y(v[-1]) - 9:.1f}" text-anchor="end" font-size="12" '
            f'font-weight="700" fill="{ACCENT}">{vn(v[-1], 0)} {unit}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg">{grid}'
            f'<polygon points="{area}" fill="{ACCENT}" opacity=".08"/>'
            f'<polyline points="{pts}" fill="none" stroke="{ACCENT}" stroke-width="2.5"/>{dots}{xl}</svg>')


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
    addr = ", ".join(x for x in [r.get("phuong"), quan_label] if isinstance(x, str) and x)
    html(f'<div class="z-lcard"><div class="img" style="background-image:url({house_img(r["loai_hinh"], True, idx)})">'
         f'{price_status(r)}</div>'
         # Thứ tự như thẻ tin trên nhatot.com: tiêu đề → phòng · loại hình → giá + đơn giá + m² → khu vực
         f'<div class="bd"><div class="t" title="{escape(str(r["tieu_de"]))}">{escape(str(r["tieu_de"]))}</div>'
         f'<div class="s">{specs(r)} · {escape(str(r["loai_hinh"]))}</div>'
         f'<div><span class="p">{ty(r["gia_ban_ty"])}</span>'
         f'<span class="z-meta" style="margin-left:8px">{vn(r["don_gia_trm2"], 0)} tr/m²</span></div>'
         f'<div class="a">📍 {escape(addr)}</div></div></div>')


def range_bar(low, high, est, price=None):
    """Thanh so sánh: dải giá ước tính (xanh nhạt), mốc ước tính (xanh), mốc giá đăng (đỏ/xanh lá)."""
    pts = [low, high, est] + ([price] if price else [])
    lo, hi = min(pts), max(pts)
    pad = (hi - lo) * 0.12 + 1e-9
    lo, hi = max(lo - pad, 0), hi + pad
    pos = lambda v: (v - lo) / (hi - lo) * 100
    mk = f'<span class="mk" style="left:{pos(est):.1f}%;background:{ACCENT}"></span>'
    lab = (f'<span style="left:{pos(low):.1f}%">{vn(low, 2)}</span>'
           f'<span style="left:{pos(high):.1f}%">{vn(high, 2)}</span>')
    if price:
        inside = low <= price <= high
        color = "#11734b" if inside else "#d1242f"
        mk += f'<span class="mk" style="left:{pos(price):.1f}%;background:{color}"></span>'
        lab += (f'<span style="left:{pos(price):.1f}%;top:20px;color:{color};font-weight:600">'
                f'Giá đăng {vn(price, 2)}</span>')
    lab += f'<span style="left:{pos(est):.1f}%;top:20px;color:{ACCENT};font-weight:600">Ước tính {vn(est, 2)}</span>'
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
