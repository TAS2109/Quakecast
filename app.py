# -*- coding: utf-8 -*-
"""
地震 強震モニタ風シミュレーター

Scratch作品「【複数対応】地震の到達時間シミュレーション v1.5.0β」の
震度計算ロジックを参考に、Pythonへ移植したWebアプリ。

  ・震源（地図クリック）・マグニチュード・深さを指定
  ・全国1748観測点のリアルタイム震度が、P波・S波の伝播とともに
    広がっていく様子を強震モニタ風にアニメーション表示

構成
  app.py            … 計算エンジン + API + フロントエンド(HTML/CSS/JS) のすべて
  data/stations.csv … 観測点（経度・緯度・地盤増幅ARV・地盤深さ・所属区域）  [不変データ]
  data/areas.csv    … 区域名                                              [不変データ]
  data/jma2001.json … JMA2001走時表の近似式係数                            [不変データ]

起動:  python app.py        (開発)
       gunicorn app:app     (本番 / Render)
"""
import base64
import csv
import gzip
import json
import math
import os
from pathlib import Path

import numpy as np
from flask import Flask, Response, request

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

app = Flask(__name__)


# =====================================================================
# 1. データ読み込み（不変データ）
# =====================================================================
def _load_data():
    lon, lat, arv, dep, area = [], [], [], [], []
    with open(DATA_DIR / "stations.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lon.append(float(row["lon"]))
            lat.append(float(row["lat"]))
            # ARVが欠損（NA）の観測点は、データ中の最小値0.1を採用
            arv.append(float(row["arv"]) if row["arv"].strip() else 0.1)
            dep.append(float(row["dep"]))
            area.append(int(row["area"]))
    names = {}
    with open(DATA_DIR / "areas.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            names[int(row["idx"])] = row["name"]
    with open(DATA_DIR / "jma2001.json", encoding="utf-8") as f:
        jma = np.array(json.load(f), dtype=np.float64)
    return (
        np.array(lon), np.array(lat), np.array(arv), np.array(dep),
        np.array(area, dtype=np.int64), names, jma,
    )


ST_LON, ST_LAT, ST_ARV, ST_DEP, ST_AREA, AREA_NAMES, JMA = _load_data()
N_ST = len(ST_LON)


# =====================================================================
# 2. 計算エンジン
# =====================================================================
def hubeny_km(lat0, lon0, lat, lon):
    """2点間の距離[km]（ヒュベニの公式。Scratch版「震度計算」と同じ式）"""
    e2 = 0.00669438
    phim = (lat0 + lat) / 2.0
    w = 1.0 - e2 * np.sin(np.radians(phim)) ** 2
    m_r = 6378137.0 * (1.0 - e2) / (w * np.sqrt(w))
    n_r = 6378137.0 / np.sqrt(w)
    dy = np.radians(lat0 - lat) * m_r
    dx = np.radians(lon0 - lon) * n_r * np.cos(np.radians(phim))
    return np.sqrt(dx * dx + dy * dy) / 1000.0


def wave_epicentral_km(t, depth_p, is_p):
    """
    JMA2001走時表の近似式から、経過時間t[秒]における波の到達距離（震央距離[km]）を求める。
    Scratch版「JMA2001距離近似」の移植。
      hypocentral_distance = Σ_{i=1..6} c_i * t^i
      epicentral_distance  = sqrt(hypo^2 - depth^2)
    depth_p はScratch版と同じく「震源の深さ-10km」を渡す（行選択が round(depth_p/10)+1 のため）。
    """
    row = int(math.floor(depth_p / 10.0 + 0.5)) + 1
    row = min(max(row, 0), 70)
    base = 6 * row + (0 if is_p else 426)
    coef = JMA[base:base + 6]
    t = np.asarray(t, dtype=np.float64)
    hypo = np.zeros_like(t)
    for i in range(6):
        hypo += coef[i] * t ** (i + 1)
    hypo = np.maximum(hypo, 0.0)
    return np.sqrt(np.maximum(hypo * hypo - depth_p * depth_p, 0.0))


def sanitize_p_curve(tg, p_raw):
    """
    P波の近似式は約300秒（震央距離2500km前後）を超えると発散するため、
    速度が非現実的（>14km/s または負）になった時点以降は、直前の速度で直線外挿する。
    """
    v = np.gradient(p_raw, tg)
    bad = np.where((tg > 20.0) & ((v > 14.0) | (v < 0.0)))[0]
    if len(bad) == 0:
        return np.maximum.accumulate(p_raw)
    i = int(bad[0])
    slope = float(np.clip(v[max(i - 20, 0)], 8.0, 12.0))
    out = p_raw.copy()
    out[i:] = p_raw[i - 1] + slope * (tg[i:] - tg[i - 1])
    return np.maximum.accumulate(out)


def shindo_class(x):
    """計測震度 → (震度階級の添字0..9, 表示名)。気象庁の震度階級しきい値"""
    th = [0.5, 1.5, 2.5, 3.5, 4.5, 5.0, 5.5, 6.0, 6.5]
    labels = ["0", "1", "2", "3", "4", "5弱", "5強", "6弱", "6強", "7"]
    k = 0
    for i, v in enumerate(th, start=1):
        if x >= v:
            k = i
    return k, labels[k]


def final_intensity(mag, depth, lat0, lon0):
    """
    各観測点の最終的な計測震度（Scratch版 震度計算 の非予想側の式）を返す。
    戻り値: (m01, 震央距離km, depth_p)
    """
    # Scratch版の「震度計算処理」と同じ入力補正
    #   M引数 = M - (7.1-2.2),  深さ引数 = 深さ - 10
    m_arg = mag - (7.1 - 2.2)
    depth_p = depth - 10.0

    dist = hubeny_km(lat0, lon0, ST_LAT, ST_LON)          # 震央距離
    x_hypo = np.sqrt(dist ** 2 + depth_p ** 2)            # 震源距離

    area_a = 10.0 ** (0.91 * m_arg - 3.49)                # 断層面積 [km2]
    slip_d = 10.0 ** (0.69 * m_arg - 4.8)                 # 平均すべり量 [m]
    m0 = 3.0e10 * (area_a * 1.0e6) * slip_d               # 地震モーメント [Nm]
    mw = (2.0 / 3.0) * math.log10(m0) - 6.07              # Mw再計算

    log_pgv600 = (
        0.58 * mw + 0.0038 * depth_p + 0.0
        - np.log10(x_hypo + 0.0028 * 10.0 ** (0.5 * mw))
        - 0.002 * x_hypo
        - 1.29
    )
    pgv600 = 10.0 ** log_pgv600
    pgv = pgv600 * ST_ARV * (4000.0 - ST_DEP)             # 地盤増幅を考慮
    m01 = 2.68 + 1.72 * np.log10(pgv)                     # PGV → 計測震度
    return m01, dist, depth_p


def realtime_frames(m01, kyori, dep_p, tf, p_curve, s_curve, tatsu):
    """
    強震モニタ風「リアルタイム震度」の時間変化。
    Scratch版「リアルタイム震度計さん」の移植（NumPyでベクトル化）。
      m01     : 最終震度（N,）
      kyori   : 震央距離（N,）
      dep_p   : 深さ-10
      tf      : フレーム時刻（F,）
      p_curve : フレームごとのP波到達距離（F,）
      s_curve : フレームごとのS波到達距離（F,）
      tatsu   : 観測点ごとのS波到達時刻（N,）  未到達はinf
    戻り値: (F, N) のリアルタイム震度
    """
    bai = ((1.0 + ST_ARV) / 2.2) / 2.0
    bai_dep = (ST_DEP + 1200.0) / 600.0
    # 深さ補正項。Scratch版の係数は浅い地震向けで、深さ約56km以上だと符号が反転して
    # 揺れが立ち上がらなくなるため、上限30で頭打ちにする（浅い地震の結果は不変）。
    dterm = min(max(dep_p - 10.0, 0.0), 30.0)
    k2 = np.maximum(np.minimum(kyori, 300.0), 0.5)

    base = m01 - (3.5 + kyori / 150.0) - dterm / 52.0
    kp = (-0.006 + dterm / 2000.0) - bai_dep / 300.0
    cap_s = m01 - (1.5 + kyori / 150.0) - dterm / 52.0
    cap_s44 = np.minimum(cap_s, 4.4)
    s0 = np.minimum(m01 - (1.5 + k2 / 150.0) - dterm / 52.0, 4.6)
    ks = ((-0.055 + dterm / 1800.0) - bai_dep / 300.0) * (0.1 + (92.0 / k2) / 12.0)
    decay_rate = 0.005 * (1.4 + (3.5 / bai) / 4.0)
    floor_thr = -3.3 + (1.0 + bai * 3.0) / 4.0
    floor_val = -3.6 + (1.0 + bai * 3.0) / 4.0

    out = np.empty((len(tf), N_ST), dtype=np.float32)
    chunk = 100
    with np.errstate(invalid="ignore", over="ignore"):
        for a in range(0, len(tf), chunk):
            b = min(a + chunk, len(tf))
            zp = kyori[None, :] - p_curve[a:b, None]      # P波の残り距離（正=未到達）
            zs = kyori[None, :] - s_curve[a:b, None]      # S波の残り距離（正=未到達）
            v = base[None, :] + zp * kp[None, :] * 2.0
            v_before = np.where(v > cap_s[None, :], cap_s44[None, :], v)
            v_after = s0[None, :] + zs * ks[None, :]
            v = np.where(zs > 0, v_before, v_after)
            v = np.where(zp > 0, -7.0, v)

            over = v > m01[None, :]
            tt = tf[a:b, None] - (tatsu[None, :] + 60.0)  # S波到達60秒後から減衰
            dec = np.where(tt > 0, tt * decay_rate[None, :], 0.0)
            v_over = np.where(dec > 0, np.maximum(m01[None, :] - dec, -10.0), m01[None, :])
            v = np.where(over, v_over, v)

            v = np.where(floor_thr[None, :] > v, floor_val[None, :], v)
            out[a:b] = v
    return out


def simulate(lat0, lon0, mag, depth, dt=1.0):
    m01, dist, dep_p = final_intensity(mag, depth, lat0, lon0)

    # 細かい時間軸でP/S波の到達距離曲線を作る（単調増加に整える）
    tg = np.arange(0.0, 600.0 + 1e-9, 0.05)
    p_raw = wave_epicentral_km(tg, dep_p, True)
    s_raw = wave_epicentral_km(tg, dep_p, False)
    # 近似式の有効範囲：S波曲線が減少に転じる直前まで
    dec_idx = np.where(np.diff(s_raw) < 0)[0]
    dec_idx = dec_idx[tg[dec_idx] > 5.0]
    t_valid = float(tg[dec_idx[0]]) if len(dec_idx) else 600.0
    p_curve_f = sanitize_p_curve(tg, p_raw)
    s_curve_f = np.maximum.accumulate(s_raw)

    # 各観測点のP波・S波到達時刻
    tp = np.interp(dist, p_curve_f, tg, right=np.inf)
    tatsu = np.interp(dist, s_curve_f, tg, right=np.inf)

    # シミュレーション時間：最遠観測点にS波が届いてから90秒（上限540秒）
    finite = tatsu[np.isfinite(tatsu)]
    t_far = float(finite.max()) if len(finite) else t_valid
    t_end = float(np.clip(t_far + 90.0, 60.0, min(540.0, t_valid)))
    nf = int(t_end / dt) + 1
    tf = np.arange(nf) * dt

    p_curve = np.interp(tf, tg, p_curve_f)
    s_curve = np.interp(tf, tg, s_curve_f)

    rt = realtime_frames(m01, dist, dep_p, tf, p_curve, s_curve, tatsu)
    q = np.clip(np.round(rt * 10.0), -127, 127).astype(np.int8)

    # 区域別の最大震度
    area_max = {}
    for a, v in zip(ST_AREA, m01):
        if a not in area_max or v > area_max[a]:
            area_max[a] = float(v)
    areas = []
    for a, v in sorted(area_max.items(), key=lambda kv: -kv[1]):
        cls, label = shindo_class(v)
        if cls < 1:
            continue
        areas.append({"name": AREA_NAMES.get(a, "区域%d" % a),
                      "v": round(v, 2), "cls": cls, "label": label})
        if len(areas) >= 20:
            break

    vmax = float(m01.max())
    cls, label = shindo_class(vmax)

    def _clean(a, nd=1):
        return [None if not math.isfinite(x) else round(float(x), nd) for x in a]

    return {
        "dt": dt,
        "nf": nf,
        "ns": N_ST,
        "q": base64.b64encode(q.tobytes()).decode("ascii"),
        "P": _clean(p_curve, 2),
        "S": _clean(s_curve, 2),
        "tp": _clean(tp, 1),
        "ts": _clean(tatsu, 1),
        "final": _clean(m01, 2),
        "max": round(vmax, 2),
        "maxCls": cls,
        "maxLabel": label,
        "areas": areas,
        "epi": {"lat": lat0, "lon": lon0, "mag": mag, "depth": depth},
        "tEnd": t_end,
    }


# =====================================================================
# 3. API
# =====================================================================
def _json_response(obj, status=200):
    body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    headers = {}
    if "gzip" in request.headers.get("Accept-Encoding", "") and len(body) > 2048:
        body = gzip.compress(body, compresslevel=6)
        headers["Content-Encoding"] = "gzip"
    headers["Cache-Control"] = "no-store"
    return Response(body, status=status, mimetype="application/json", headers=headers)


_STATIONS_CACHE = None


@app.get("/api/stations")
def api_stations():
    global _STATIONS_CACHE
    if _STATIONS_CACHE is None:
        _STATIONS_CACHE = {
            "lon": [round(float(x), 4) for x in ST_LON],
            "lat": [round(float(x), 4) for x in ST_LAT],
        }
    return _json_response(_STATIONS_CACHE)


@app.post("/api/simulate")
def api_simulate():
    try:
        j = request.get_json(force=True, silent=False) or {}
        lat = float(j["lat"])
        lon = float(j["lon"])
        mag = float(j["mag"])
        depth = float(j["depth"])
    except Exception:
        return _json_response({"error": "入力が不正です"}, 400)
    if not (10.0 <= lat <= 55.0 and 110.0 <= lon <= 165.0):
        return _json_response({"error": "震源は日本周辺（緯度10〜55, 経度110〜165）で指定してください"}, 400)
    mag = min(max(mag, 0.0), 9.9)
    depth = min(max(depth, 0.0), 699.0)
    return _json_response(simulate(lat, lon, mag, depth))


@app.get("/healthz")
def healthz():
    return {"status": "ok", "stations": N_ST}


@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


# =====================================================================
# 4. フロントエンド（HTML / CSS / JavaScript）
# =====================================================================
INDEX_HTML = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>地震 強震モニタ風シミュレーター</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  :root{--bg:#10141c;--panel:#161c27ee;--line:#2b3547;--txt:#e6ebf5;--sub:#93a0b8;--acc:#3b9cff;}
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--txt);
    font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Noto Sans JP","Yu Gothic UI",sans-serif;}
  #map{position:absolute;inset:0}
  .leaflet-container{background:#0b0f16}
  #panel{position:absolute;z-index:1100;top:10px;left:10px;width:310px;max-height:calc(100% - 20px);
    overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:12px;
    padding:14px;backdrop-filter:blur(6px)}
  #panel h1{font-size:15px;margin:0 0 10px;letter-spacing:.02em}
  #panel h2{font-size:12px;margin:14px 0 6px;color:var(--sub);font-weight:600}
  label{display:block;font-size:12px;color:var(--sub);margin:8px 0 3px}
  .row{display:flex;gap:8px}
  .row>div{flex:1}
  input[type=number],select{width:100%;padding:7px 8px;border-radius:7px;border:1px solid var(--line);
    background:#0e131c;color:var(--txt);font-size:14px}
  input[type=range]{width:100%}
  button{cursor:pointer;border:1px solid var(--line);background:#1d2636;color:var(--txt);
    border-radius:8px;padding:8px 12px;font-size:14px}
  button:hover{background:#26324a}
  button.primary{background:var(--acc);border-color:var(--acc);color:#04101f;font-weight:700;width:100%;margin-top:12px;padding:10px}
  button.primary:disabled{opacity:.5;cursor:wait}
  .presets{display:flex;flex-wrap:wrap;gap:6px}
  .presets button{font-size:12px;padding:5px 8px}
  .hint{font-size:11.5px;color:var(--sub);line-height:1.5;margin-top:6px}
  #status{font-size:12px;color:var(--sub);margin-top:8px;min-height:16px}
  #toggle{position:absolute;z-index:1200;top:10px;left:10px;display:none;padding:7px 10px}
  table{width:100%;border-collapse:collapse;font-size:12.5px}
  td{padding:3px 4px;border-bottom:1px solid #222b3b}
  td.num{text-align:right;white-space:nowrap}
  .chip{display:inline-block;min-width:34px;text-align:center;border-radius:5px;padding:1px 6px;
    font-weight:700;font-size:12px;color:#fff;text-shadow:0 0 2px #000a}
  #maxbox{display:flex;align-items:center;gap:10px;margin-top:4px}
  #maxbox .big{font-size:26px;font-weight:800;min-width:64px;text-align:center;border-radius:8px;padding:2px 8px}
  #hud{position:absolute;z-index:1000;top:10px;left:50%;transform:translateX(-50%);
    background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:7px 14px;
    font-size:13px;display:flex;gap:16px;align-items:center;white-space:nowrap}
  #hud b{font-size:16px;font-variant-numeric:tabular-nums}
  #hud .lbl{color:var(--sub);font-size:11px;margin-right:4px}
  #bar{position:absolute;z-index:1000;left:50%;bottom:26px;transform:translateX(-50%);
    width:min(760px,calc(100% - 20px));background:var(--panel);border:1px solid var(--line);
    border-radius:12px;padding:10px 12px}
  #ctl{display:flex;gap:8px;align-items:center}
  #ctl button{padding:6px 10px;min-width:40px}
  #ctl input[type=range]{flex:1}
  #ctl select{width:auto;padding:5px 6px;font-size:13px}
  #tlabel{font-size:13px;min-width:64px;text-align:right;font-variant-numeric:tabular-nums}
  #legend{margin-top:8px}
  #legend canvas{width:100%;height:12px;border-radius:6px;display:block}
  #ticks{display:flex;justify-content:space-between;font-size:10.5px;color:var(--sub);margin-top:2px}
  .epi{color:#ff4d4d;font-size:26px;font-weight:900;text-shadow:0 0 4px #000,0 0 8px #000;
    line-height:26px;text-align:center}
  .foot{font-size:10.5px;color:var(--sub);line-height:1.55;margin-top:14px;border-top:1px solid var(--line);padding-top:8px}
  @media (max-width:700px){
    #panel{width:calc(100% - 20px);max-height:62%;top:52px;display:none}
    #panel.open{display:block}
    #toggle{display:block}
    #hud{top:10px;left:auto;right:60px;transform:none;padding:5px 9px;gap:8px;font-size:12px}
    #hud .lbl{display:none}
  }
</style>
</head>
<body>
<div id="map"></div>
<button id="toggle">⚙ 設定</button>

<div id="panel" class="open">
  <h1>地震 強震モニタ風シミュレーター</h1>

  <h2>① 震源（地図をクリックで設定）</h2>
  <div class="row">
    <div><label>緯度</label><input id="lat" type="number" step="0.01" value="35.68"></div>
    <div><label>経度</label><input id="lon" type="number" step="0.01" value="139.77"></div>
  </div>

  <h2>② 規模・深さ</h2>
  <label>マグニチュード <b id="magv">7.0</b></label>
  <input id="mag" type="range" min="0" max="9.9" step="0.1" value="7.0">
  <label>震源の深さ（km）</label>
  <input id="depth" type="number" min="0" max="699" step="1" value="20">

  <h2>プリセット</h2>
  <div class="presets" id="presets"></div>

  <button id="run" class="primary">▶ シミュレーション実行</button>
  <div id="status"></div>

  <div id="result" style="display:none">
    <h2>予想される最大震度</h2>
    <div id="maxbox"><span class="big" id="maxbig"></span><span id="maxsub" class="hint" style="margin:0"></span></div>
    <h2>区域別の最大震度（上位）</h2>
    <table id="areas"></table>
  </div>

  <div class="foot">
    距離減衰式による簡易推定です。実際の強震モニタ（実観測波形）や気象庁の発表とは異なります。防災・避難判断には使用しないでください。<br>
    Scratch作品「【複数対応】地震の到達時間シミュレーション v1.5.0β」のロジックを参考に作成。<br>
    地図: Tiles © Esri
  </div>
</div>

<div id="hud">
  <span><span class="lbl">経過</span><b id="hT">0.0</b> 秒</span>
  <span><span class="lbl">P波</span><b id="hP">-</b> km</span>
  <span><span class="lbl">S波</span><b id="hS">-</b> km</span>
  <span><span class="lbl">現在の最大</span><b id="hM">-</b></span>
</div>

<div id="bar">
  <div id="ctl">
    <button id="restart" title="最初に戻る">⏮</button>
    <button id="play" title="再生/一時停止">▶</button>
    <input id="seek" type="range" min="0" max="0" step="1" value="0" disabled>
    <span id="tlabel">0.0 s</span>
    <select id="speed" title="再生速度">
      <option value="0.5">×0.5</option><option value="1">×1</option>
      <option value="2" selected>×2</option><option value="5">×5</option><option value="10">×10</option>
    </select>
  </div>
  <div id="legend"><canvas id="lgc" width="600" height="12"></canvas>
    <div id="ticks"><span>-3</span><span>-2</span><span>-1</span><span>0</span><span>1</span><span>2</span><span>3</span><span>4</span><span>5</span><span>6</span><span>7</span></div>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
(function(){
'use strict';
const $ = id => document.getElementById(id);

/* ---------- 色（Scratch版の震度階級色を基準に、連続グラデーション化） ---------- */
const CLASS_RGB = [[105,105,113],[44,142,176],[76,208,165],[246,202,81],[254,153,55],
                   [228,42,24],[195,26,29],[165,12,107],[146,11,122],[95,13,161]];
const CLASS_LABEL = ['0','1','2','3','4','5弱','5強','6弱','6強','7'];
const STOPS = [[-3.5,10,14,36],[-3.0,20,30,70],[-2.0,24,52,120],[-1.0,28,86,176],[0.0,34,124,222],
  [0.5,40,168,226],[1.0,44,190,190],[2.0,76,208,165],[3.0,246,202,81],[4.0,254,153,55],
  [4.75,228,42,24],[5.25,195,26,29],[5.75,165,12,107],[6.25,146,11,122],[6.75,95,13,161],[7.5,70,10,150]];
const LUT_MIN=-3.5, LUT_STEP=0.1, LUT_N=111;
function colorAt(v){
  if(v<=STOPS[0][0]) return STOPS[0].slice(1);
  for(let i=1;i<STOPS.length;i++){
    if(v<=STOPS[i][0]){
      const a=STOPS[i-1], b=STOPS[i], k=(v-a[0])/(b[0]-a[0]);
      return [a[1]+(b[1]-a[1])*k, a[2]+(b[2]-a[2])*k, a[3]+(b[3]-a[3])*k].map(Math.round);
    }
  }
  return STOPS[STOPS.length-1].slice(1);
}
const LUT=[]; for(let k=0;k<LUT_N;k++){ const c=colorAt(LUT_MIN+k*LUT_STEP); LUT.push('rgb('+c[0]+','+c[1]+','+c[2]+')'); }
function shindoCls(x){
  const th=[0.5,1.5,2.5,3.5,4.5,5.0,5.5,6.0,6.5]; let k=0;
  for(let i=0;i<th.length;i++) if(x>=th[i]) k=i+1; return k;
}
function clsColor(k){ const c=CLASS_RGB[k]; return 'rgb('+c[0]+','+c[1]+','+c[2]+')'; }

/* 凡例 */
(function(){
  const cv=$('lgc'), g=cv.getContext('2d'), w=cv.width;
  for(let x=0;x<w;x++){ const v=-3+(x/(w-1))*10; g.fillStyle=LUT[Math.round((v-LUT_MIN)/LUT_STEP)]; g.fillRect(x,0,1,cv.height); }
})();

/* ---------- 地図 ---------- */
const map = L.map('map',{zoomSnap:0.25,minZoom:4,maxZoom:11,worldCopyJump:false}).setView([36.3,137.6],5.5);
const ESRI='https://server.arcgisonline.com/ArcGIS/rest/services/';
const esri=(path,attr,maxNative)=>L.tileLayer(ESRI+path+'/MapServer/tile/{z}/{y}/{x}',
  {attribution:attr,maxNativeZoom:maxNative||16,maxZoom:19});
const ATTR_BASE='Tiles © Esri — Esri, DeLorme, NAVTEQ, USGS, METI, Esri Japan and the GIS User Community';
const ATTR_IMG='Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community';
const dark = L.layerGroup([esri('Canvas/World_Dark_Gray_Base',ATTR_BASE),esri('Canvas/World_Dark_Gray_Reference','')]);
const light = L.layerGroup([esri('Canvas/World_Light_Gray_Base',ATTR_BASE),esri('Canvas/World_Light_Gray_Reference','')]);
const imagery = L.layerGroup([esri('World_Imagery',ATTR_IMG,19),esri('Reference/World_Boundaries_and_Places','',16)]);
dark.addTo(map);
L.control.layers({'ダーク（Esri）':dark,'ライト（Esri）':light,'衛星写真（Esri）':imagery},null,{position:'topright'}).addTo(map);

const epiIcon = L.divIcon({className:'',html:'<div class="epi">✕</div>',iconSize:[26,26],iconAnchor:[13,13]});
let epiMarker=null;
function setEpi(lat,lon,pan){
  lat=Math.round(lat*100)/100; lon=Math.round(lon*100)/100;
  $('lat').value=lat.toFixed(2); $('lon').value=lon.toFixed(2);
  if(!epiMarker) epiMarker=L.marker([lat,lon],{icon:epiIcon,interactive:false,keyboard:false}).addTo(map);
  else epiMarker.setLatLng([lat,lon]);
  if(pan) map.panTo([lat,lon]);
}
map.on('click',e=>setEpi(e.latlng.lat,e.latlng.lng,false));
['lat','lon'].forEach(id=>$(id).addEventListener('change',()=>{
  const la=parseFloat($('lat').value), lo=parseFloat($('lon').value);
  if(isFinite(la)&&isFinite(lo)) setEpi(la,lo,false);
}));
setEpi(35.68,139.77,false);

/* ---------- 入力 ---------- */
$('mag').addEventListener('input',()=>{ $('magv').textContent=parseFloat($('mag').value).toFixed(1); });
const PRESETS=[
  {n:'阪神・淡路(1995)',lat:34.60,lon:135.04,m:7.3,d:16},
  {n:'東北沖(2011)',lat:38.10,lon:142.86,m:9.0,d:24},
  {n:'熊本(2016)',lat:32.74,lon:130.76,m:7.3,d:12},
  {n:'能登半島(2024)',lat:37.49,lon:137.27,m:7.6,d:15},
  {n:'南海トラフ(想定)',lat:33.00,lon:136.50,m:9.0,d:20},
  {n:'首都直下(想定)',lat:35.68,lon:139.77,m:7.3,d:30},
  {n:'深発地震の例',lat:37.0,lon:134.0,m:7.0,d:400}
];
PRESETS.forEach(p=>{
  const b=document.createElement('button'); b.textContent=p.n;
  b.onclick=()=>{ setEpi(p.lat,p.lon,true); $('mag').value=p.m; $('magv').textContent=p.m.toFixed(1); $('depth').value=p.d; };
  $('presets').appendChild(b);
});
$('toggle').onclick=()=>$('panel').classList.toggle('open');
if(window.innerWidth<=700) $('panel').classList.remove('open');

/* ---------- 観測点 ---------- */
let N=0, SLON=null, SLAT=null, PX=null, PY=null, VALS=null, viewDirty=true;
fetch('/api/stations').then(r=>r.json()).then(j=>{
  SLON=Float32Array.from(j.lon); SLAT=Float32Array.from(j.lat); N=SLON.length;
  PX=new Float32Array(N); PY=new Float32Array(N); VALS=new Float32Array(N).fill(-3.1); viewDirty=true;
});

/* ---------- キャンバス ---------- */
const cv=document.createElement('canvas');
cv.style.cssText='position:absolute;top:0;left:0;z-index:450;pointer-events:none';
map.getContainer().appendChild(cv);
const ctx=cv.getContext('2d');
function resizeCanvas(){
  const dpr=window.devicePixelRatio||1, s=map.getSize();
  cv.width=s.x*dpr; cv.height=s.y*dpr; cv.style.width=s.x+'px'; cv.style.height=s.y+'px';
  ctx.setTransform(dpr,0,0,dpr,0,0); viewDirty=true;
}
map.on('resize',resizeCanvas); map.on('move zoomend viewreset',()=>{viewDirty=true;}); resizeCanvas();

function updatePoints(){
  for(let i=0;i<N;i++){ const p=map.latLngToContainerPoint([SLAT[i],SLON[i]]); PX[i]=p.x; PY[i]=p.y; }
  viewDirty=false;
}
function circlePts(lat0,lon0,rkm,n){
  const R=6371.0, d=rkm/R, la=lat0*Math.PI/180, lo=lon0*Math.PI/180, pts=[];
  for(let i=0;i<=n;i++){
    const th=i/n*2*Math.PI;
    const lat=Math.asin(Math.sin(la)*Math.cos(d)+Math.cos(la)*Math.sin(d)*Math.cos(th));
    const lon=lo+Math.atan2(Math.sin(th)*Math.sin(d)*Math.cos(la),Math.cos(d)-Math.sin(la)*Math.sin(lat));
    pts.push(map.latLngToContainerPoint([lat*180/Math.PI,lon*180/Math.PI]));
  }
  return pts;
}
function drawCircle(rkm,style,width){
  if(!(rkm>0.5)) return;
  const pts=circlePts(sim.epi.lat,sim.epi.lon,rkm,180);
  ctx.beginPath(); ctx.moveTo(pts[0].x,pts[0].y);
  for(let i=1;i<pts.length;i++) ctx.lineTo(pts[i].x,pts[i].y);
  ctx.strokeStyle=style; ctx.lineWidth=width; ctx.stroke();
}
function frac(x){ return x-Math.floor(x); }

/* ---------- シミュレーション状態 ---------- */
let sim=null, Q=null, tcur=0, playing=false, lastTs=0;

function valuesAt(t){
  const nf=sim.nf; let f=t/sim.dt; if(f<0)f=0; if(f>nf-1)f=nf-1;
  const f0=Math.floor(f), f1=Math.min(f0+1,nf-1), a=f-f0, o0=f0*N, o1=f1*N;
  const tk=Math.floor(t*6);
  for(let i=0;i<N;i++){
    let v=(Q[o0+i]*(1-a)+Q[o1+i]*a)*0.1;
    if(v<-2.3) v+=(frac(Math.sin(i*12.9898+tk*78.233)*43758.5453)-0.5)*0.14;   // 微小ノイズの揺らぎ
    VALS[i]=v;
  }
}
function interp(arr,t){
  const f=Math.min(Math.max(t/sim.dt,0),arr.length-1), f0=Math.floor(f), f1=Math.min(f0+1,arr.length-1);
  return arr[f0]+(arr[f1]-arr[f0])*(f-f0);
}

function draw(ts){
  const s=map.getSize(); ctx.clearRect(0,0,s.x,s.y);
  if(!N) return;
  if(viewDirty) updatePoints();
  if(sim) valuesAt(tcur);
  else { const tk=Math.floor((ts||0)/160); for(let i=0;i<N;i++) VALS[i]=-3.1+(frac(Math.sin(i*12.9898+tk*78.233)*43758.5453)-0.5)*0.14; }

  const z=map.getZoom(), r0=Math.max(2.0,(z-3.6)*0.95);
  const buckets=new Array(LUT_N); for(let k=0;k<LUT_N;k++) buckets[k]=[];
  let vmax=-99;
  for(let i=0;i<N;i++){
    const v=VALS[i]; if(v>vmax) vmax=v;
    let k=Math.round((v-LUT_MIN)/LUT_STEP); if(k<0)k=0; if(k>=LUT_N)k=LUT_N-1;
    buckets[k].push(i);
  }
  for(let k=0;k<LUT_N;k++){
    const b=buckets[k]; if(!b.length) continue;
    const v=LUT_MIN+k*LUT_STEP;
    const r=r0*(v<0?0.85:v<2?1.0:v<4?1.2:v<5?1.45:1.75);
    ctx.fillStyle=LUT[k]; ctx.beginPath();
    for(let j=0;j<b.length;j++){ const i=b[j]; ctx.moveTo(PX[i]+r,PY[i]); ctx.arc(PX[i],PY[i],r,0,6.2832); }
    ctx.fill();
    if(v>=4.5){ ctx.strokeStyle='rgba(255,255,255,.55)'; ctx.lineWidth=0.8; ctx.stroke(); }
  }
  if(sim){
    drawCircle(interp(sim.P,tcur),'rgba(90,200,255,.95)',1.6);
    drawCircle(interp(sim.S,tcur),'rgba(255,95,95,.98)',2.2);
    $('hT').textContent=tcur.toFixed(1);
    $('hP').textContent=interp(sim.P,tcur).toFixed(0);
    $('hS').textContent=interp(sim.S,tcur).toFixed(0);
    const hm=$('hM'); const m=Math.max(vmax,-3);
    if(m<0.5){ hm.textContent='—'; hm.style.color='#93a0b8'; }
    else { hm.textContent=m.toFixed(1)+'（'+CLASS_LABEL[shindoCls(m)]+'）'; hm.style.color=clsColor(shindoCls(m)); }
  }
}

function tick(ts){
  if(sim && playing){
    const d=Math.min((ts-lastTs)/1000,0.1);
    tcur+=d*parseFloat($('speed').value);
    if(tcur>=sim.tEnd){ tcur=sim.tEnd; playing=false; $('play').textContent='▶'; }
    $('seek').value=Math.round(tcur*10); $('tlabel').textContent=tcur.toFixed(1)+' s';
  }
  lastTs=ts; draw(ts); requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

$('play').onclick=()=>{
  if(!sim) return;
  if(tcur>=sim.tEnd-0.01) tcur=0;
  playing=!playing; $('play').textContent=playing?'⏸':'▶';
};
$('restart').onclick=()=>{ if(!sim) return; tcur=0; $('seek').value=0; $('tlabel').textContent='0.0 s'; };
$('seek').addEventListener('input',()=>{ if(!sim) return; tcur=parseInt($('seek').value)/10; $('tlabel').textContent=tcur.toFixed(1)+' s'; });

/* ---------- 実行 ---------- */
$('run').onclick=async()=>{
  const body={lat:parseFloat($('lat').value),lon:parseFloat($('lon').value),
              mag:parseFloat($('mag').value),depth:parseFloat($('depth').value)};
  if(![body.lat,body.lon,body.mag,body.depth].every(isFinite)){ $('status').textContent='入力値を確認してください'; return; }
  $('run').disabled=true; $('status').textContent='計算中…'; playing=false;
  try{
    const r=await fetch('/api/simulate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();
    if(!r.ok){ $('status').textContent=j.error||'エラーが発生しました'; return; }
    const bin=atob(j.q), u=new Uint8Array(bin.length);
    for(let i=0;i<bin.length;i++) u[i]=bin.charCodeAt(i);
    Q=new Int8Array(u.buffer); sim=j; tcur=0;
    $('seek').disabled=false; $('seek').max=Math.round(j.tEnd*10); $('seek').value=0; $('tlabel').textContent='0.0 s';
    showResult(j);
    $('status').textContent='完了（'+j.nf+'フレーム）';
    playing=true; $('play').textContent='⏸';
    if(window.innerWidth<=700) $('panel').classList.remove('open');
  }catch(e){ $('status').textContent='通信エラー: '+e; }
  finally{ $('run').disabled=false; }
};

function showResult(j){
  $('result').style.display='block';
  const big=$('maxbig'); big.textContent=j.maxLabel; big.style.background=clsColor(j.maxCls);
  $('maxsub').textContent='最大計測震度 '+j.max.toFixed(1)+'（M'+j.epi.mag.toFixed(1)+' / 深さ'+j.epi.depth+'km）';
  const t=$('areas'); t.innerHTML='';
  if(!j.areas.length){ t.innerHTML='<tr><td>震度1以上の区域はありません</td></tr>'; return; }
  j.areas.forEach(a=>{
    const tr=document.createElement('tr');
    tr.innerHTML='<td>'+a.name+'</td><td class="num">'+a.v.toFixed(1)+'</td><td class="num"><span class="chip" style="background:'+clsColor(a.cls)+'">'+a.label+'</span></td>';
    t.appendChild(tr);
  });
}
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
