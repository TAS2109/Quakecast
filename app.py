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


# ---------------------------------------------------------------------
# 区域境界ポリゴン(data/areas.geojson)による観測点の区域判定（任意）
#   観測点の座標がどの区域ポリゴンに入るかで ST_AREA を決め直す。
#   ポリゴン外（海上・簡略化で欠けた沿岸など）は、15km以内の最寄り区域、
#   それも無ければ stations.csv の値を使う。区域名が areas.csv とあまり
#   一致しない（<80%）場合は、誤判定を避けるため何も変更しない。
# ---------------------------------------------------------------------
def _apply_geojson_areas():
    import unicodedata
    path = DATA_DIR / "areas.geojson"
    if not path.exists():
        return
    try:
        with open(path, encoding="utf-8") as f:
            gj = json.load(f)
    except Exception as e:
        print("[areas.geojson] 読み込み失敗:", e)
        return

    def norm(s):
        return unicodedata.normalize("NFKC", str(s or "")).replace(" ", "").replace("　", "")

    by_name = {norm(v): k for k, v in AREA_NAMES.items()}
    keys = ("name", "Name", "NAME", "area_name", "AREA_NAME", "regionName", "Area", "area")
    feats = []                       # (area_id, [ring ndarray(k,2)], bbox)
    for ft in gj.get("features", []):
        pr = ft.get("properties") or {}
        aid = None
        for k in keys:
            if k in pr and norm(pr[k]) in by_name:
                aid = by_name[norm(pr[k])]
                break
        g = ft.get("geometry") or {}
        if aid is None or not g:
            continue
        polys = g["coordinates"] if g.get("type") == "MultiPolygon" else (
            [g["coordinates"]] if g.get("type") == "Polygon" else [])
        rings = [np.asarray(r, dtype=np.float64)[:, :2] for poly in polys for r in poly if len(r) >= 3]
        if rings:
            allp = np.concatenate(rings)
            feats.append((aid, rings, (allp[:, 0].min(), allp[:, 0].max(), allp[:, 1].min(), allp[:, 1].max())))
    n_total = len(gj.get("features", []))
    del gj
    if len(feats) < 0.8 * len(AREA_NAMES):
        print("[areas.geojson] 区域名の一致が少ないため（%d/%d件, areas.csv=%d区域）観測点の区域は変更しません"
              % (len(feats), n_total, len(AREA_NAMES)))
        return

    def inside(px, py, ring):
        x, y = ring[:, 0], ring[:, 1]
        x2, y2 = np.roll(x, -1), np.roll(y, -1)
        dy = np.where(y2 == y, 1e-12, y2 - y)
        res = np.zeros(len(px), dtype=bool)
        for a in range(0, len(px), 64):
            qx, qy = px[a:a + 64, None], py[a:a + 64, None]
            cond = (y[None, :] > qy) != (y2[None, :] > qy)
            xint = x[None, :] + (qy - y[None, :]) * ((x2 - x) / dy)[None, :]
            res[a:a + 64] = (np.sum(cond & (qx < xint), axis=1) % 2) == 1
        return res

    new_area = np.full(N_ST, -1, dtype=np.int64)
    for aid, rings, (x0, x1, y0, y1) in feats:
        cand = np.where((ST_LON >= x0) & (ST_LON <= x1) & (ST_LAT >= y0) & (ST_LAT <= y1)
                        & (new_area < 0))[0]
        if not len(cand):
            continue
        ins = np.zeros(len(cand), dtype=bool)
        for r in rings:                               # 穴を含め偶奇則で判定
            ins ^= inside(ST_LON[cand], ST_LAT[cand], r)
        new_area[cand[ins]] = aid
    n_in = int((new_area >= 0).sum())

    # ポリゴン外の観測点: 15km以内の最寄り区域
    left = np.where(new_area < 0)[0]
    n_near = 0
    if len(left):
        vx = np.concatenate([r[:, 0] for _, rs, _ in feats for r in rs])
        vy = np.concatenate([r[:, 1] for _, rs, _ in feats for r in rs])
        vid = np.concatenate([np.full(len(r), aid) for aid, rs, _ in feats for r in rs])
        for i in left:
            kx = 111.32 * math.cos(math.radians(ST_LAT[i]))
            d = np.hypot((vx - ST_LON[i]) * kx, (vy - ST_LAT[i]) * 110.57)
            j = int(np.argmin(d))
            if d[j] <= 15.0:
                new_area[i] = vid[j]
                n_near += 1
    ok = new_area >= 0
    changed = int((new_area[ok] != ST_AREA[ok]).sum())
    ST_AREA[ok] = new_area[ok]
    print("[areas.geojson] 区域一致 %d/%d件 / ポリゴン内 %d点・最寄り %d点・csv据え置き %d点 / 区域を変更 %d点"
          % (len(feats), n_total, n_in, n_near, int((~ok).sum()), changed))


_apply_geojson_areas()


# ---------------------------------------------------------------------
# 震央地名（気象庁の震源地名）: data/epicenters.geojson のポリゴンから緯度経度→地名
#   陸域（石川県能登地方など）・海域（能登半島沖など）・海外の地域を含む。
#   無ければ None を返し、呼び出し側が簡易的な地名にフォールバックする。
# ---------------------------------------------------------------------
def _load_epicenters():
    path = DATA_DIR / "epicenters.geojson"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            gj = json.load(f)
    except Exception as e:
        print("[epicenters.geojson] 読み込み失敗:", e)
        return []
    out = []
    for ft in gj.get("features", []):
        nm = (ft.get("properties") or {}).get("name")
        g = ft.get("geometry") or {}
        polys = g.get("coordinates", []) if g.get("type") == "MultiPolygon" else (
            [g.get("coordinates", [])] if g.get("type") == "Polygon" else [])
        rings = [np.asarray(r, dtype=np.float64)[:, :2] for poly in polys for r in poly if len(r) >= 3]
        if not nm or not rings:
            continue
        allp = np.concatenate(rings)
        bb = (allp[:, 0].min(), allp[:, 0].max(), allp[:, 1].min(), allp[:, 1].max())
        out.append((nm, rings, bb, (bb[1] - bb[0]) * (bb[3] - bb[2]), allp))
    print("[epicenters.geojson] %d区域を読み込み" % len(out))
    return out


EPI_FEATS = _load_epicenters()


def _in_ring(lon, lat, ring):
    x, y = ring[:, 0], ring[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cond = (y > lat) != (y2 > lat)
    dy = np.where(y2 == y, 1e-12, y2 - y)
    xint = x + (lat - y) * (x2 - x) / dy
    return bool(np.sum(cond & (lon < xint)) % 2 == 1)


def epicenter_region(lat, lon):
    """緯度経度を含む震央地名。複数に含まれれば最小の区域。どこにも入らなければ100km以内の最寄り。"""
    if not EPI_FEATS:
        return None
    best = None
    for nm, rings, (x0, x1, y0, y1), area, _ in EPI_FEATS:
        if not (x0 <= lon <= x1 and y0 <= lat <= y1):
            continue
        ins = False
        for r in rings:                              # 穴を含め偶奇則
            ins ^= _in_ring(lon, lat, r)
        if ins and (best is None or area < best[0]):
            best = (area, nm)
    if best:
        return best[1]
    kx = 111.32 * math.cos(math.radians(lat))
    near = None
    for nm, _, _, _, allp in EPI_FEATS:
        d = float(np.min(np.hypot((allp[:, 0] - lon) * kx, (allp[:, 1] - lat) * 110.57)))
        if near is None or d < near[0]:
            near = (d, nm)
    return near[1] if near and near[0] <= 100.0 else None

# ---------------------------------------------------------------------
# 区域 → 都道府県 → 地域区分 の対応（警報対象範囲の表示用）
#   各都道府県の代表点（県庁所在地ほか）への最寄りで観測点の都道府県を決め、
#   区域内で多数決。区域名に都道府県名が含まれていればそちらを優先する。
# ---------------------------------------------------------------------
_PREF_PTS = {
    "北海道": [(141.35, 43.06), (140.73, 41.77), (142.37, 43.77), (144.38, 43.0), (143.2, 42.9),
             (144.27, 44.02), (141.65, 45.42), (145.58, 43.33), (141.5, 42.6)],
    "青森県": [(140.74, 40.82), (141.49, 40.51), (140.47, 40.61)],
    "岩手県": [(141.15, 39.70), (141.97, 39.64), (141.13, 38.99)],
    "宮城県": [(140.87, 38.27), (141.3, 38.9)],
    "秋田県": [(140.10, 39.72), (140.55, 40.2)],
    "山形県": [(140.36, 38.24), (139.84, 38.91)],
    "福島県": [(140.47, 37.75), (139.93, 37.49), (140.89, 37.05)],
    "茨城県": [(140.45, 36.34), (140.3, 36.0)],
    "栃木県": [(139.88, 36.57), (139.7, 36.9)],
    "群馬県": [(139.06, 36.39), (138.9, 36.6)],
    "埼玉県": [(139.65, 35.86), (139.2, 36.0)],
    "千葉県": [(140.12, 35.61), (140.2, 35.2), (139.9, 35.0)],
    "東京都": [(139.69, 35.69), (139.8, 33.1), (142.2, 27.1)],
    "神奈川県": [(139.64, 35.45), (139.2, 35.3)],
    "新潟県": [(139.02, 37.90), (138.25, 37.1), (138.4, 38.05), (139.4, 37.6)],
    "富山県": [(137.21, 36.70), (137.0, 36.8)],
    "石川県": [(136.63, 36.59), (136.9, 37.4), (136.8, 37.0)],
    "福井県": [(136.22, 36.07), (135.9, 35.65)],
    "山梨県": [(138.57, 35.66), (138.6, 35.5)],
    "長野県": [(138.18, 36.65), (137.97, 36.23), (138.0, 35.7), (138.3, 36.0)],
    "岐阜県": [(136.72, 35.39), (137.2, 36.2), (137.25, 35.5)],
    "静岡県": [(138.38, 34.98), (137.73, 34.71), (139.0, 35.0), (138.9, 34.75)],
    "愛知県": [(136.91, 35.18), (137.15, 34.85)],
    "三重県": [(136.51, 34.73), (136.35, 34.2), (136.5, 33.8)],
    "滋賀県": [(135.87, 35.00), (136.1, 35.3)],
    "京都府": [(135.76, 35.02), (135.2, 35.4)],
    "大阪府": [(135.52, 34.69)],
    "兵庫県": [(135.18, 34.69), (134.8, 35.3), (134.7, 34.9), (134.85, 34.35)],
    "奈良県": [(135.83, 34.69), (135.9, 34.3)],
    "和歌山県": [(135.17, 34.23), (135.8, 33.6)],
    "鳥取県": [(134.24, 35.50), (133.5, 35.4)],
    "島根県": [(133.05, 35.47), (132.1, 35.0), (133.3, 36.2)],
    "岡山県": [(133.93, 34.66), (133.7, 35.0)],
    "広島県": [(132.46, 34.40), (133.2, 34.5)],
    "山口県": [(131.47, 34.19), (131.2, 34.0), (131.0, 34.4)],
    "徳島県": [(134.56, 34.07), (134.3, 33.8)],
    "香川県": [(134.04, 34.34)],
    "愛媛県": [(132.77, 33.84), (132.5, 33.3), (133.1, 33.6)],
    "高知県": [(133.53, 33.56), (133.0, 33.0), (134.0, 33.5)],
    "福岡県": [(130.42, 33.61), (130.8, 33.3)],
    "佐賀県": [(130.30, 33.25)],
    "長崎県": [(129.87, 32.75), (129.9, 33.2), (129.0, 33.0), (128.3, 34.6)],
    "熊本県": [(130.74, 32.79), (130.8, 32.3), (131.1, 32.9)],
    "大分県": [(131.61, 33.24), (131.2, 33.2)],
    "宮崎県": [(131.42, 31.91), (131.1, 32.1)],
    "鹿児島県": [(130.56, 31.56), (130.3, 31.2), (129.5, 28.4), (130.5, 30.4)],
    "沖縄県": [(127.68, 26.21), (127.9, 26.6), (124.15, 24.34), (123.0, 24.45)],
}
_PREF_REGION = {}
for _reg, _prefs in {
    "北海道": ["北海道"],
    "東北": ["青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県"],
    "関東": ["茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県"],
    "北陸・甲信越": ["新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県"],
    "東海": ["岐阜県", "静岡県", "愛知県"],
    "近畿": ["三重県", "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県"],
    "中国": ["鳥取県", "島根県", "岡山県", "広島県", "山口県"],
    "四国": ["徳島県", "香川県", "愛媛県", "高知県"],
    "九州": ["福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県"],
    "沖縄": ["沖縄県"],
}.items():
    for _p in _prefs:
        _PREF_REGION[_p] = _reg


def _build_area_geo():
    names, plon, plat = [], [], []
    for p, pts in _PREF_PTS.items():
        for lo_, la_ in pts:
            names.append(p)
            plon.append(lo_)
            plat.append(la_)
    plon, plat = np.array(plon), np.array(plat)
    kx = 111.32 * math.cos(math.radians(36.0))
    d = ((ST_LON[:, None] - plon[None, :]) * kx) ** 2 + ((ST_LAT[:, None] - plat[None, :]) * 110.57) ** 2
    st_pref = [names[j] for j in np.argmin(d, axis=1)]
    out = {}
    for a in np.unique(ST_AREA):
        votes = {}
        for i in np.where(ST_AREA == a)[0]:
            votes[st_pref[i]] = votes.get(st_pref[i], 0) + 1
        pref = max(votes, key=votes.get)
        nm = AREA_NAMES.get(int(a), "")
        for p in _PREF_PTS:                       # 区域名に都道府県名があれば優先
            if nm.startswith(p):
                pref = p
                break
        else:
            if nm.startswith("伊豆諸島") or nm.startswith("小笠原"):
                pref = "東京都"
        out[int(a)] = (pref, _PREF_REGION[pref])
    return out


AREA_GEO = _build_area_geo()

# ---------------------------------------------------------------------
# 海底観測網（S-net・DONET・N-net・気象庁の海底地震計など）: data/seafloor.csv（任意）
#   列: name, network, lon, lat [, arv, dep]     先頭が # の行はコメント
#   arv/dep が空なら、最寄りの陸上観測点5点の中央値を使う（地盤増幅の実測値が無いための仮置き）。
#   海域観測点は「P波検知（EEWの震源決定・規模推定）」と「地図表示」に使うが、
#   区域別の震度集計・警報対象地域・最大震度・PLUM法の起点には含めない（ST_LAND で除外）。
# ---------------------------------------------------------------------
ST_SEA = np.zeros(N_ST, dtype=bool)
ST_LAND = ~ST_SEA
ST_NAME = [""] * N_ST
SEA_AREA_ID = int(ST_AREA.max()) + 1


def _append_seafloor():
    global ST_LON, ST_LAT, ST_ARV, ST_DEP, ST_AREA, ST_SEA, ST_LAND, ST_NAME, N_ST
    path = DATA_DIR / "seafloor.csv"
    if not path.exists():
        return
    rows, seen = [], set()
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        for row in csv.DictReader(lines):
            try:
                lo, la = float(row["lon"]), float(row["lat"])
            except (KeyError, TypeError, ValueError):
                continue
            nm = (row.get("name") or "").strip()
            if not (10.0 <= la <= 55.0 and 110.0 <= lo <= 170.0) or (nm and nm in seen):
                continue
            seen.add(nm)

            def opt(k):
                try:
                    return float(row.get(k) or "")
                except ValueError:
                    return None
            rows.append((nm, (row.get("network") or "").strip(), lo, la, opt("arv"), opt("dep")))
    except Exception as e:
        print("[seafloor.csv] 読み込み失敗:", e)
        return
    if not rows:
        return
    land_lon, land_lat = ST_LON, ST_LAT
    lon, lat, arv, dep, names = [], [], [], [], []
    for nm, nw, lo, la, a, d in rows:
        if a is None or d is None:
            kx = 111.32 * math.cos(math.radians(la))
            dd = np.hypot((land_lon - lo) * kx, (land_lat - la) * 110.57)
            near = np.argsort(dd)[:5]
            a = float(np.median(ST_ARV[near])) if a is None else a
            d = float(np.median(ST_DEP[near])) if d is None else d
        lon.append(lo); lat.append(la); arv.append(a); dep.append(d)
        names.append((nw + " " + nm).strip())
    n_new = len(lon)
    ST_LON = np.concatenate([ST_LON, lon])
    ST_LAT = np.concatenate([ST_LAT, lat])
    ST_ARV = np.concatenate([ST_ARV, arv])
    ST_DEP = np.concatenate([ST_DEP, dep])
    ST_AREA = np.concatenate([ST_AREA, np.full(n_new, SEA_AREA_ID, dtype=np.int64)])
    ST_SEA = np.concatenate([ST_SEA, np.ones(n_new, dtype=bool)])
    ST_LAND = ~ST_SEA
    ST_NAME = ST_NAME + names
    N_ST = len(ST_LON)
    AREA_NAMES[SEA_AREA_ID] = "海域（海底観測網）"
    AREA_GEO[SEA_AREA_ID] = ("海域", "海域")
    print("[seafloor.csv] 海底観測点 %d点を追加（陸上%d点＋海底%d点）" % (n_new, N_ST - n_new, n_new))


_append_seafloor()

# 海溝・トラフ軸の近似ポリラインとその名称（経度, 緯度）。判別用の概略値
TRENCHES = {
    "千島海溝": [(155.0, 48.5), (150.5, 45.5), (147.3, 43.3), (145.0, 41.3)],
    "日本海溝": [(144.5, 41.0), (144.0, 39.5), (143.7, 38.0), (143.0, 36.5), (142.5, 35.3)],
    "伊豆・小笠原海溝": [(142.5, 35.3), (142.1, 33.5), (141.9, 31.5), (142.3, 29.0), (143.0, 26.0)],
    "相模トラフ": [(139.2, 34.9), (139.9, 34.4), (140.9, 34.3)],
    "南海トラフ": [(138.5, 34.6), (137.7, 33.9), (136.7, 33.3), (135.7, 32.7),
                (134.5, 32.2), (133.2, 31.7), (132.2, 31.2)],
    "琉球海溝": [(132.2, 31.2), (131.0, 29.7), (129.8, 28.0), (128.5, 26.0),
               (126.5, 24.2), (124.5, 23.4), (122.5, 23.2)],
}


def classify_source(mag, depth, lat0, lon0):
    # 震源が海溝軸の近く、かつプレート境界の深さ以浅なら「海溝型」と自動判別する。
    # 海溝型かつM7以上は、震源断層の広がり（断層長L=10^(0.5M-1.85) km）を海溝走向に沿って考慮する。
    kx = 111.32 * math.cos(math.radians(lat0))
    ky = 110.57
    best = None
    for name, pts in TRENCHES.items():
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            ax, ay = (x1 - lon0) * kx, (y1 - lat0) * ky
            bx, by = (x2 - lon0) * kx, (y2 - lat0) * ky
            vx, vy = bx - ax, by - ay
            l2 = vx * vx + vy * vy
            t = 0.0 if l2 == 0 else min(max(-(ax * vx + ay * vy) / l2, 0.0), 1.0)
            d = math.hypot(ax + t * vx, ay + t * vy)
            if best is None or d < best[0]:
                n = math.sqrt(l2) or 1.0
                best = (d, name, vx / n, vy / n)
    d, name, ux, uy = best
    is_trench = (d <= 250.0 and mag >= 5.5 and depth <= min(20.0 + 0.3 * d, 60.0))
    length = min(10.0 ** (0.5 * mag - 1.85), 600.0) if (is_trench and mag >= 7.0) else 0.0
    return {"type": "海溝型" if is_trench else "内陸・その他", "name": name if is_trench else "",
            "L": length, "ux": ux, "uy": uy, "kx": kx, "ky": ky}


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
    # 深発地震では震源の真上にP波が着く（震央距離が0から立ち上がる）瞬間に、震央距離の
    # 伸びる速さが理論上いくらでも大きくなる（sqrt(hypo^2-depth^2)の微分が発散）。
    # これを「近似式の発散」と誤判定して直線外挿に置き換えないよう、立ち上がりから
    # 30秒間は判定対象から外す。
    pos = np.where(p_raw > 0.0)[0]
    t_on = float(tg[pos[0]]) if len(pos) else 0.0
    bad = np.where((tg > max(20.0, t_on + 30.0)) & ((v > 14.0) | (v < 0.0)))[0]
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


def final_intensity(mag, depth, lat0, lon0, min_x=0.0):
    """
    各観測点の最終的な計測震度（Scratch版 震度計算 の非予想側の式）を返す。
    戻り値: (m01, 震央距離km, depth_p, src)
    min_x: 震源距離の下限[km]。EEWの予測では、推定震源が観測点の直下にあるとき
           距離≈0で式が発散し震度7になるのを防ぐために使う（シミュレーション本体は0）。
    """
    # Scratch版の「震度計算処理」と同じ入力補正
    #   M引数 = M - (7.1-2.2),  深さ引数 = 深さ - 10
    m_arg = mag - (7.1 - 2.2)
    depth_p = depth - 10.0

    dist = hubeny_km(lat0, lon0, ST_LAT, ST_LON)          # 震央距離
    src = classify_source(mag, depth, lat0, lon0)
    dfault = dist
    if src["L"] > 0:                                      # 海溝型の大地震: 断層面までの距離で減衰を評価
        sx = (ST_LON - lon0) * src["kx"]
        sy = (ST_LAT - lat0) * src["ky"]
        tt = np.clip(sx * src["ux"] + sy * src["uy"], -src["L"] / 2, src["L"] / 2)
        dfault = np.minimum(dist, np.hypot(sx - tt * src["ux"], sy - tt * src["uy"]))
    x_hypo = np.sqrt(dfault ** 2 + depth_p ** 2)          # 震源距離
    if min_x > 0.0:
        x_hypo = np.maximum(x_hypo, min_x)

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
    return m01, dist, depth_p, src


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


# =====================================================================
# 2b. 緊急地震速報（EEW）: 観測された揺れから震源・規模・深さを逐次推定する
#   実際のEEWと同様の流れ:
#     1) 観測点がP波を検知（検知時刻にはピック誤差を含める）
#     2) 検知済み観測点の到達時刻差＋「まだ検知していない周辺観測点」の制約から、
#        グリッドサーチで震源（緯度・経度・深さ・発生時刻）を決定
#        （検知点が少ないうちは観測点付近に仮定、4点以上で深さも推定）
#     3) P波検知後3秒間の振幅から、推定震源までの距離で補正してマグニチュードを推定
#        （波形が増えるほど大きな地震の規模を推定できる＝初期は過小評価になりやすい）
#     4) 推定した震源・Mから各区域の震度を予測し、予報/警報/特別警報を判定
#     5) 推定値が変化するたびに続報（第N報）を発表
#     6) PLUM法を併用: 観測された震度を周囲30km以内へそのまま予測震度とする
#        （規模推定が頭打ちでも実際に強く揺れた地域を取りこぼさない）
#     7) 警報の対象地域は一度発表したら続報でも取り消さない（追加時は即続報）
# =====================================================================
TG = np.arange(0.0, 600.0 + 1e-9, 0.05)
# 深さ候補。深発地震（最大699km）まで、深いほど粗くなり過ぎないよう細かく刻む
EEW_DEPTHS = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0, 200.0, 250.0,
              300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 690.0]
EEW_DEPTHS_COARSE = (10.0, 60.0, 150.0, 300.0, 450.0, 600.0)   # 粗い探索（広域グリッド）で試す深さ
EEW_DELAY = 2.0          # 観測点のデータ伝送(1秒)＋処理(1秒)の遅れ[秒]
_TT_CACHE = {}


def _tt(depth):
    """深さdepth[km]のP波・S波の (震央距離[km]の曲線) を返す。時刻軸はTG"""
    if depth not in _TT_CACHE:
        dp = depth - 10.0
        p = sanitize_p_curve(TG, wave_epicentral_km(TG, dp, True))
        s_ = np.maximum.accumulate(wave_epicentral_km(TG, dp, False))
        _TT_CACHE[depth] = (p, s_)
    return _TT_CACHE[depth]


def _planar_km(lat1, lon1, lat2, lon2):
    """高速な簡易距離[km]（等距円筒近似。EEWの探索用）"""
    p1, p2 = np.radians(lat1), np.radians(lat2)       # 遠地震源でも誤差が出ないよう大圏距離
    a = np.sin((p1 - p2) / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon1 - lon2) / 2.0) ** 2
    return 12742.0 * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


_INV_STEP = 0.5          # 走時テーブルの距離刻み[km]
_INV_CACHE = {}


def _tt_inv(depth):
    """深さdepthのP波について、震央距離(0, 0.5, 1.0, ...km)から伝播時間[秒]を引く等間隔テーブル。
    np.interp（二分探索）より、配列演算だけで引ける等間隔テーブルのほうが格段に速い。"""
    if depth not in _INV_CACHE:
        pc = _tt(depth)[0]
        xs = np.arange(0.0, float(pc[-1]) + 2.0 * _INV_STEP, _INV_STEP)
        tab = np.interp(xs, pc, TG, right=600.0)
        _INV_CACHE[depth] = (tab, np.diff(tab))
    return _INV_CACHE[depth]


def _lookup_tt(ds, depth):
    tab, dtab = _tt_inv(depth)
    x = np.minimum(ds * (1.0 / _INV_STEP), len(dtab) - 1e-6)
    i = x.astype(np.intp)
    return tab[i] + dtab[i] * (x - i)


def _unit_vec(lat, lon):
    p, l = np.radians(lat), np.radians(lon)
    c = np.cos(p)
    return np.stack([c * np.cos(l), c * np.sin(l), np.sin(p)], axis=-1)


def _gc_dist(ua, ub):
    """単位ベクトル (A,3) と (B,3) の全組み合わせの大圏距離[km] (A,B)（_planar_kmと同じ値）"""
    d2 = np.maximum(2.0 - 2.0 * (ua @ ub.T), 0.0)         # 弦長の2乗
    return 12742.0 * np.arcsin(np.minimum(np.sqrt(d2) * 0.5, 1.0))


def _grid_cost(ds, du, depth, tobs, t_data, base):
    """ds: グリッド点→検知済み観測点の距離(G,S), du: →未検知の観測点の距離(G,U)またはNone,
    base: グリッド点ごとの事前分布項(G,)。距離行列は深さに依存しないので呼び出し側で1回だけ作る。"""
    tt_ = _lookup_tt(ds, depth)
    t0 = np.mean(tobs[None, :] - tt_, axis=1)                 # 発生時刻（最小二乗）
    res = tobs[None, :] - tt_ - t0[:, None]
    cost = np.mean(res * res, axis=1)
    if du is not None:
        # まだ検知していない観測点: 予測到達時刻が現在より前なら矛盾（ペナルティ）
        arr = t0[:, None] + _lookup_tt(du, depth)
        pen = np.maximum(t_data - arr, 0.0)
        cost = cost + 0.1 * np.sum(pen * pen, axis=1)
    return cost + base + 0.0002 * depth, t0


def _locate(tobs, lat_s, lon_s, lat_u, lon_u, t_data, free_depth, prior):
    """粗いグリッド→細かいグリッドの2段階で震源を探索。戻り値 (lat, lon, depth, t0, cost)
    粗い探索（1°刻み±24°）は代表的な観測点だけで盆地を決め、以降の細かい探索は全点で行う。"""
    lat1, lon1 = float(lat_s[0]), float(lon_s[0])
    depths = EEW_DEPTHS if free_depth else [10.0]
    us = _unit_vec(lat_s, lon_s)
    uu = _unit_vec(lat_u, lon_u) if len(lat_u) else None
    u1 = us[:1]
    ns = len(tobs)
    sub = (np.unique(np.concatenate([np.arange(10), np.linspace(10, ns - 1, 18).astype(int)]))
           if ns > 28 else np.arange(ns))                    # 早い検知10点＋広く散らばった点
    best = None
    clat, clon = lat1, lon1
    for stage in (0, 1, 2):                              # 1°刻み(±24°) → 0.2°刻み → 0.05°刻み
        if stage == 0:
            offs = np.arange(-24.0, 24.0001, 1.0)      # 遠地の巨大地震も探索できる範囲
        else:
            offs = np.arange(-1.0, 1.0001, 0.2) if stage == 1 else np.arange(-0.2, 0.2001, 0.05)
            clat, clon = best[1], best[2]
        la, lo = np.meshgrid(clat + offs, clon + offs, indexing="ij")
        glat, glon = la.ravel(), lo.ravel()
        ug = _unit_vec(glat, glon)
        if stage == 0:
            ds, tob = _gc_dist(ug, us[sub]), tobs[sub]
            du = _gc_dist(ug, uu[:20]) if uu is not None else None
        else:
            ds, tob = _gc_dist(ug, us), tobs
            du = _gc_dist(ug, uu) if uu is not None else None
        base = prior * _gc_dist(ug, u1)[:, 0]
        best_st = None
        for d in (depths if stage else [x for x in depths if x in EEW_DEPTHS_COARSE] or depths):
            c, t0 = _grid_cost(ds, du, d, tob, t_data, base)
            k = int(np.argmin(c))
            if best_st is None or c[k] < best_st[0]:
                best_st = (float(c[k]), float(glat[k]), float(glon[k]), d, float(t0[k]))
        best = best_st      # 段ごとに最良を取り直す（粗い探索は部分集合のコストなので段をまたいで比べない）
    return best[1], best[2], best[3], best[4], best[0]


def _mw_of_mag(mag):
    m_arg = mag - (7.1 - 2.2)
    a = 10.0 ** (0.91 * m_arg - 3.49)
    d = 10.0 ** (0.69 * m_arg - 4.8)
    return (2.0 / 3.0) * math.log10(3.0e10 * (a * 1.0e6) * d) - 6.07


_MW_B = _mw_of_mag(4.9)
_MW_A = _mw_of_mag(5.9) - _MW_B


def _mag_of_mw(mw):
    return 4.9 + (mw - _MW_B) / _MW_A


def _mw_from_m01(m01_i, x, depth_p, arv, sdep):
    """観測点の計測震度から、距離減衰式を逆算してMwを求める（二分法）"""
    lo = np.full(m01_i.shape, -5.0)      # 元の式のMw換算ではM4でも約-1.6になるため下限を広く取る
    hi = np.full(m01_i.shape, 13.0)
    for _ in range(40):
        mid = (lo + hi) / 2.0
        lp = (0.58 * mid + 0.0038 * depth_p
              - np.log10(x + 0.0028 * 10.0 ** (0.5 * mid)) - 0.002 * x - 1.29)
        m = 2.68 + 1.72 * (lp + np.log10(arv * (4000.0 - sdep)))
        up = m > m01_i
        hi = np.where(up, mid, hi)
        lo = np.where(up, lo, mid)
    return (lo + hi) / 2.0


def _estimate_mag(avail, tp_n, lat_e, lon_e, dep_e, t0_e, t_data, rt, tau, rng):
    """P波到達後〜S波到達前の振幅からマグニチュードを推定。使える観測点が無ければNone"""
    pc, sc = _tt(dep_e)
    t_ceil = np.ceil(tp_n[avail]).astype(int)
    d_all = np.maximum(hubeny_km(lat_e, lon_e, ST_LAT[avail], ST_LON[avail]), 8.0)
    ts_e = t0_e + np.interp(d_all, sc, TG, right=600.0)
    # 各観測点で「P波が届いてからS波が届く前」の最新フレームの振幅を使う。
    # 推定震源が真の震源からずれていると推定S到達時刻(ts_e)が実際より遅くなり、S波到達後の
    # フレームをP波の式で逆算して規模が極端に小さくなる（M9→M2）。そこで観測波形そのものの
    # 立ち上がり（S波の急な振幅ジャンプ）を検出し、それ以降のフレームはP波解析に使わない。
    fmax = min(int(math.floor(t_data)), rt.shape[0] - 1)
    with np.errstate(invalid="ignore"):
        sub = rt[:fmax + 1, avail].astype(np.float64)
        jump = (sub[1:] - sub[:-1]) > 0.8
    jump &= np.arange(1, fmax + 1)[:, None] >= (t_ceil + 2)[None, :]   # P波立ち上がり直後は除外
    has_on = jump.any(axis=0)
    onset = np.where(has_on, jump.argmax(axis=0) + 1, 10 ** 6)          # S波が観測された最初のフレーム
    f = np.minimum(np.minimum(fmax, np.floor(ts_e - 1.5).astype(int)), onset - 1)
    ok = f >= t_ceil + 1

    dep_p = dep_e - 10.0
    dterm = min(max(dep_p - 10.0, 0.0), 30.0)
    mw_p = None
    if ok.any():
        sel = np.where(ok)[0]
        sel = sel[np.argsort(d_all[sel])[:10]]            # 震源に近い観測点ほど信号が強い（上位10点）
        idx, ff, d = avail[sel], f[sel], d_all[sel]
        obs = rt[ff, idx].astype(np.float64) + rng.normal(0.0, 0.25, len(idx))   # 振幅の観測ノイズ
        bai = ((1.0 + ST_ARV[idx]) / 2.2) / 2.0
        fthr = -3.3 + (1.0 + bai * 3.0) / 4.0             # 検出下限（これ以下は規模の上限しか分からない）
        cen = obs <= fthr + 0.3
        obs = np.maximum(obs, fthr + 0.3)
        bai_dep = (ST_DEP[idx] + 1200.0) / 600.0
        kp = (-0.006 + dterm / 2000.0) - bai_dep / 300.0
        pr = np.interp(ff - t0_e, TG, pc)                  # 推定震源から見たP波の到達距離
        zp = d - pr
        # 振幅が上限（cap）に達した区間。順方向は v = base + zp*kp*2（zp<0, kp<0 で増加）が
        # cap_s = base + 2.0 を超えたら cap_s に張り付く。よって条件は「増加分 zp*kp*2 >= 2.0」。
        # （以前は符号が逆で、P波通過後の経過距離が大きい点＝深発・遠地ほど上限張り付きを
        #   見逃し、増加分を過大に引いて規模を極端に小さく見積もっていた）
        capd = (zp * kp * 2.0) >= 2.0
        m01_i = np.where(capd, obs + 1.5 + d / 150.0 + dterm / 52.0,
                         obs - zp * kp * 2.0 + 3.5 + d / 150.0 + dterm / 52.0)
        x = np.sqrt(d * d + dep_p * dep_p)
        mws = _mw_from_m01(m01_i, x, dep_p, ST_ARV[idx], ST_DEP[idx])
        # 上限が min(cap_s, 4.4) なので、4.4に張り付いた点は規模の「下限」しか分からない
        lbm = capd & ~cen & (obs >= 4.35)
        exact = ~cen & ~lbm
        if exact.any():
            mw_p = float(np.median(mws[exact]))
            if lbm.any():
                mw_p = max(mw_p, float(mws[lbm].max()))
        elif lbm.any():
            mw_p = float(mws[lbm].max())
        if cen.any():                                      # 下限以下の観測点が示す上限を超えない
            ub = float(mws[cen].min())
            mw_p = (ub - 0.4) if mw_p is None else min(mw_p, ub)

    # --- S波（主要動）の振幅も使う: S波が観測された点の立ち上がりフレームから逆算 ---
    # リアルタイム震度のS波側の式: v = s0 + zs*ks,  s0 = min(m01-(1.5+k2/150)-dterm/52, 4.6)
    # 立ち上がり直後は zs≈0 なので v≈s0。s0が上限4.6に張り付いていれば規模の下限しか分からない。
    mw_s = mw_lb = mw_ub = None
    son = np.where(has_on)[0]
    if len(son):
        son = son[np.argsort(d_all[son])[:10]]
        idx, d = avail[son], d_all[son]
        obs = rt[onset[son], idx].astype(np.float64) + rng.normal(0.0, 0.25, len(idx))
        bai = ((1.0 + ST_ARV[idx]) / 2.2) / 2.0
        fthr = -3.3 + (1.0 + bai * 3.0) / 4.0
        low = obs <= fthr + 0.3                            # 検出下限以下（上限のみ）
        hi = obs >= 4.5                                    # s0が上限に張り付き（下限のみ）
        k2 = np.clip(d, 0.5, 300.0)
        m01_s = np.maximum(obs, fthr + 0.3) + 1.5 + k2 / 150.0 + dterm / 52.0
        x = np.sqrt(d * d + dep_p * dep_p)
        mws_s = _mw_from_m01(m01_s, x, dep_p, ST_ARV[idx], ST_DEP[idx])
        mid = ~low & ~hi
        if mid.any():
            mw_s = float(np.median(mws_s[mid]))
        if hi.any():
            mw_lb = float(mws_s[hi].max())
        if low.any():
            mw_ub = float(mws_s[low].min())

    mw = mw_p
    if mw_s is not None:
        mw = mw_s
    if mw_lb is not None:
        mw = mw_lb if mw is None else max(mw, mw_lb)
    if mw_ub is not None:
        mw = (mw_ub - 0.4) if mw is None else min(mw, mw_ub)
    if mw is None:
        return None
    mag = min(max(_mag_of_mw(mw), 2.0), 15.0)
    # 観測波形が短いうちは大地震の規模を過小評価する（経過時間とともに上限が上がる）
    msat = 6.2 + 0.95 * math.log2(1.0 + max(tau, 0.0) / 3.0)
    return min(mag, msat), mag > msat      # (推定M, 上限に頭打ち中=まだ規模が成長中か)


def _eew_tt_payload(nf):
    """EEW推定震源から描く推定P/S波用の走時曲線（深さ別・1秒刻みの震央距離km）"""
    out = {}
    for d in EEW_DEPTHS:
        pc, sc = _tt(d)
        out[str(int(d))] = [[round(float(x), 1) for x in pc[::20][:nf]],
                            [round(float(x), 1) for x in sc[::20][:nf]]]
    return out


def eew_stream(mag, depth, lat0, lon0, tp, rt, t_end):
    """緊急地震速報を演算できた順に返すジェネレータ。
    ("upto", T): T秒までの速報は確定 / ("report", 報) / ("final", 報番号) / ("done"|"none", None)"""
    rng = np.random.default_rng(int(abs(lat0) * 1000 + abs(lon0) * 100 + mag * 10 + depth) % (2 ** 31))
    tp_n = tp + rng.normal(0.0, 0.1, N_ST)            # P波ピックの誤差
    order = np.argsort(tp_n)
    n_fin = int(np.isfinite(tp_n).sum())
    if n_fin < 2:
        yield ("none", None)
        return
    order = order[:n_fin]
    t_first = float(tp_n[order[0]])
    T = float(tp_n[order[0]]) + EEW_DELAY          # 最初の1点検知から開始（切り上げなし）
    T0 = T
    # 実際に最大の揺れが観測される時刻。最終報はこれより後に発表する
    rl = rt[:, ST_LAND]                                # 陸上観測点のみで最大震度・終了時刻を決める
    peak = float(rl.max())
    t_peak = int(np.argmax(rl.max(axis=1) >= peak - 0.05))
    # 発表の終了目安: 震度4以上の揺れが最後の地域に達する時刻まで（直下型で最大震度が
    # 早く出ても、遠方の地域に向けて続報を出し続ける）
    m4 = rl >= 3.5
    has4 = m4.any(axis=0)
    t_last = float(np.argmax(m4[:, has4], axis=0).max()) if has4.any() else float(t_peak)
    T_stop = min(t_end - 1.0, max(t_first + 60.0, t_last + 6.0))
    max_a = int(ST_AREA.max())
    reports, last, lv_max = [], None, 0
    warned = {}      # 警報対象地域（発表済みは続報でも保持）
    yield ("upto", T0 - 0.001)

    while T <= T_stop:
        step = 1.0 if T < T0 + 60 else (2.0 if T < T0 + 180 else 4.0)
        is_last = (T + step > T_stop)                 # この回が最後 → 最終報
        t_data = T - EEW_DELAY
        avail = order[tp_n[order] <= t_data]
        n = len(avail)
        if n >= 1:
            # 最初の20点＋広く散らばった検知点（開口が広いほど遠地の距離が決まる）
            use = avail[:20] if n <= 20 else np.concatenate(
                [avail[:20], avail[20:][::max(1, (n - 20) // 40)][:40]])
            mask = np.ones(N_ST, dtype=bool)
            mask[avail] = False
            cand = np.where(mask)[0]
            dc = _planar_km(ST_LAT[use[0]], ST_LON[use[0]], ST_LAT[cand], ST_LON[cand])
            unt = cand[np.argsort(dc)[:50]]
            if n == 1:     # 1点検知: 観測点の直下(深さ10km)と仮定（テリトリー法）
                lat_e, lon_e, dep_e = float(ST_LAT[avail[0]]), float(ST_LON[avail[0]]), 10.0
                t0_e, cost_e = float(tp_n[avail[0]]), 1.0
            else:
                lat_e, lon_e, dep_e, t0_e, cost_e = _locate(
                    tp_n[use], ST_LAT[use], ST_LON[use], ST_LAT[unt], ST_LON[unt],
                    t_data, n >= 4, 0.002 if n < 4 else 0.0001)
            est = _estimate_mag(avail, tp_n, lat_e, lon_e, dep_e, t0_e, t_data, rt,
                                T - t_first, rng)
            # 規模がまだ推定できない間（P波検知直後で振幅が得られない）は、仮の固定値を
            # 発表せず、直前の報の値を引き継ぐ。最初の報は規模が推定できてから発表する。
            mag_known = (est is not None) or (last is not None)
            if est is None:
                mag_e, capped = (last["mag"] if last else 3.0), False
            else:
                mag_e, capped = est

            # 推定震源が観測点直下でも式が発散しないよう、震源距離に下限(10km)を設ける
            m01_p, _, _, _ = final_intensity(mag_e, dep_e, lat_e, lon_e, min_x=10.0)
            # --- PLUM法（実際の気象庁EEWでも併用）---
            fr = min(int(math.floor(t_data)), rt.shape[0] - 1)
            ob = np.nan_to_num(rt[fr].astype(np.float64), nan=-9.0, neginf=-9.0, posinf=-9.0)
            trig = np.where((ob >= 4.5) & ST_LAND)[0]          # 震度5弱以上を観測した点のみ起点
            plum_used = False
            if len(trig):
                dd = _planar_km(ST_LAT[trig][:, None], ST_LON[trig][:, None],
                                ST_LAT[None, :], ST_LON[None, :])
                pl = np.max(np.where(dd <= 30.0, ob[trig][:, None], -9.0), axis=0)
                upm = pl > m01_p + 0.05
                if upm.any():
                    m01_p = np.where(upm, pl, m01_p)
                    plum_used = True
            # 予想震度の塗りつぶし用: 観測点ごとのばらつき(地盤増幅の差)で地図がまだらにならないよう、
            # 気象庁のEEWと同じく「区域ごとの最大予想震度」を区域内の全観測点の値とする
            amx_all = np.full(max_a + 1, -99.0)
            np.maximum.at(amx_all, ST_AREA[ST_LAND], m01_p[ST_LAND])
            pq_vals = np.where(ST_SEA, -9.0, amx_all[ST_AREA])    # 海域は予想震度の塗りつぶしの対象外
            vmax_e = float(m01_p[ST_LAND].max())
            cls_e, lb_max = shindo_class(vmax_e)
            if cls_e >= 7:            # 推定最大震度6弱以上 → 特別警報
                lv = 3
            elif cls_e >= 5:          # 推定最大震度5弱以上 → 警報
                lv = 2
            elif cls_e >= 3 or mag_e >= 3.5:   # 震度3以上またはM3.5以上 → 予報
                lv = 1
            else:
                lv = 0
            if n == 1:
                lv = min(lv, 1)                       # 1点検知では警報は出さない
            if not mag_known:
                lv = 0                                # 規模未確定の間は発表しない
            lv = max(lv, lv_max)                      # 警報・特別警報は途中で引き下げない

            areas = []
            if lv >= 2:
                amx = np.full(max_a + 1, -99.0)
                np.maximum.at(amx, ST_AREA[ST_LAND], m01_p[ST_LAND])
                de = hubeny_km(lat_e, lon_e, ST_LAT, ST_LON)
                ts_all = t0_e + np.interp(de, _tt(dep_e)[1], TG, right=600.0)
                ats = np.full(max_a + 1, np.inf)
                np.minimum.at(ats, ST_AREA[ST_LAND], ts_all[ST_LAND])
                for a in np.argsort(-amx):
                    c, lb = shindo_class(float(amx[a]))
                    if c < 4:
                        break
                    pf, rg = AREA_GEO.get(int(a), ("その他", "その他"))
                    areas.append({"name": AREA_NAMES.get(int(a), "区域%d" % int(a)),
                                  "pref": pf, "region": rg,
                                  "label": lb, "cls": c,
                                  "ts": round(float(ats[a]), 1) if np.isfinite(ats[a]) else None})
                    if len(areas) >= 60:
                        break
                for a_ in areas:                    # 発表済みの対象地域は取り消さない
                    warned[a_["name"]] = a_
                areas = sorted(warned.values(), key=lambda x: -x["cls"])[:60]

            if lv >= 1:
                after_peak = is_last                  # 発表終了の目安に達した → 最終報
                if last is None:
                    changed = True
                else:
                    # 発表基準: 規模が大きい/成長中、または震源の推定が不確か（検知点が少ない・
                    # 走時残差が大きい）なほど細かく更新。確定的な推定では更新しない。
                    unsure = (n < 8) or (cost_e > 0.05)
                    dm_th = 0.1 if (mag_e >= 7.0 or capped) else (0.2 if mag_e >= 5.5 else 0.3)
                    pos_th = 10.0 if unsure else 25.0
                    dpos = float(_planar_km(lat_e, lon_e, last["lat"], last["lon"]))
                    moved = (abs(mag_e - last["mag"]) >= dm_th or dpos >= pos_th
                             or abs(dep_e - last["depth"]) >= 20.0
                             or (cls_e != last["maxCls"] and abs(vmax_e - last["max"]) >= 0.3))
                    area_added = len(areas) > len(last["areas"])   # 警報対象地域の追加は即続報
                    changed = (lv != last["lv"]) or area_added or (moved and T - last["t"] >= (1 if lv >= 2 else 2))
                if (changed and len(reports) < 80) or after_peak:
                    # 震源地名: 推定震源に最も近い観測点の区域名。25km超離れていれば「○○県沖」
                    de_ = np.where(ST_LAND, _planar_km(lat_e, lon_e, ST_LAT, ST_LON), 1e9)   # 地名は陸上観測点基準
                    j_ = int(np.argmin(de_))
                    epi_geo = epicenter_region(lat_e, lon_e)      # 気象庁の震央地名（データがあれば）
                    if epi_geo:
                        epi_name = epi_geo
                    elif de_[j_] <= 25.0:
                        epi_name = AREA_NAMES.get(int(ST_AREA[j_]), "")
                    elif de_[j_] <= 300.0:
                        epi_name = AREA_GEO.get(int(ST_AREA[j_]), ("", ""))[0] + "沖"
                    else:
                        epi_name = "日本の遠方"
                    last = {"no": len(reports) + 1, "t": round(float(T), 1), "lat": round(lat_e, 2),
                            "epiName": epi_name,
                            "lon": round(lon_e, 2), "depth": int(dep_e), "mag": round(mag_e, 1),
                            "nSt": int(n), "t0": round(float(t0_e), 2), "max": round(vmax_e, 1), "maxLabel": lb_max,
                            "maxCls": cls_e, "plum": bool(plum_used), "lv": lv, "areas": areas, "final": after_peak,
                            # 観測点ごとの予想震度（×10, int8, base64）: 地図の塗りつぶし用
                            "pq": base64.b64encode(np.clip(np.round(np.nan_to_num(pq_vals, nan=-9.0) * 10.0), -127, 127)
                                                   .astype(np.int8).tobytes()).decode("ascii")}
                    reports.append(last)
                    yield ("report", last)
                    lv_max = lv
                    if after_peak:
                        break
        T += step
        yield ("upto", T - 0.001)

    if not reports:
        yield ("none", None)
        return
    if not reports[-1].get("final"):
        reports[-1]["final"] = True
        yield ("final", reports[-1]["no"])
    yield ("done", None)


def build_eew(mag, depth, lat0, lon0, tp, rt, t_end):
    """一括版（従来互換）"""
    reports = []
    for kind, v in eew_stream(mag, depth, lat0, lon0, tp, rt, t_end):
        if kind == "report":
            reports.append(v)
        elif kind == "final":
            reports[-1]["final"] = True
    return {"reports": reports} if reports else None


def quake_core(lat0, lon0, mag, depth, dt=1.0):
    """1つの地震の計算（最終震度・P/S波の到達距離曲線・リアルタイム震度フレーム）。
    震源の発生時刻を0秒とした「その地震だけのローカル時間」で返す。"""
    m01, dist, dep_p, src = final_intensity(mag, depth, lat0, lon0)

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
    return {"m01": m01, "src": src, "p_curve": p_curve, "s_curve": s_curve,
            "tp": tp, "rt": rt, "t_end": t_end, "nf": nf}


def _area_list(m01):
    """区域別の最大震度（上位20区域・震度1以上）"""
    area_max = {}
    for a, v, sea in zip(ST_AREA, m01, ST_SEA):
        if sea:
            continue                                      # 海底観測点は区域集計に含めない
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
    return areas


def _clean(a, nd=1):
    return [None if not math.isfinite(x) else round(float(x), nd) for x in a]


def sim_multi(quakes, dt=1.0, gz=False):
    """複数の地震を同時に扱う。quakes: [{lat, lon, mag, depth, delay(秒, 整数)}, ...]
    各地震を独立に計算し、delay秒ずらして重ね合わせる。
    観測点ごとのリアルタイム震度・最終震度は、全地震の最大値を採用する。
    戻り値: (基本データdict, [各地震の (tp, rt, t_end)])"""
    cores = [quake_core(q["lat"], q["lon"], q["mag"], q["depth"], dt) for q in quakes]
    t_end = max(q["delay"] + c["t_end"] for q, c in zip(quakes, cores))
    nf = int(t_end / dt) + 1

    rt_all, m01_all = None, None
    for q, c in zip(quakes, cores):
        d = int(round(q["delay"] / dt))
        rt = c["rt"]
        sh = np.empty((nf, N_ST), dtype=np.float32)
        sh[:d] = rt[0]                                    # 発生前: 揺れなし（観測点ごとの下限値）
        sh[d:d + len(rt)] = rt
        sh[d + len(rt):] = rt[-1]                         # 計算範囲の終了後は最終フレームを保持
        rt_all = sh if rt_all is None else np.fmax(rt_all, sh)
        m01_all = c["m01"] if m01_all is None else np.maximum(m01_all, c["m01"])

    qz = np.clip(np.round(rt_all * 10.0), -127, 127).astype(np.int8)
    vmax = float(m01_all.max())
    cls, label = shindo_class(vmax)
    out = {
        "dt": dt,
        "nf": nf,
        "ns": N_ST,
        "q": base64.b64encode(gzip.compress(qz.tobytes(), compresslevel=6) if gz else qz.tobytes()).decode("ascii"),
        "gz": bool(gz),
        "final": _clean(m01_all, 2),
        "max": round(vmax, 2),
        "maxCls": cls,
        "maxLabel": label,
        "areas": _area_list(m01_all),
        "tEnd": t_end,
        "ett": _eew_tt_payload(nf),
        "quakes": [{
            "lat": q["lat"], "lon": q["lon"], "mag": q["mag"], "depth": q["depth"],
            "delay": q["delay"],
            "P": _clean(c["p_curve"], 2),                 # その地震のローカル時間でのP波到達距離
            "S": _clean(c["s_curve"], 2),
            "tEnd": c["t_end"],
            "src": {"type": c["src"]["type"], "name": c["src"]["name"], "L": round(c["src"]["L"])},
        } for q, c in zip(quakes, cores)],
    }
    return out, [(c["tp"], c["rt"], c["t_end"]) for c in cores]


def _shift_report(r, delay):
    """EEWの報（その地震のローカル時間）を、全体の時間軸へdelay秒ずらしたコピーにする"""
    r = dict(r)
    r["t"] = round(r["t"] + delay, 1)
    r["t0"] = round(r["t0"] + delay, 2)
    r["areas"] = [dict(a, ts=(None if a["ts"] is None else round(a["ts"] + delay, 1)))
                  for a in r["areas"]]
    return r


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
            "area": [int(x) for x in ST_AREA],
            "sea": [int(x) for x in ST_SEA],
            "anames": {str(k): v for k, v in AREA_NAMES.items()},
        }
    return _json_response(_STATIONS_CACHE)


MAX_QUAKES = 5


@app.post("/api/simulate")
def api_simulate():
    try:
        j = request.get_json(force=True, silent=False) or {}
        raw = j.get("quakes")
        if raw is None:
            raw = [j]                                     # 単発（従来の {lat, lon, mag, depth} 形式）
        quakes = [{"lat": float(r["lat"]), "lon": float(r["lon"]),
                   "mag": float(r["mag"]), "depth": float(r["depth"]),
                   "delay": float(r.get("delay") or 0.0)} for r in raw]
    except Exception:
        return _json_response({"error": "入力が不正です"}, 400)
    if not (1 <= len(quakes) <= MAX_QUAKES):
        return _json_response({"error": "地震は1〜%d個まで指定できます" % MAX_QUAKES}, 400)
    for q in quakes:
        if not (10.0 <= q["lat"] <= 55.0 and 110.0 <= q["lon"] <= 165.0):
            return _json_response({"error": "震源は日本周辺（緯度10〜55, 経度110〜165）で指定してください"}, 400)
        q["mag"] = min(max(q["mag"], 0.0), 15.0)
        q["depth"] = min(max(q["depth"], 0.0), 699.0)
        q["delay"] = float(int(round(min(max(q["delay"], 0.0), 300.0))))   # 発生時刻（秒後）: 整数秒
    gz = bool(j.get("gz"))
    base, cores = sim_multi(quakes, gz=gz)

    def gen():
        def dump(o):
            return (json.dumps(o, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        yield dump({"k": "base", "v": base})              # まず震度フレームを返して再生開始
        # 各地震のEEWを交互に1ステップずつ進め、演算できた順に送る（時刻は全体の時間軸へずらす）
        streams = {i: eew_stream(q["mag"], q["depth"], q["lat"], q["lon"], tp, rt, t_end)
                   for i, (q, (tp, rt, t_end)) in enumerate(zip(quakes, cores))}
        while streams:
            for i in list(streams):
                try:
                    kind, v = next(streams[i])
                except StopIteration:
                    del streams[i]
                    continue
                except Exception as e:
                    del streams[i]
                    yield dump({"k": "error", "q": i, "v": str(e)})
                    continue
                d = quakes[i]["delay"]
                if kind == "upto":
                    v = v + d
                elif kind == "report":
                    v = _shift_report(v, d)
                yield dump({"k": kind, "q": i, "v": v})

    return Response(gen(), mimetype="application/x-ndjson",
                    headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/api/areas_geojson")
def api_areas_geojson():
    """区域の境界ポリゴン（任意）。data/areas.geojson があれば返す（無ければ404）"""
    p = DATA_DIR / "areas.geojson"
    if not p.exists():
        return _json_response({"error": "no geojson"}, 404)
    body = p.read_bytes()
    headers = {"Cache-Control": "public, max-age=3600"}
    if "gzip" in request.headers.get("Accept-Encoding", ""):
        body = gzip.compress(body, compresslevel=6)
        headers["Content-Encoding"] = "gzip"
    return Response(body, mimetype="application/json", headers=headers)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "stations": N_ST, "seafloor": int(ST_SEA.sum())}


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
  #rightcol{position:absolute;z-index:1000;right:10px;top:56px;width:350px;max-height:calc(100% - 170px);
    overflow:auto;display:flex;flex-direction:column;gap:8px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 10px;font-size:12px;line-height:1.5}
  .card h3{margin:0 0 5px;font-size:11.5px;color:var(--sub);font-weight:600}
  .card .sub{color:var(--sub);font-size:11px;margin-top:3px}
  #rank td.no{color:var(--sub);width:14px}
  .eewcard{border-width:2px;padding:5px 8px}
  .eewcard .eh{display:flex;align-items:center;gap:6px;height:24px;margin-bottom:4px}
  .eewcard .ett{flex:1;min-width:0;font-size:11.5px;color:#cfd8ea;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .eewcard .lvbadge{font-size:14px;font-weight:800;padding:1px 8px;border-radius:6px;background:#0006;min-width:74px;text-align:center;white-space:nowrap}
  .eewcard .eno{font-size:12px;font-weight:700;white-space:nowrap}
  .eewcard .tiles{display:grid;grid-template-columns:1.15fr 1fr .9fr;gap:5px}
  .eewcard .tile{background:#0005;border-radius:6px;padding:2px 3px 1px;text-align:center;height:48px;overflow:hidden}
  .eewcard .tile .k{font-size:9.5px;color:#cfd8ea;line-height:1.25}
  .eewcard .tile .v{font-size:22px;font-weight:800;line-height:1.25;font-variant-numeric:tabular-nums;white-space:nowrap}
  .eewcard .eewDep{font-size:19px;line-height:1.5}
  .eewcard .eewDep small{font-size:11px;margin-left:2px;font-weight:600}
  .eewcard .chipbig{display:inline-block;min-width:42px;border-radius:7px;padding:0 6px;font-size:20px;line-height:1.3;color:#fff;text-shadow:0 0 3px #000a}
  .eewcard .pos,.eewcard .warn{height:16px;font-size:11.5px;line-height:16px;font-variant-numeric:tabular-nums;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
  .eewcard .pos{margin-top:4px}
  .eewcard details.histbox{margin-top:2px}
  .eewcard details.histbox>summary{cursor:pointer;font-size:10.5px;color:#cfd8ea;font-weight:600;user-select:none}
  .eewcard table.hist{margin-top:0}
  .eewcard table.hist td{border-bottom:none;font-size:10.5px;padding:0 4px;color:var(--sub)}
  .eewcard.lv1{background:#12304fee;border-color:#3b9cff}
  .eewcard.lv2{background:#7a1010ee;border-color:#ff4d4d;animation:eewb 1s infinite}
  .eewcard.lv3{background:#2a0a3eee;border-color:#c04bff;animation:eewb .6s infinite}
  @keyframes eewb{50%{box-shadow:0 0 18px 4px #ff3b3b99}}
  #eewcards{display:flex;flex-direction:column;gap:8px}
  #obs{display:none;align-items:center;gap:8px;padding:5px 8px}
  #obs .ok{font-size:11px;color:var(--sub);white-space:nowrap}
  #obs .ov{font-size:12.5px;font-weight:700;white-space:nowrap}
  #obs .oa{flex:1;min-width:0;font-size:11.5px;color:var(--sub);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .epi{position:relative}
  .epi .epn{position:absolute;left:19px;top:-3px;font-size:12px;line-height:12px;font-weight:800;color:#fff;
    text-shadow:0 0 3px #000,0 0 6px #000}
  .qb{border:1px solid var(--line);border-left-width:4px;border-radius:8px;padding:2px 8px 8px;margin-bottom:8px;background:#0e131c55}
  .qb.sel{background:#1d263655;border-color:#3b4a66}
  .qh{display:flex;align-items:center;gap:6px;margin-top:6px;font-size:13px}
  .qh .qd{flex:1;font-size:10.5px;color:var(--sub)}
  .qh .qx{padding:1px 7px;font-size:12px}
  button.sec{width:100%;margin-top:2px;font-size:13px;padding:7px 10px}
  button.sec:disabled{opacity:.45;cursor:not-allowed}
  @media (max-width:700px){
    #panel{width:calc(100% - 20px);max-height:62%;top:52px;display:none}
    #panel.open{display:block}
    #toggle{display:block}
    #hud{top:10px;left:auto;right:60px;transform:none;padding:5px 9px;gap:8px;font-size:12px}
    #hud .lbl{display:none}
    #rightcol{top:92px;width:min(320px,calc(100% - 20px));max-height:calc(100% - 230px)}
  }
</style>
</head>
<body>
<div id="map"></div>
<button id="toggle">⚙ 設定</button>

<div id="panel" class="open">
  <h1>地震 強震モニタ風シミュレーター</h1>

  <h2>① 地震（複数指定できます）</h2>
  <div id="qlist"></div>
  <button id="qadd" class="sec">＋ 地震を追加</button>
  <div class="hint">選択中の地震の震源は、地図をクリックして設定できます。「発生（秒後）」で時間差を付けられます。観測点ごとの震度は、全地震の揺れのうち最大のものを表示します。緊急地震速報は地震ごとに独立して発表します。</div>

  <label style="display:flex;align-items:center;gap:6px;margin-top:10px"><input id="showPred" type="checkbox" checked style="width:auto"> 予想震度を地図に薄く塗る</label>
  <label style="display:flex;align-items:center;gap:6px;margin-top:6px"><input id="showObs" type="checkbox" checked style="width:auto"> その時点で観測された最大震度を表示</label>
  <div class="hint">黄色の○＋＝EEW推定震源／✕＝設定した震源（番号は地震番号）／地図の薄い塗り＝予想震度</div>

  <h2>② プリセット（選択中の地震に適用）</h2>
  <div class="presets" id="presets"></div>

  <button id="run" class="primary">▶ シミュレーション実行</button>
  <div id="status"></div>

  <div id="result" style="display:none">
    <div id="srcinfo" class="hint"></div>
    <h2 id="areasH">最終的な区域別最大震度（シミュレーション結果・上位）</h2>
    <table id="areas"></table>
  </div>

  <div class="foot">
    距離減衰式による簡易推定です。実際の強震モニタ（実観測波形）や気象庁の発表とは異なります。防災・避難判断には使用しないでください。<br>
    Scratch作品「【複数対応】地震の到達時間シミュレーション v1.5.0β」のロジックを参考に作成。<br>
    海底観測点（data/seafloor.csv）も陸上と同じ見た目で表示します。区域別集計・警報対象・最大震度には含めません。<br>
    地図: Tiles © Esri
  </div>
</div>

<div id="hud">
  <span><span class="lbl">経過</span><b id="hT">0.0</b> 秒</span>
  <span><span class="lbl">P波</span><b id="hP">-</b> km</span>
  <span><span class="lbl">S波</span><b id="hS">-</b> km</span>
  <span><span class="lbl">現在の最大</span><b id="hM">-</b></span>
</div>

<div id="rightcol">
  <div id="obs" class="card"><span class="ok">観測最大震度</span><span class="ov" id="obsV"></span><span class="oa" id="obsA"></span></div>
  <div id="eewcards"></div>
  <div id="rank" class="card">
    <h3>リアルタイム震度ランキング（区域別）</h3>
    <table id="rankTbl"><tr><td style="color:var(--sub)">シミュレーション実行後に表示</td></tr></table>
  </div>
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
const NUM_LABEL = ['0','1','2','3','4','5-','5+','6-','6+','7'];
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
const map = L.map('map',{zoomSnap:0.25,minZoom:4,maxZoom:11,worldCopyJump:false,zoomAnimation:false,markerZoomAnimation:false,fadeAnimation:false}).setView([36.3,137.6],5.5);
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

/* ---------- 震源（複数地震） ---------- */
const MAX_Q=5;
const QCOL=['#ff4d4d','#ffb020','#4dd2ff','#b07bff','#5fe08a'];     // 地震ごとの識別色
let QS=[{lat:35.68,lon:139.77,mag:7.0,depth:20,delay:0}], selQ=0;
const epiMarkers=[];
const epiIcon=i=>L.divIcon({className:'',html:'<div class="epi" style="color:'+QCOL[i%QCOL.length]+'">✕<span class="epn">'+(i+1)+'</span></div>',iconSize:[26,26],iconAnchor:[13,13]});
function syncMarkers(){
  while(epiMarkers.length>QS.length) map.removeLayer(epiMarkers.pop());
  QS.forEach((q,i)=>{
    if(!epiMarkers[i]) epiMarkers[i]=L.marker([q.lat,q.lon],{icon:epiIcon(i),interactive:false,keyboard:false}).addTo(map);
    else{ epiMarkers[i].setLatLng([q.lat,q.lon]); epiMarkers[i].setIcon(epiIcon(i)); }
  });
}
const clampN=(v,lo,hi)=>Math.min(Math.max(v,lo),hi);
function selectQ(i){
  selQ=i;
  Array.from($('qlist').children).forEach((b,k)=>{
    b.classList.toggle('sel',k===i);
    b.querySelector('.qd').textContent=(k===i?'地図クリックで震源を設定':'');
  });
}
function renderQuakes(){
  const box=$('qlist'); box.innerHTML='';
  QS.forEach((q,i)=>{
    const b=document.createElement('div'); b.className='qb'+(i===selQ?' sel':''); b.style.borderLeftColor=QCOL[i%QCOL.length];
    b.innerHTML='<div class="qh"><b>地震'+(i+1)+'</b><span class="qd">'+(i===selQ?'地図クリックで震源を設定':'')+'</span>'
      +(QS.length>1?'<button class="qx" title="この地震を削除">✕</button>':'')+'</div>'
      +'<div class="row"><div><label>緯度</label><input data-f="lat" type="number" step="0.01"></div>'
      +'<div><label>経度</label><input data-f="lon" type="number" step="0.01"></div></div>'
      +'<label>マグニチュード（スライダーは9.9まで／右の欄は15まで入力可）</label>'
      +'<div class="row" style="align-items:center"><div style="flex:2.2"><input data-f="magr" type="range" min="0" max="9.9" step="0.1"></div>'
      +'<div><input data-f="mag" type="number" min="0" max="15" step="0.1"></div></div>'
      +'<div class="row"><div><label>震源の深さ（km）</label><input data-f="depth" type="number" min="0" max="699" step="5"></div>'
      +'<div><label>発生（秒後）</label><input data-f="delay" type="number" min="0" max="300" step="1"></div></div>';
    const F=f=>b.querySelector('[data-f="'+f+'"]');
    F('lat').value=q.lat.toFixed(2); F('lon').value=q.lon.toFixed(2);
    F('mag').value=q.mag.toFixed(1); F('magr').value=Math.min(q.mag,9.9);
    F('depth').value=q.depth; F('delay').value=q.delay;
    const num=(f,fn)=>F(f).addEventListener('input',()=>{ const v=parseFloat(F(f).value); if(isFinite(v)) fn(v); });
    num('lat',v=>{ q.lat=v; syncMarkers(); });
    num('lon',v=>{ q.lon=v; syncMarkers(); });
    num('mag',v=>{ q.mag=clampN(v,0,15); F('magr').value=Math.min(q.mag,9.9); });
    num('depth',v=>{ q.depth=clampN(v,0,699); });
    num('delay',v=>{ q.delay=Math.round(clampN(v,0,300)); });
    F('magr').addEventListener('input',()=>{ q.mag=parseFloat(F('magr').value); F('mag').value=q.mag.toFixed(1); });
    F('mag').addEventListener('change',()=>{ F('mag').value=q.mag.toFixed(1); });
    b.addEventListener('focusin',()=>{ if(selQ!==i) selectQ(i); });
    b.addEventListener('click',()=>{ if(selQ!==i) selectQ(i); });
    const x=b.querySelector('.qx');
    if(x) x.onclick=ev=>{
      ev.stopPropagation(); QS.splice(i,1);
      if(i<selQ) selQ--; else if(i===selQ) selQ=Math.min(selQ,QS.length-1);
      renderQuakes(); syncMarkers();
    };
    box.appendChild(b);
  });
  $('qadd').disabled=QS.length>=MAX_Q;
  $('qadd').textContent=QS.length>=MAX_Q?'地震は最大'+MAX_Q+'個までです':'＋ 地震を追加';
}
function setEpi(lat,lon,pan){
  lat=Math.round(lat*100)/100; lon=Math.round(lon*100)/100;
  const q=QS[selQ]; q.lat=lat; q.lon=lon;
  const b=$('qlist').children[selQ];
  if(b){ b.querySelector('[data-f="lat"]').value=lat.toFixed(2); b.querySelector('[data-f="lon"]').value=lon.toFixed(2); }
  syncMarkers();
  if(pan) map.panTo([lat,lon]);
}
map.on('click',e=>setEpi(e.latlng.lat,e.latlng.lng,false));
$('qadd').onclick=()=>{
  if(QS.length>=MAX_Q) return;
  const c=map.getCenter();
  QS.push({lat:Math.round(c.lat*100)/100,lon:Math.round(c.lng*100)/100,mag:6.0,depth:10,delay:0});
  selQ=QS.length-1; renderQuakes(); syncMarkers();
};
renderQuakes(); syncMarkers();

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
  b.onclick=()=>{
    const q=QS[selQ]; q.lat=p.lat; q.lon=p.lon; q.mag=p.m; q.depth=p.d;     // 発生時刻（秒後）は据え置き
    renderQuakes(); syncMarkers(); map.panTo([p.lat,p.lon]);
  };
  $('presets').appendChild(b);
});
{ const b=document.createElement('button'); b.textContent='2地震の例（能登＋首都直下）';
  b.onclick=()=>{
    QS=[{lat:37.49,lon:137.27,mag:7.6,depth:15,delay:0},{lat:35.68,lon:139.77,mag:7.3,depth:30,delay:20}];
    selQ=0; renderQuakes(); syncMarkers(); map.setView([36.3,138.3],6);
  };
  $('presets').appendChild(b); }
$('toggle').onclick=()=>$('panel').classList.toggle('open');
if(window.innerWidth<=700) $('panel').classList.remove('open');

/* ---------- 観測点 ---------- */
let N=0, SLON=null, SLAT=null, PX=null, PY=null, VALS=null, viewDirty=true, AREA=null, ANAMES={}, AM=null, SEA=null, SEAIDX=[];
fetch('/api/stations').then(r=>r.json()).then(j=>{
  SLON=Float32Array.from(j.lon); SLAT=Float32Array.from(j.lat); N=SLON.length;
  AREA=Int32Array.from(j.area); ANAMES=j.anames; SEA=Uint8Array.from(j.sea||new Array(N).fill(0)); SEAIDX=[]; for(let i=0;i<N;i++) if(SEA[i]) SEAIDX.push(i); let mx=0; for(const a of AREA) if(a>mx) mx=a; AM=new Float32Array(mx+1);
  PX=new Float32Array(N); PY=new Float32Array(N); VALS=new Float32Array(N).fill(-3.1); viewDirty=true; loadGeo();
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
function drawCircle(lat,lon,rkm,style,width){
  if(!(rkm>0.5)) return;
  const pts=circlePts(lat,lon,rkm,180);
  ctx.beginPath(); ctx.moveTo(pts[0].x,pts[0].y);
  for(let i=1;i<pts.length;i++) ctx.lineTo(pts[i].x,pts[i].y);
  ctx.strokeStyle=style; ctx.lineWidth=width; ctx.stroke();
}
function drawCircleAt(lat,lon,rkm,style,width,dash){
  if(!(rkm>0.5)) return;
  const pts=circlePts(lat,lon,rkm,180);
  ctx.beginPath(); ctx.moveTo(pts[0].x,pts[0].y);
  for(let i=1;i<pts.length;i++) ctx.lineTo(pts[i].x,pts[i].y);
  ctx.setLineDash(dash||[]); ctx.strokeStyle=style; ctx.lineWidth=width; ctx.stroke(); ctx.setLineDash([]);
}
function frac(x){ return x-Math.floor(x); }

/* ---------- シミュレーション状態 ---------- */
let sim=null, Q=null, tcur=0, playing=false, lastTs=0, runCtl=null, waitFlag=false;
// EEWの演算が追いついていない地震があれば、その地震の確定済み時刻までしか再生しない
const limT=()=>{ if(!sim) return 0; let lim=sim.tEnd; for(const q of sim.quakes) if(!q.eewDone) lim=Math.min(lim,q.eewUpto||0); return lim; };

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

/* ---------- 区域境界ポリゴン（data/areas.geojson があるときだけ使用） ---------- */
let GEO=null;   // [{id, rings:[Float32Array [wx,wy,...]]}]  wx,wy は正規化メルカトル(0..1)
function loadGeo(){
  fetch('/api/areas_geojson').then(r=>r.ok?r.json():null).then(gj=>{
    if(!gj||!gj.features) return;
    const norm=s=>String(s||'').normalize('NFKC').replace(/\s+/g,'');
    const byName={}; for(const k in ANAMES) byName[norm(ANAMES[k])]=parseInt(k);
    const KEYS=['name','Name','NAME','area_name','AREA_NAME','regionName','Area','area'];
    const out=[]; let hit=0;
    const toW=(lon,lat)=>{ const s=Math.sin(lat*Math.PI/180);
      return [(lon+180)/360, 0.5-Math.log((1+s)/(1-s))/(4*Math.PI)]; };
    for(const f of gj.features){
      const p=f.properties||{}; let id=null;
      for(const k of KEYS){ if(p[k]!=null && byName[norm(p[k])]!=null){ id=byName[norm(p[k])]; break; } }
      if(id==null && p.code!=null && ANAMES[p.code]!=null) id=parseInt(p.code);
      if(id==null||!f.geometry) continue;
      const polys=f.geometry.type==='Polygon'?[f.geometry.coordinates]:(f.geometry.type==='MultiPolygon'?f.geometry.coordinates:[]);
      const rings=[];
      for(const poly of polys) for(const ring of poly){
        const arr=[]; let lx=-9,ly=-9;
        for(let i=0;i<ring.length;i++){            // 約300m未満の頂点は間引いて描画を軽くする
          const w=toW(ring[i][0],ring[i][1]);
          if(i===0||i===ring.length-1||Math.abs(w[0]-lx)+Math.abs(w[1]-ly)>2e-5){ arr.push(w[0],w[1]); lx=w[0]; ly=w[1]; }
        }
        if(arr.length>=6) rings.push(Float32Array.from(arr));
      }
      if(rings.length){ out.push({id:id,rings:rings}); hit++; }
    }
    GEO=hit?out:null;
    console.log('areas.geojson: '+hit+'/'+gj.features.length+' features matched');
  }).catch(()=>{});
}
function drawPredPolygons(pv,z){
  const val=new Float32Array(AM.length).fill(-99);
  for(let i=0;i<N;i++) val[AREA[i]]=pv[i]/10;
  const s=256*Math.pow(2,z), p0=map.latLngToContainerPoint([0,0]);
  const ox=p0.x-0.5*s, oy=p0.y-0.5*s, W=cv.width/(window.devicePixelRatio||1), H=cv.height/(window.devicePixelRatio||1);
  ctx.save(); ctx.globalAlpha=0.38;
  for(let c=1;c<10;c++){
    ctx.fillStyle=clsColor(c); ctx.beginPath(); let any=false;
    for(const g of GEO){
      const v=val[g.id]; if(!(v>=0.5) || shindoCls(v)!==c) continue;
      for(const r of g.rings){
        let minx=1e9,maxx=-1e9,miny=1e9,maxy=-1e9;
        for(let i=0;i<r.length;i+=2){ const x=ox+r[i]*s, y=oy+r[i+1]*s;
          if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
          if(x<minx)minx=x; if(x>maxx)maxx=x; if(y<miny)miny=y; if(y>maxy)maxy=y; }
        ctx.closePath(); any=true;
      }
    }
    if(any) ctx.fill('evenodd');
  }
  ctx.restore();
}

let predCv=null;
function predOf(rep){
  if(!rep._pv){                                        // 予想震度(×10, int8)を初回だけ展開
    const bin=atob(rep.pq); const a=new Int8Array(bin.length);
    for(let i=0;i<bin.length;i++) a[i]=(bin.charCodeAt(i)<<24)>>24;
    rep._pv=a;
  }
  return rep._pv;
}
function mergedPred(){                                 // 各地震の現在の予想震度を、観測点ごとに最大で合成
  const list=[]; for(const cur of eewCurs){ if(cur&&cur.pq){ const pv=predOf(cur); if(pv.length===N) list.push(pv); } }
  if(!list.length) return null;
  if(list.length===1) return list[0];
  const out=Int8Array.from(list[0]);
  for(let k=1;k<list.length;k++){ const pv=list[k]; for(let i=0;i<N;i++) if(pv[i]>out[i]) out[i]=pv[i]; }
  return out;
}
function drawPredFill(pv,z){
  if(GEO && GEO.length){ drawPredPolygons(pv,z); return; }   // 区域境界データがあればそちらで塗る
  const dpr=window.devicePixelRatio||1, w=cv.width, h=cv.height;
  if(!predCv) predCv=document.createElement('canvas');
  if(predCv.width!==w||predCv.height!==h){ predCv.width=w; predCv.height=h; }
  const g=predCv.getContext('2d');
  g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,w/dpr,h/dpr);
  const mpp=156543.03*Math.cos(36*Math.PI/180)/Math.pow(2,z);   // m/px
  const r=Math.max(6,22000/mpp);                                // 観測点1つが受け持つ半径≈22km
  const byCls=[]; for(let k=0;k<10;k++) byCls.push([]);
  for(let i=0;i<N;i++){ const v=pv[i]/10; if(v>=0.5) byCls[shindoCls(v)].push(i); }
  for(let k=1;k<10;k++){                                        // 弱い階級から順に、強い階級を上書き
    const b=byCls[k]; if(!b.length) continue;
    g.fillStyle=clsColor(k); g.beginPath();
    for(let j=0;j<b.length;j++){ const i=b[j]; g.moveTo(PX[i]+r,PY[i]); g.arc(PX[i],PY[i],r,0,6.2832); }
    g.fill();
  }
  ctx.save(); ctx.setTransform(1,0,0,1,0,0); ctx.globalAlpha=0.3;
  ctx.drawImage(predCv,0,0); ctx.restore();
}

function draw(ts){
  const s=map.getSize(); ctx.clearRect(0,0,s.x,s.y);
  if(!N) return;
  updatePoints();   // 毎フレーム再投影（ズーム/パン中も地図とずれない）
  if(sim) valuesAt(tcur);
  else { const tk=Math.floor((ts||0)/160); for(let i=0;i<N;i++) VALS[i]=-3.1+(frac(Math.sin(i*12.9898+tk*78.233)*43758.5453)-0.5)*0.14; }

  const z=map.getZoom(), r0=Math.max(2.0,(z-3.6)*0.95);
  const buckets=new Array(LUT_N); for(let k=0;k<LUT_N;k++) buckets[k]=[];
  let vmax=-99, vmaxI=-1; AM.fill(-99);
  for(let i=0;i<N;i++){
    const v=VALS[i];
    if(!(SEA&&SEA[i])){ if(v>vmax){ vmax=v; vmaxI=i; } const ar=AREA[i]; if(v>AM[ar]) AM[ar]=v; }   // 海底観測点は最大・ランキングに含めない
    let k=Math.round((v-LUT_MIN)/LUT_STEP); if(k<0)k=0; if(k>=LUT_N)k=LUT_N-1;
    buckets[k].push(i);
  }
  // EEWの予想震度で地図を薄く塗りつぶす（観測点ごとの予想震度の階級色。高い階級を上に重ねる）
  if(sim && $('showPred').checked){ const pv=mergedPred(); if(pv) drawPredFill(pv,z); }
  // 震度1以上の観測点は「円の中に震度階級の数字」で表示（ズームに応じて円を拡大）
  const rn=Math.min(16,5.5+(z-4)*1.2), showTxt=rn>=6.5;
  ctx.textAlign='center'; ctx.textBaseline='middle';
  for(let k=0;k<LUT_N;k++){
    const b=buckets[k]; if(!b.length) continue;
    const v=LUT_MIN+k*LUT_STEP;
    const num=v>=0.49, cl=num?shindoCls(v+0.01):0;
    const r=num?rn*(cl>=5?1.2:1.0):r0*(v<0?0.85:1.0);
    ctx.fillStyle=LUT[k]; ctx.beginPath();
    for(let j=0;j<b.length;j++){ const i=b[j]; ctx.moveTo(PX[i]+r,PY[i]); ctx.arc(PX[i],PY[i],r,0,6.2832); }
    ctx.fill();
    if(num){
      ctx.strokeStyle='rgba(255,255,255,.85)'; ctx.lineWidth=1; ctx.stroke();
      if(showTxt){
        ctx.font='bold '+Math.round(r*(cl>=5?1.0:1.25))+'px sans-serif';
        ctx.fillStyle=(cl===2||cl===3)?'#111':'#fff';
        const t=NUM_LABEL[cl];
        for(let j=0;j<b.length;j++){ const i=b[j]; ctx.fillText(t,PX[i],PY[i]+0.5); }
      }
    }
  }
  updateRank(ts);
  if(sim){
    updateEew();
    sim.quakes.forEach(q=>{                             // 各地震の実際のP波・S波（発生後のみ）
      const tl=tcur-q.delay; if(tl<0) return;
      drawCircle(q.lat,q.lon,interp(q.P,tl),'rgba(90,200,255,.95)',1.6);
      drawCircle(q.lat,q.lon,interp(q.S,tl),'rgba(255,95,95,.98)',2.2);
    });
    eewCurs.forEach(cur=>{
      if(!cur) return;
      if(sim.ett && cur.t0!=null){                      // EEW推定震源から見たP波・S波（推定、薄く破線）
        const tt=sim.ett[cur.depth], tau=tcur-cur.t0;
        if(tt && tau>0){
          drawCircleAt(cur.lat,cur.lon,interp(tt[0],tau),'rgba(90,200,255,.38)',1.3,[6,5]);
          drawCircleAt(cur.lat,cur.lon,interp(tt[1],tau),'rgba(255,120,120,.45)',1.6,[6,5]);
        }
      }
      const p=map.latLngToContainerPoint([cur.lat,cur.lon]);
      ctx.strokeStyle='#ffd23b'; ctx.lineWidth=2.5; ctx.beginPath(); ctx.arc(p.x,p.y,9,0,6.2832);
      ctx.moveTo(p.x-14,p.y); ctx.lineTo(p.x+14,p.y); ctx.moveTo(p.x,p.y-14); ctx.lineTo(p.x,p.y+14); ctx.stroke();
    });
    $('hT').textContent=tcur.toFixed(1);
    const hv=f=>sim.quakes.map(q=>{ const tl=tcur-q.delay; return tl<0?'-':interp(q[f],tl).toFixed(0); }).join('／');
    $('hP').textContent=hv('P');
    $('hS').textContent=hv('S');
    const hm=$('hM'); const m=Math.max(vmax,-3);
    updateObs(m,vmaxI);
    if(m<0.5){ hm.textContent='—'; hm.style.color='#93a0b8'; }
    else { hm.textContent=m.toFixed(1)+'（'+CLASS_LABEL[shindoCls(m)]+'）'; hm.style.color=clsColor(shindoCls(m)); }
  }
}

let lastRank=0, eewCurs=[], eewCards=[];
function updateRank(ts){
  if(!sim||!AM||(ts||0)-lastRank<250) return; lastRank=ts||0;
  const arr=[]; for(let a=0;a<AM.length;a++){ if(AM[a]>=0.5) arr.push([a,AM[a]]); }
  arr.sort((p,q)=>q[1]-p[1]);
  let h='';
  arr.slice(0,10).forEach((e,i)=>{
    const c=shindoCls(e[1]);
    h+='<tr><td class="no">'+(i+1)+'</td><td>'+(ANAMES[e[0]]||('区域'+e[0]))+'</td><td class="num">'+e[1].toFixed(1)+
       '</td><td class="num"><span class="chip" style="background:'+clsColor(c)+'">'+CLASS_LABEL[c]+'</span></td></tr>';
  });
  $('rankTbl').innerHTML=h||'<tr><td style="color:#93a0b8">震度1以上の観測なし</td></tr>';
}
const EEW_LBL=['待機中','予報','警報','特別警報'];
function eewSet(el,html){ if(el._h!==html){ el.innerHTML=html; el._h=html; } }
function eewTxt(el,t){ if(el._h!==t){ el.textContent=t; el.title=t; el._h=t; } }
// その時点で観測された最大震度（陸上観測点）。チェックボックスで表示/非表示
function updateObs(m,i){
  const box=$('obs'), on=!!sim && $('showObs').checked;
  box.style.display=on?'flex':'none'; if(!on) return;
  if(m<0.5||i<0){ eewSet($('obsV'),'<span style="color:#93a0b8">—</span>'); eewTxt($('obsA'),''); return; }
  const c=shindoCls(m), tc=(c===2||c===3)?'#111':'#fff';
  eewSet($('obsV'),'<span class="chip" style="background:'+clsColor(c)+';color:'+tc+'">'+CLASS_LABEL[c]+'</span> '+m.toFixed(1));
  eewTxt($('obsA'),ANAMES[AREA[i]]||('区域'+AREA[i]));
}
// 地震の数だけ緊急地震速報カードを作る（報の履歴は初期状態で閉じる）
function buildEewCards(n){
  const box=$('eewcards'); box.innerHTML=''; eewCards=[]; eewCurs=new Array(n).fill(null);
  for(let i=0;i<n;i++){
    const c=document.createElement('div'); c.className='card eewcard lv0';
    c.innerHTML='<div class="eh"><span class="ett">緊急地震速報'+(n>1?'　<span style="color:'+QCOL[i%QCOL.length]+'">●</span>地震'+(i+1):'')+'</span>'
      +'<span class="lvbadge r-lv">待機中</span><span class="eno r-no"></span></div>'
      +'<div class="tiles">'
      +'<div class="tile"><div class="k">予想最大震度</div><div class="v r-max"><span class="chipbig" style="background:#3a4357">-</span></div></div>'
      +'<div class="tile"><div class="k">マグニチュード</div><div class="v r-mag">-</div></div>'
      +'<div class="tile"><div class="k">深さ</div><div class="v eewDep r-dep">-</div></div></div>'
      +'<div class="pos r-pos">シミュレーション実行後に表示</div>'
      +'<div class="warn r-warn"></div>'
      +'<details class="histbox"><summary>報の履歴</summary><table class="hist r-hist"></table></details>';
    box.appendChild(c);
    const els={}; ['lv','no','max','mag','dep','pos','warn','hist'].forEach(k=>els[k]=c.querySelector('.r-'+k));
    eewCards.push({el:c, r:k=>els[k]});
  }
}
buildEewCards(1);
function updateEew(){ sim.quakes.forEach((qk,i)=>updateEewCard(qk,i)); }
function updateEewCard(qk,i){
  const C=eewCards[i]; if(!C) return;
  const card=C.el, R=C.r, e=qk.eew;
  let cur=null, lvmax=0;
  if(e){ for(let k=0;k<e.reports.length;k++){ const r=e.reports[k]; if(r.t>tcur) break; cur=r; if(r.lv>lvmax) lvmax=r.lv; } }
  eewCurs[i]=cur;
  if(!cur){
    card.className='card eewcard lv0';
    eewSet(R('lv'),e?'発表前':'発表なし'); eewSet(R('no'),'');
    eewSet(R('max'),'<span class="chipbig" style="background:#3a4357">-</span>');
    eewSet(R('mag'),'-'); eewSet(R('dep'),'-');
    eewTxt(R('pos'),e?(tcur<qk.delay?'発生前（'+qk.delay+'秒後に発生）':'観測点でP波を検知中…'):'発表基準に達しませんでした');
    eewTxt(R('warn'),''); eewSet(R('hist'),'');
    return;
  }
  card.className='card eewcard lv'+lvmax;
  eewSet(R('lv'),EEW_LBL[lvmax]);
  eewSet(R('no'),'第'+cur.no+'報'+(cur.final?'（最終）':''));
  const tc=(cur.maxCls===2||cur.maxCls===3)?'#111':'#fff';
  eewSet(R('max'),'<span class="chipbig" style="background:'+clsColor(cur.maxCls)+';color:'+tc+'">'+cur.maxLabel+'</span>');
  eewSet(R('mag'),'M'+cur.mag.toFixed(1));
  eewSet(R('dep'),cur.depth+'<small>km</small>');
  eewTxt(R('pos'),(cur.epiName||'震源付近')+'　N'+cur.lat.toFixed(1)+'° E'+cur.lon.toFixed(1)+'° ／ 検知'+cur.nSt+'点'+(cur.plum?'／PLUM':''));
  // 警報対象は地名だけを1行で表示（区域別の予想震度は表示しない）。範囲が広ければ地域区分でまとめる
  let w='';
  if(lvmax>=2){
    if(cur.areas.length){
      const byRegion=new Set(cur.areas.map(a=>a.pref)).size>6, best={};
      cur.areas.forEach(a=>{ const key=byRegion?a.region:a.pref; if(!(key in best)||a.cls>best[key]) best[key]=a.cls; });
      w='警報対象：'+Object.keys(best).sort((p,q)=>best[q]-best[p]).join('、');
    }else w='警報対象：該当地域なし';
  }
  eewTxt(R('warn'),w);
  const hist=e.reports.filter(r=>r.t<=tcur).slice(-4).reverse();
  eewSet(R('hist'),'<tr><td>報</td><td>経過</td><td>M</td><td>深さ</td><td>最大</td></tr>'
    +hist.map(r=>'<tr><td>'+r.no+'</td><td>'+r.t+'秒</td><td>'+r.mag.toFixed(1)+'</td><td>'+r.depth+'km</td><td>'+r.maxLabel+'</td></tr>').join(''));
}

function tick(ts){
  if(sim && playing){
    const d=Math.min((ts-lastTs)/1000,0.1);
    tcur+=d*parseFloat($('speed').value);
    const lim=limT();                                  // EEWの演算が確定している時刻（これ以降は再生できない）
    const speed=parseFloat($('speed').value), done=lim>=sim.tEnd-0.01;
    if(waitFlag){
      // 演算待ちで止まっている間は、先読みが少し貯まる（または演算完了）まで再開しない。
      // 1フレームごとに止まったり進んだりを繰り返さず、まとめて待ってから滑らかに再生する。
      if(done || lim-tcur>=Math.max(3,2*speed)){
        waitFlag=false;
        if(!sim.quakes.every(x=>x.eewDone)) $('status').textContent='再生中（EEW演算中）';
      }
    }
    if(!waitFlag){
      tcur+=d*speed;
      if(tcur>lim){
        tcur=lim;
        if(!done){ waitFlag=true; $('status').textContent='EEW演算待ち…（先読みを確保中）'; }
      }
    }
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
$('seek').addEventListener('input',()=>{ if(!sim) return; tcur=Math.min(parseInt($('seek').value)/10,limT()); $('seek').value=Math.round(tcur*10); $('tlabel').textContent=tcur.toFixed(1)+' s'; });

/* ---------- 実行 ---------- */
$('run').onclick=async()=>{
  const body={quakes:QS.map(q=>({lat:q.lat,lon:q.lon,mag:q.mag,depth:q.depth,delay:q.delay})),
              gz:(typeof DecompressionStream!=='undefined')};
  if(!QS.every(q=>[q.lat,q.lon,q.mag,q.depth,q.delay].every(isFinite))){ $('status').textContent='入力値を確認してください'; return; }
  if(runCtl) runCtl.abort(); const ctl=runCtl=new AbortController();
  $('run').disabled=true; $('status').textContent='計算中…'; playing=false; waitFlag=false;
  const handle=async(m)=>{
    if(ctl!==runCtl) return;
    const k=m.k, v=m.v;
    if(k==='base'){                                    // 震度フレーム到着 → すぐ再生開始
      const bin=atob(v.q); let u=new Uint8Array(bin.length);
      for(let i=0;i<bin.length;i++) u[i]=bin.charCodeAt(i);
      if(v.gz) u=new Uint8Array(await new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
      Q=new Int8Array(u.buffer,u.byteOffset,u.length);
      v.quakes.forEach(q=>{ q.eew={reports:[]}; q.eewUpto=0; q.eewDone=false; });
      sim=v; tcur=0; buildEewCards(v.quakes.length);
      $('seek').disabled=false; $('seek').max=Math.round(v.tEnd*10); $('seek').value=0; $('tlabel').textContent='0.0 s';
      showResult(v);
      $('status').textContent='再生中（EEW演算中）';
      playing=true; $('play').textContent='⏸'; $('run').disabled=false;
      if(window.innerWidth<=700) $('panel').classList.remove('open');
    }else if(!sim){ return; }
    else{
      const qk=sim.quakes[m.q]; if(!qk) return;
      const fin=()=>{ if(!sim.err && sim.quakes.every(x=>x.eewDone)) $('status').textContent='完了（'+sim.nf+'フレーム／リプレイ可）'; };
      if(k==='upto'){ qk.eewUpto=v; }
      else if(k==='report'){ qk.eew.reports.push(v); }
      else if(k==='final'){ const r=qk.eew.reports.find(x=>x.no===v); if(r) r.final=true; }
      else if(k==='none'){ qk.eew=null; qk.eewDone=true; fin(); }
      else if(k==='done'){ qk.eewDone=true; fin(); }
      else if(k==='error'){ qk.eewDone=true; sim.err=true; $('status').textContent='EEW演算エラー（地震'+(m.q+1)+'）: '+v; }
    }
  };
  try{
    const r=await fetch('/api/simulate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:ctl.signal});
    if(!r.ok){ let msg='エラーが発生しました'; try{ msg=(await r.json()).error||msg; }catch(_){} $('status').textContent=msg; return; }
    const rd=r.body.getReader(), dec=new TextDecoder(); let buf='';
    for(;;){
      const {value,done}=await rd.read(); if(done) break;
      buf+=dec.decode(value,{stream:true}); let k;
      while((k=buf.indexOf('\n'))>=0){ const line=buf.slice(0,k); buf=buf.slice(k+1); if(line.trim()) await handle(JSON.parse(line)); }
    }
    if(buf.trim()) await handle(JSON.parse(buf));
  }catch(e){ if(e.name!=='AbortError') $('status').textContent='通信エラー: '+e; }
  finally{ if(ctl===runCtl) $('run').disabled=false; }
};

function showResult(j){
  $('result').style.display='block';
  const many=j.quakes.length>1;
  $('srcinfo').style.whiteSpace='pre-line';
  $('srcinfo').textContent=j.quakes.map((q,i)=>(many?'地震'+(i+1)+'：':'')+'震源タイプ: '+q.src.type+(q.src.name?'（'+q.src.name+'）':'')
    +(q.src.L>0?' ／ 想定断層長 約'+q.src.L+'km（断層の広がりを考慮して震度を計算）':'')).join('\n');
  $('areasH').textContent=many?'最終的な区域別最大震度（全地震の最大値・上位）':'最終的な区域別最大震度（シミュレーション結果・上位）';
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
