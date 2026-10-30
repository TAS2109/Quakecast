# -*- coding: utf-8 -*-
"""
Quakecast

【このプロジェクトの目標】
  JQuake のような「強震モニタ・地震津波情報等総合配信サイト」の再現を目指している。
  強震モニタ（リアルタイム震度）・緊急地震速報・津波警報/注意報・津波情報などを、
  気象庁の実際の発表内容・発表速度にできるだけ合わせて再現する。

「リアルタイム地震ビューアー」(kotoho7様作)を参考に作成した
Webアプリ（震度計算ロジックはPythonへ移植）。
  効果音・読み上げの場面構成も同作品を参考にしているが、音そのものはブラウザの
  Web Audio で新規に合成したもの（元の音声ファイルは使用していない）。読み上げは
  ブラウザ標準の日本語音声（SpeechSynthesis）。

  ・震源（地図クリック）・マグニチュード・深さを指定
  ・全国1748観測点のリアルタイム震度が、P波・S波の伝播とともに
    広がっていく様子を強震モニタ風にアニメーション表示

構成
  app.py            … 計算エンジン + API + フロントエンド(HTML/CSS/JS) のすべて
  data/stations.csv … 観測点（経度・緯度・地盤増幅ARV・地盤深さ・所属区域）  [不変データ]
  data/areas.csv    … 区域名                                              [不変データ]
  data/jma2001.json … JMA2001走時表の近似式係数                            [不変データ]
  data/deep_params.json … （任意）深発地震の減衰係数。calibrate_deep.py が実観測から較正して出力  [任意]

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
# 小笠原村（父島・母島・硫黄島・南鳥島）の観測点を補う
#   stations.csv では小笠原村付近の観測点が実際より少ないため、概略の座標で追加する。
#   すでに1.5km以内に観測点がある地点は重複させない。地盤増幅(ARV)・地盤深さは最寄りの陸上観測点5点の
#   中央値を仮置きとする。所属区域は「小笠原」で始まる区域（無ければ最寄りの観測点の区域）。
# ---------------------------------------------------------------------
_OGASAWARA_EXTRA = [
    # (名称, 緯度, 経度)  ※概略座標
    ("父島 二見", 27.0945, 142.1925), ("父島 扇浦", 27.0800, 142.1995), ("父島 小港", 27.0650, 142.2150),
    ("父島 中央山", 27.0820, 142.2230), ("父島 初寝浦", 27.1090, 142.2300), ("父島 長崎", 27.1180, 142.2120),
    ("母島 沖村", 26.6340, 142.1580), ("母島 乳房山", 26.6500, 142.1650), ("母島 北村", 26.6710, 142.1480),
    ("硫黄島", 24.7800, 141.3200), ("南鳥島", 24.2900, 153.9800),
]


def _append_ogasawara():
    global ST_LON, ST_LAT, ST_ARV, ST_DEP, ST_AREA, N_ST
    oga = [k for k, v in AREA_NAMES.items() if str(v).startswith("小笠原")]
    lon, lat, arv, dep, area = [], [], [], [], []
    for nm, la, lo in _OGASAWARA_EXTRA:
        kx = 111.32 * math.cos(math.radians(la))
        dd = np.hypot((ST_LON - lo) * kx, (ST_LAT - la) * 110.57)
        if float(dd.min()) < 1.5:
            continue
        near = np.argsort(dd)[:5]
        lon.append(lo); lat.append(la)
        arv.append(float(np.median(ST_ARV[near]))); dep.append(float(np.median(ST_DEP[near])))
        area.append(int(oga[0]) if oga else int(ST_AREA[near[0]]))
    if not lon:
        return
    ST_LON = np.concatenate([ST_LON, lon]); ST_LAT = np.concatenate([ST_LAT, lat])
    ST_ARV = np.concatenate([ST_ARV, arv]); ST_DEP = np.concatenate([ST_DEP, dep])
    ST_AREA = np.concatenate([ST_AREA, np.array(area, dtype=np.int64)])
    N_ST = len(ST_LON)
    print("[小笠原] 観測点 %d点を追加（計%d点）" % (len(lon), N_ST))


_append_ogasawara()


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


# ---------------------------------------------------------------------
# 深発地震の異常震域（スラブ導波モデル）
#   沈み込む冷たい海洋プレート（スラブ）の内部は地震波の減衰が小さい(Qが高い)ため、スラブ内の深発地震の
#   波は、スラブ沿いに遠くまで届き、スラブが浅くなる太平洋側（前弧側）で強く揺れる。一方、スラブより上の
#   マントルウェッジは、スラブが深くなる側（火山フロントより背弧側）で高温・低Qになり、そこを通る波は
#   大きく減衰する（→ 震源の真上でも日本海側は小さい）。
#   観測点ごとに、
#     ① 震源 → 観測点直下のスラブ上面 : スラブ内を伝わる区間（減衰が小さい: c_slab）
#     ② スラブ上面 → 地殻の底        : マントルウェッジを上がる区間（スラブが浅いほど低温で減衰小: c_cold、
#                                       深いほど高温で減衰大: c_hot）
#   の2区間で、距離減衰式に元から含まれる非弾性減衰(-0.002/km)との差を log10(PGV) に加える。
#   観測点直下のスラブ上面の深さは、海溝軸（TRENCHESを連ねた鎖）からの距離と概略の沈み込み形状から求める
#   （火山フロントの座標は使わない）。
#   ※係数は暫定値。実観測（気象庁の震度データベース等）に合わせるには calibrate_deep.py で較正し、
#     結果を data/deep_params.json に置けば自動で読み込む。
#   ※緊急地震速報の予測はこの効果を見込まない（実際にも予測が難しいため。PLUM法で補う）
# ---------------------------------------------------------------------
DEEP_ANOM_D0, DEEP_ANOM_D1 = 40.0, 100.0   # 異常震域が効き始める深さ〜最大になる深さ[km]
# 深さ項(0.0038*深さ)は深さ-10が d0[km] を超えると傾きを slope 倍に抑える（DEEP_Q の "d0","slope"。較正対象）
#（元の式のまま深発地震に使うと遠方まで震度が過大になる）
DEEP_Q = {
    "c_slab": 0.00076,
    "c_cold": 0.0049,
    "c_hot": 0.0033,
    "zvf": 261.0,
    "wvf": 20.0,
    "zcrust": 35.0,
    "lmax": 3000.0,
    "d0": 30.0,
    "slope": 0.0,
}

_QP = DATA_DIR / "deep_params.json"
if _QP.exists():                                   # 較正結果があれば上書き
    try:
        with open(_QP, encoding="utf-8") as _f:
            DEEP_Q.update({k: float(v) for k, v in json.load(_f).items() if k in DEEP_Q})
        print("[deep_params.json] 深発地震の減衰係数を読み込み:", DEEP_Q)
    except Exception as e:
        print("[deep_params.json] 読み込み失敗:", e)

# 沈み込む板ごとの海溝鎖（連続した線。進行方向の右が陸側）
_SLAB_CHAINS = {
    "PAC": [(155.0, 48.5), (150.5, 45.5), (147.3, 43.3), (145.0, 41.3), (144.0, 39.5), (143.7, 38.0),
            (143.0, 36.5), (142.5, 35.3), (142.1, 33.5), (141.9, 31.5), (142.3, 29.0), (143.0, 26.0)],   # 千島〜日本〜伊豆・小笠原
    "PSP": [(138.5, 34.6), (137.7, 33.9), (136.7, 33.3), (135.7, 32.7), (134.5, 32.2), (133.2, 31.7),
            (132.2, 31.2), (131.0, 29.7), (129.8, 28.0), (128.5, 26.0), (126.5, 24.2), (124.5, 23.4),
            (122.5, 23.2)],                                                                              # 南海〜琉球
    "SAG": [(139.2, 34.9), (139.9, 34.4), (140.9, 34.3)],                                                # 相模
}
# スラブ上面の深さ[km]  vs  海溝軸からの陸側水平距離[km]（概略）
_SLAB_TAB = {
    "pac": ([0, 50, 150, 250, 350, 500, 700, 900, 1050, 4000], [0, 10, 35, 90, 150, 250, 400, 520, 600, 660]),
    "izu": ([0, 60, 120, 200, 250, 300, 350, 500, 4000],       [0, 30, 60, 120, 300, 500, 640, 680, 700]),
    "psp": ([0, 60, 150, 250, 350, 450, 600, 4000],            [0, 20, 45, 80, 130, 200, 300, 450]),
    "ryu": ([0, 100, 200, 300, 450, 600, 4000],                [0, 40, 100, 160, 250, 350, 600]),
}


def _extend_line(pts, km=900.0):
    """折れ線の両端を、端の線分の向きにkm延長する（端付近の距離計算が不連続にならないように）"""
    def ext(xp, yp, xq, yq):
        kx = 111.32 * math.cos(math.radians(yq))
        vx, vy = (xq - xp) * kx, (yq - yp) * 110.57
        n = math.hypot(vx, vy) or 1.0
        return (xq + vx / n * km / kx, yq + vy / n * km / 110.57)
    return [ext(pts[1][0], pts[1][1], pts[0][0], pts[0][1])] + list(pts) + [ext(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1])]


def _signed_dist_km(pts, lon, lat):
    """折れ線ptsまでの符号付き距離[km]（進行方向の左が正）。lon/latは配列可"""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    kx = 111.32 * np.cos(np.radians(lat))
    ky = 110.57
    best = np.full(lon.shape, 1e9)
    sgn = np.ones(lon.shape)
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        ax, ay = (x1 - lon) * kx, (y1 - lat) * ky
        bx, by = (x2 - lon) * kx, (y2 - lat) * ky
        vx, vy = bx - ax, by - ay
        l2 = np.maximum(vx * vx + vy * vy, 1e-9)
        t = np.clip(-(ax * vx + ay * vy) / l2, 0.0, 1.0)
        d = np.hypot(ax + t * vx, ay + t * vy)
        cr = vx * (-ay) - vy * (-ax)
        m = d < best
        best = np.where(m, d, best)
        sgn = np.where(m, np.where(cr >= 0.0, 1.0, -1.0), sgn)
    return best * sgn


_SLAB_EXT = {k: _extend_line(v) for k, v in _SLAB_CHAINS.items()}
_SLAB_Z_CACHE = {}


def _slab_depth_km(chain, lon, lat):
    """観測点直下のスラブ上面の深さ[km]（海溝より海側は0）"""
    dl = -_signed_dist_km(_SLAB_EXT[chain], lon, lat)           # 陸側が正
    def tb(k):
        x, y = _SLAB_TAB[k]
        return np.interp(np.maximum(dl, 0.0), x, y)
    if chain == "PAC":                                           # 千島・日本海溝(pac) → 伊豆・小笠原(izu: 急傾斜) を緯度で接続
        w = np.clip((36.0 - lat) / 3.0, 0.0, 1.0)
        z = (1 - w) * tb("pac") + w * tb("izu")
    elif chain == "PSP":                                         # 南海(psp) → 琉球(ryu)
        w = np.clip((131.0 - lon) / 2.0, 0.0, 1.0)
        z = (1 - w) * tb("psp") + w * tb("ryu")
    else:
        z = tb("psp")
    return np.where(dl <= 0.0, 0.0, z)


def _station_slab_depth(chain):
    z = _SLAB_Z_CACHE.get((chain, N_ST))
    if z is None:
        z = _slab_depth_km(chain, ST_LON, ST_LAT)
        _SLAB_Z_CACHE[(chain, N_ST)] = z
    return z


def _source_chain(lat0, lon0):
    """震源が属する板（最寄りの海溝鎖）"""
    best = None
    for name, pts in _SLAB_CHAINS.items():
        d = min(abs(float(_signed_dist_km(pts[i:i + 2], np.array([lon0]), np.array([lat0]))[0])) for i in range(len(pts) - 1))
        if best is None or d < best[0]:
            best = (d, name)
    return best[1]


def deep_anomaly_dlog(depth, x_hypo, lat0, lon0, q=None):
    """深発地震の異常震域による log10(PGV) の補正量（観測点ごと）。浅い地震は0。
    震源→観測点直下のスラブ上面（スラブ内: 低減衰）→ 地殻の底（ウェッジ: 前弧側は低減衰・背弧側は高減衰）の
    2区間で、GMPEに元から含まれる非弾性減衰(-0.002/km)との差を積算する"""
    q = q or DEEP_Q
    s = float(np.clip((depth - DEEP_ANOM_D0) / (DEEP_ANOM_D1 - DEEP_ANOM_D0), 0.0, 1.0))
    if s <= 0.0:
        return 0.0
    z = _station_slab_depth(_source_chain(lat0, lon0))           # 観測点直下のスラブ上面の深さ
    dist = hubeny_km(lat0, lon0, ST_LAT, ST_LON)
    l1 = np.minimum(np.sqrt(dist ** 2 + np.maximum(depth - z, 0.0) ** 2), q["lmax"])   # スラブ内の伝播距離
    lw = np.maximum(z - q["zcrust"], 0.0)                        # ウェッジを上がる距離
    hot = 1.0 / (1.0 + np.exp(-(z - q["zvf"]) / q["wvf"]))        # スラブが深い所の上ほど高温ウェッジ
    c_w = q["c_cold"] + (q["c_hot"] - q["c_cold"]) * hot
    d = (0.002 - q["c_slab"]) * l1 + (0.002 - c_w) * lw
    return s * np.clip(d, -3.0, 3.0)


def _deff(depth_p):
    """距離減衰式の深さ項に使う有効な深さ（深発地震で過大にならないよう頭打ち気味にする）"""
    return np.minimum(depth_p, DEEP_Q["d0"]) + DEEP_Q["slope"] * np.maximum(depth_p - DEEP_Q["d0"], 0.0)


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
    if is_trench and _sea_factor(lat0, lon0, "内陸・その他")[0] < 0.99:
        is_trench = False          # 陸域・内陸の地震は、海溝に近くても海溝型にしない
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


def _t1(v):
    """計測震度の表示値: 小数第2位以下を切り捨てて第1位まで（気象庁方式。float32の誤差は1e-4で吸収）。
    四捨五入すると 6.46 が「6.5」と表示されながら階級は6強、という不整合になる。"""
    return math.floor(float(v) * 10.0 + 1e-4) / 10.0


def shindo_class(x):
    """計測震度 → (震度階級の添字0..9, 表示名)。気象庁の震度階級しきい値（第1位まで切り捨てた値で判定）"""
    x = _t1(x)
    th = [0.5, 1.5, 2.5, 3.5, 4.5, 5.0, 5.5, 6.0, 6.5]
    labels = ["0", "1", "2", "3", "4", "5弱", "5強", "6弱", "6強", "7"]
    k = 0
    for i, v in enumerate(th, start=1):
        if x >= v:
            k = i
    return k, labels[k]


def final_intensity(mag, depth, lat0, lon0, min_x=0.0, anomaly=True):
    """
    各観測点の最終的な計測震度（Scratch版 震度計算 の非予想側の式）を返す。
    戻り値: (m01, 震央距離km, depth_p, src)
    min_x: 震源距離の下限[km]。EEWの予測では、推定震源が観測点の直下にあるとき
           距離≈0で式が発散し震度7になるのを防ぐために使う（シミュレーション本体は0）。
    anomaly: 深発地震の異常震域（火山フロントの前弧側で強く・背弧側で弱く揺れる）を考慮するか。
           シミュレーション本体はTrue。EEWの予測はFalse（実際の予測式は異常震域を見込めない）。
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
        0.58 * mw + 0.0038 * _deff(depth_p) + 0.0
        - np.log10(x_hypo + 0.0028 * 10.0 ** (0.5 * mw))
        - 0.002 * x_hypo
        - 1.29
    )
    if anomaly and depth > DEEP_ANOM_D0:                  # 深発地震の異常震域
        log_pgv600 = log_pgv600 + deep_anomaly_dlog(depth, x_hypo, lat0, lon0)
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
CLEAR_CAP_S = 2400.0     # 強震モニタの表示を延ばす上限[秒]（全観測点が震度1未満に戻るまで）
# 深さ候補。深発地震（最大699km）まで、深いほど粗くなり過ぎないよう細かく刻む
EEW_DEPTHS = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0, 200.0, 250.0,
              300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 690.0]
EEW_DEPTHS_COARSE = (10.0, 60.0, 150.0, 300.0, 450.0, 600.0)   # 粗い探索（広域グリッド）で試す深さ
EEW_DEEP_KM = 150.0      # 推定震源の深さがこれ以上なら深発地震とみなし、推定震源に基づく警報・特別警報は発表しない
                         # （震度4以下の予報のみ。警報はPLUM法による予測のみ可）
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
        lp = (0.58 * mid + 0.0038 * _deff(depth_p)
              - np.log10(x + 0.0028 * 10.0 ** (0.5 * mid)) - 0.002 * x - 1.29)
        m = 2.68 + 1.72 * (lp + np.log10(arv * (4000.0 - sdep)))
        up = m > m01_i
        hi = np.where(up, mid, hi)
        lo = np.where(up, lo, mid)
    return (lo + hi) / 2.0


def _mw_finite(m01_i, d, dep_p, arv, sdep, trench):
    """観測震度からMwを逆算。順方向の震度計算（final_intensity）は海溝型の大地震で震源断層面までの
    距離を使うため、点震源の距離で逆算するとMwを過大評価する（東日本大震災でM9.9になる原因）。
    断層長 L=10^(0.5M-1.85) km（上限600km）から平均的な距離の短縮分を見込み、数回反復して整合させる。"""
    mw = _mw_from_m01(m01_i, np.sqrt(d * d + dep_p * dep_p), dep_p, arv, sdep)
    if not trench:
        return mw
    for _ in range(4):
        mg = _mag_of_mw(mw)
        L = np.where(mg >= 7.0, np.minimum(10.0 ** (0.5 * np.minimum(mg, 12.0) - 1.85), 600.0), 0.0)
        de = np.maximum(d - 0.32 * L, 0.25 * d)
        new = _mw_from_m01(m01_i, np.sqrt(de * de + dep_p * dep_p), dep_p, arv, sdep)
        mw = 0.5 * (mw + new)                              # 振動を避けるため減衰を入れる
    return mw


def _estimate_mag(avail, tp_n, lat_e, lon_e, dep_e, t0_e, t_data, rt, tau, rng, noise=None):
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
    trench = classify_source(8.0, dep_e, lat_e, lon_e)["type"] == "海溝型"   # 海溝型の大地震なら断層面距離で評価
    mw_p = None
    if ok.any():
        sel = np.where(ok)[0]
        sel = sel[np.argsort(d_all[sel])[:10]]            # 震源に近い観測点ほど信号が強い（上位10点）
        idx, ff, d = avail[sel], f[sel], d_all[sel]
        obs = rt[ff, idx].astype(np.float64) + (noise[idx] if noise is not None else rng.normal(0.0, 0.25, len(idx)))   # 振幅の観測ノイズ（観測点ごとに固定）
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
        mws = _mw_finite(m01_i, d, dep_p, ST_ARV[idx], ST_DEP[idx], trench)
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
        obs = rt[onset[son], idx].astype(np.float64) + (noise[idx] if noise is not None else rng.normal(0.0, 0.25, len(idx)))
        bai = ((1.0 + ST_ARV[idx]) / 2.2) / 2.0
        fthr = -3.3 + (1.0 + bai * 3.0) / 4.0
        low = obs <= fthr + 0.3                            # 検出下限以下（上限のみ）
        hi = obs >= 4.5                                    # s0が上限に張り付き（下限のみ）
        k2 = np.clip(d, 0.5, 300.0)
        m01_s = np.maximum(obs, fthr + 0.3) + 1.5 + k2 / 150.0 + dterm / 52.0
        x = np.sqrt(d * d + dep_p * dep_p)
        mws_s = _mw_finite(m01_s, d, dep_p, ST_ARV[idx], ST_DEP[idx], trench)
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
    if mag > 9.3:                                          # 実在の地震の規模を超える推定は緩やかに頭打ち
        mag = 9.3 + 0.4 * math.tanh((mag - 9.3) / 0.4)
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
    amp_noise = rng.normal(0.0, 0.25, N_ST)           # 振幅の観測ノイズ（観測点ごとに固定。続報ごとに引き直すとMが毎回ばらつく）
    mag_s = None                                      # 推定Mの平滑値（報ごとの±0.3〜0.5のばらつきを抑える）
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
                                T - t_first, rng, amp_noise)
            # 規模がまだ推定できない間（P波検知直後で振幅が得られない）は、仮の固定値を
            # 発表せず、直前の報の値を引き継ぐ。最初の報は規模が推定できてから発表する。
            mag_known = (est is not None) or (last is not None)
            if est is None:
                mag_e, capped = (last["mag"] if last else 3.0), False
            else:
                mag_e, capped = est
                # 規模が成長中（頭打ち）や大きな変化はそのまま採用し、小さな変動は平滑化する
                if mag_s is None or capped or abs(mag_e - mag_s) >= 1.0:
                    mag_s = mag_e
                else:
                    mag_s += 0.4 * (mag_e - mag_s)
                mag_e = mag_s

            # 推定震源が観測点直下でも式が発散しないよう、震源距離に下限(10km)を設ける
            m01_p, _, _, _ = final_intensity(mag_e, dep_e, lat_e, lon_e, min_x=10.0, anomaly=False)
            # 深発地震と推定された場合: 推定震源・規模からの予測は警報の根拠にできない（震度4止まり）。
            # 警報・特別警報の根拠はPLUM法（観測された揺れの周辺30kmへの予測）のみとする。
            deep_est = bool(dep_e >= EEW_DEEP_KM)
            if deep_est:
                m01_p = np.minimum(m01_p, 4.4)
            plum_mask = np.zeros(N_ST, dtype=bool)
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
                    plum_mask = upm
                    plum_used = True
            # 警報・特別警報の判定と警報対象地域に使う予測震度（深発地震ではPLUM法の予測のみ）
            m01_w = np.where(plum_mask, m01_p, -9.0) if deep_est else m01_p
            # 予想震度の塗りつぶし用: 観測点ごとのばらつき(地盤増幅の差)で地図がまだらにならないよう、
            # 気象庁のEEWと同じく「区域ごとの最大予想震度」を区域内の全観測点の値とする
            amx_all = np.full(max_a + 1, -99.0)
            np.maximum.at(amx_all, ST_AREA[ST_LAND], m01_p[ST_LAND])
            pq_vals = np.where(ST_SEA, -9.0, amx_all[ST_AREA])    # 海域は予想震度の塗りつぶしの対象外
            vmax_e = float(m01_p[ST_LAND].max())
            cls_e, lb_max = shindo_class(vmax_e)
            cls_w = shindo_class(float(m01_w[ST_LAND].max()))[0]
            if cls_w >= 7:            # 推定最大震度6弱以上 → 特別警報
                lv = 3
            elif cls_w >= 5:          # 推定最大震度5弱以上 → 警報
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
                np.maximum.at(amx, ST_AREA[ST_LAND], m01_w[ST_LAND])
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
                    # 推定に変化がなくても、実際のEEWのように一定間隔で続報（精度の更新）を出す
                    per = 2.0 if T < T0 + 40 else (4.0 if T < T0 + 120 else 8.0)
                    changed = ((lv != last["lv"]) or area_added or (moved and T - last["t"] >= (1 if lv >= 2 else 2))
                               or (T - last["t"] >= per))
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
                            "maxCls": cls_e, "plum": bool(plum_used), "deepEst": deep_est, "lv": lv, "areas": areas, "final": after_peak,
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

    # シミュレーション時間：最遠観測点にS波が届いてから90秒（上限540秒）を最低限とし、
    # さらに「全観測点が震度1未満（計測震度0.5未満）に戻る」までは強震モニタを表示し続ける
    finite = tatsu[np.isfinite(tatsu)]
    t_far = float(finite.max()) if len(finite) else t_valid
    t_end0 = float(np.clip(t_far + 90.0, 60.0, min(540.0, t_valid)))
    nf_cap = int(CLEAR_CAP_S / dt) + 1
    tf = np.arange(nf_cap) * dt
    p_curve = np.interp(tf, tg, p_curve_f)
    s_curve = np.interp(tf, tg, s_curve_f)
    # 走時表の範囲(600秒)を超えた後は、最後の傾きで波面を延長する（頭打ちのままだと、波が通過した後の
    # 観測点の震度が最終値に達せず、減衰が始まらないため）
    late = tf > tg[-1]
    if late.any():
        for cur_f, arr in ((p_curve_f, p_curve), (s_curve_f, s_curve)):
            sl = (cur_f[-1] - cur_f[-201]) / (tg[-1] - tg[-201])
            arr[late] = cur_f[-1] + sl * (tf[late] - tg[-1])
    rt = realtime_frames(m01, dist, dep_p, tf, p_curve, s_curve, tatsu)
    reach = np.isfinite(tatsu)                            # S波が届かない観測点は判定から除く
    act = np.where((rt[:, reach] >= 0.5).any(axis=1))[0] if reach.any() else []
    t_clear = float(tf[act[-1]] + dt) if len(act) else 0.0
    t_end = float(max(t_end0, min(t_clear + 5.0, CLEAR_CAP_S - dt)))
    nf = int(t_end / dt) + 1
    tf, p_curve, s_curve, rt = tf[:nf], p_curve[:nf], s_curve[:nf], rt[:nf]
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
                      "v": _t1(v), "cls": cls, "label": label})
        if len(areas) >= 20:
            break
    return areas


def _clean(a, nd=1):
    return [None if not math.isfinite(x) else round(float(x), nd) for x in a]


# =====================================================================
# 2d. 震度速報・地震情報（発生後に随時更新されながら発表される情報）
#   ・ほぼ同時に起きた地震は一つの地震としてまとめて発表し、時間が離れた地震は別々に発表する。
#       発生時刻の間隔が INFO_GROUP_GAP 秒以内 かつ 震央間の距離が INFO_GROUP_DIST km 以内なら同じ地震
#       （A→B→Cのように連鎖して判定。グループの最後の地震からの間隔で測る）
#   ・発表する情報（気象庁の運用を簡略化。時刻は地震発生（グループ内の各地震の発生）からの経過）
#       震度速報               約1分半後（震度3以上を観測した区域。直後の数分間に区域・階級が増えれば更新）
#       震源に関する情報       約2分半後（Mは速報値。震度3以上のときのみ）
#       震源・震度に関する情報 約5分後（Mを確定。震度3以上の区域。震度2以下は震度1以上の区域）
#       各地の震度に関する情報 約6分後と10分後（震度1以上の区域・観測点数）
#   ・観測点名のデータは無いため、「各地の震度」は区域単位（観測点数つき）で表示する
# =====================================================================
INFO_GROUP_GAP = 60.0
INFO_GROUP_DIST = 200.0


def group_quakes(quakes):
    """quakes のインデックスを、同じ地震として扱うグループに分ける（各グループは発生順）"""
    order = sorted(range(len(quakes)), key=lambda i: quakes[i]["delay"])
    groups = []
    for i in order:
        q = quakes[i]
        best = None
        for g in groups:
            if q["delay"] - max(quakes[j]["delay"] for j in g) > INFO_GROUP_GAP:
                continue
            d = min(float(_planar_km(q["lat"], q["lon"], quakes[j]["lat"], quakes[j]["lon"])) for j in g)
            if d <= INFO_GROUP_DIST and (best is None or d < best[0]):
                best = (d, g)
        if best:
            best[1].append(i)
        else:
            groups.append([i])
    return groups


def _epi_name(lat, lon):
    """震央地名。データがあれば気象庁の震央地名、無ければ最寄りの陸上観測点の区域から簡易的に作る"""
    geo = epicenter_region(lat, lon)
    if geo:
        return geo
    de = np.where(ST_LAND, _planar_km(lat, lon, ST_LAT, ST_LON), 1e9)
    j = int(np.argmin(de))
    if de[j] <= 25.0:
        return AREA_NAMES.get(int(ST_AREA[j]), "")
    if de[j] <= 300.0:
        return AREA_GEO.get(int(ST_AREA[j]), ("", ""))[0] + "沖"
    return "日本の遠方"


def build_info(quakes, groups, grp_rt, tsus, dt):
    """グループごとの情報発表タイムライン（全地震を通した時間軸[秒]）。
    grp_rt[k]: グループkの全観測点のリアルタイム震度フレーム（グループ内の地震の最大値）"""
    rng = np.random.default_rng(int(abs(sum(q["lat"] * 1000 + q["lon"] * 100 + q["mag"] * 10 for q in quakes))) % (2 ** 31))
    la = ST_AREA[ST_LAND]
    nA = int(ST_AREA.max()) + 1

    def areas_ge(row, minc, cnt=False):
        am = np.full(nA, -99.0)
        np.maximum.at(am, la, row)
        if cnt:
            cc = np.zeros(nA, dtype=np.int64)
            np.add.at(cc, la, (row >= 0.5).astype(np.int64))
        res = []
        for a in np.argsort(-am):
            v = float(am[a])
            c_ = shindo_class(v)[0]
            if v < 0.5 or c_ < minc:
                break
            # [区域名, 階級, 計測震度(小数第2位切り捨て), 観測点数(無ければNone), 都道府県]
            e = [AREA_NAMES.get(int(a), "区域%d" % int(a)), c_, _t1(v), int(cc[a]) if cnt else None,
                 AREA_GEO.get(int(a), ("", ""))[0]]
            res.append(e)
            if len(res) >= 150:
                break
        return res

    out, end, end0 = [], 0.0, 0.0
    for k, g in enumerate(groups):
        cm = np.maximum.accumulate(grp_rt[k][:, ST_LAND].astype(np.float32), axis=0)   # 各観測点の「その時点までの最大」
        nf = cm.shape[0]

        def row_at(t):
            return cm[min(max(int(round(t / dt)), 0), nf - 1)]

        peak = float(cm[-1].max())
        if peak < 0.5:
            continue                                      # 震度1未満: 情報は発表されない
        peak_cls = shindo_class(peak)[0]
        srcs = sorted(g, key=lambda i: quakes[i]["delay"])
        t0g, tlast = float(quakes[srcs[0]]["delay"]), float(quakes[srcs[-1]]["delay"])
        members = []
        for i in srcs:
            q = quakes[i]
            m1 = round(min(max(q["mag"] + float(rng.normal(0, 0.3)), q["mag"] - 0.5), q["mag"] + 0.5), 1)   # 速報値
            members.append({"epi": _epi_name(q["lat"], q["lon"]), "lat": round(q["lat"], 2), "lon": round(q["lon"], 2),
                            "depth": int(round(q["depth"])), "depth1": int(round(q["depth"] / 10.0)) * 10,
                            "mag": round(q["mag"], 1), "mag1": m1, "delay": q["delay"]})
            members[-1].update(giant=bool(q["mag"] >= 8.0),                         # M8以上: 当初は「M8以上の巨大地震と推定」、数十分後に更新
                               mag2=round(q["mag"] + float(rng.normal(0, 0.2)), 1),
                               tg1=q["delay"] + 1500.0 + float(rng.integers(-120, 121)),
                               tg2=q["delay"] + 3300.0 + float(rng.integers(-180, 181)))
        # 津波に関する付加文は、その情報の発表時点の津波警報等の状況で決める（発表の約2分先まで見込む）
        trep = sorted((r["t"], max([a["lv"] for a in r["areas"]] or [0])) for i in g for r in tsus[i]["reports"])
        later_any = any(l_ > 0 for _, l_ in trep)

        def tsu_at(t):
            top = max([l_ for tt, l_ in trep if tt <= t + 120.0] or [0])
            if top == 0:
                return 3 if later_any else 0              # 3: 津波の影響は調査中
            return 1 if top == 1 else 2

        def mem_at(t, upto):
            res = []
            for m in members[:upto + 1]:
                est = t < m["delay"] + 300                # 発生5分以内は速報値（Mは概算、深さは10km刻み）
                gst = False
                if m["giant"]:
                    if t < m["tg1"]:
                        mg_, gst = m["mag1"], True        # M8以上の巨大地震と推定（Mは決め切れない）
                    elif t < m["tg2"]:
                        mg_ = m["mag2"]                   # 精査後の暫定値
                    else:
                        mg_ = m["mag"]                    # 確定値
                else:
                    mg_ = m["mag1"] if est else m["mag"]
                res.append({"epi": m["epi"], "lat": m["lat"], "lon": m["lon"], "delay": m["delay"],
                            "depth": m["depth1"] if est else m["depth"], "mag": mg_,
                            "ge": bool(est and mg_ >= 8.0 and not m["giant"]), "gst": gst})
            return res

        def mx_of(t):
            c_, l_ = shindo_class(float(row_at(t).max()))
            return [c_, l_]

        rels = []

        def add(t, kind, upd, **kw):
            rels.append(dict(t=round(float(t), 1), k=kind, upd=bool(upd), mx=mx_of(t), **kw))

        def jit(a=-8, b=9):
            return float(rng.integers(a, b))

        if peak_cls >= 3:
            first3 = float(np.argmax(cm.max(axis=1) >= 2.5)) * dt     # 最初に震度3以上を観測した時刻
            t_sok = max(t0g + 90.0, first3 + 30.0) + jit(-5, 6)
            add(t_sok, "sokuho", False, areas=areas_ge(row_at(t_sok), 3))
            seen = {(a[0], a[1]) for a in rels[-1]["areas"]}
            for du in (60.0, 120.0, 180.0):               # 直後に区域・階級が増えたら震度速報を更新
                a_ = areas_ge(row_at(t_sok + du), 3)
                s_ = {(x[0], x[1]) for x in a_}
                if s_ != seen:
                    add(t_sok + du, "sokuho", True, areas=a_)
                    seen = s_
            tg = t_sok
            for j, m in enumerate(members):
                tg = max(m["delay"] + 150.0 + jit(-10, 11), (t_sok + 30.0) if j == 0 else (tg + 20.0))
                add(tg, "gensen", j > 0, mem=mem_at(tg, j), tsu=tsu_at(tg))
        else:
            tg = t0g + 240.0
        minc = 3 if peak_cls >= 3 else 1
        ts_ = tg
        for j, m in enumerate(members):
            ts_ = max(m["delay"] + 300.0 + jit(-10, 11), ts_ + 60.0)
            add(ts_, "gs", j > 0, mem=mem_at(ts_, j), areas=areas_ge(row_at(ts_), minc), tsu=tsu_at(ts_), minc=minc)
        tk = max(tlast + 360.0, ts_ + 30.0)
        for n_, tt in enumerate((tk, tk + 240.0)):        # 各地の震度に関する情報（初報と更新）
            add(tt, "kakuchi", n_ > 0, mem=mem_at(tt, len(members) - 1), areas=areas_ge(row_at(tt), 1, True))
        gmem = [m for m in members if m["giant"]]
        if gmem:                                          # M8以上: 数十分後に震源要素を更新（暫定値→確定値の2回）
            for m in gmem:
                m["tg1"] = max(m["tg1"], tk + 600.0)
                m["tg2"] = max(m["tg2"], m["tg1"] + 1200.0)
            for stage in (1, 2):
                tgs = max(m["tg1" if stage == 1 else "tg2"] for m in gmem) + 1.0
                old = [((-1 if stage == 1 else m["mag2"]) if m["giant"] else m["mag"]) for m in members]
                add(tgs, "genkou", False, mem=mem_at(tgs, len(members) - 1), old=old, tsu=tsu_at(tgs))
        elif peak_cls >= 5:                               # 震度5弱以上: 約25〜30分後に震源要素を精査して更新
            tgk = max(tlast + 1500.0 + jit(-60, 61), tk + 600.0)
            add(tgk, "genkou", False, mem=mem_at(tgk, len(members) - 1), old=[m["mag1"] for m in members],
                tsu=tsu_at(tgk))
        rels.sort(key=lambda r: r["t"])
        end = max(end, rels[-1]["t"])
        end0 = max(end0, max([r_["t"] for r_ in rels if r_["k"] != "genkou"] or [0.0]))
        out.append({"t0": t0g, "members": mem_at(1e9, len(members) - 1), "releases": rels})
    return {"groups": out, "end": round(end + 20.0, 1), "end0": round(end0 + 20.0, 1)}


def _t_p_first(c, q):
    """陸上観測点に最初にP波が届く時刻[秒]。計算範囲外（遠すぎる）なら距離から概算"""
    tpl = c["tp"][ST_LAND]
    tpl = tpl[np.isfinite(tpl)]
    if len(tpl):
        return float(tpl.min())
    return float(hubeny_km(q["lat"], q["lon"], ST_LAT[ST_LAND], ST_LON[ST_LAND]).min()) / 8.0 + 15.0


def sim_multi(quakes, dt=1.0, gz=False):
    """複数の地震を同時に扱う。quakes: [{lat, lon, mag, depth, delay(秒, 整数)}, ...]
    各地震を独立に計算し、delay秒ずらして重ね合わせる。
    観測点ごとのリアルタイム震度・最終震度は、全地震の最大値を採用する。
    戻り値: (基本データdict, [各地震の (tp, rt, t_end)])"""
    cores = [quake_core(q["lat"], q["lon"], q["mag"], q["depth"], dt) for q in quakes]
    tsus = [_shift_tsu(tsunami_for_quake(q["lat"], q["lon"], q["mag"], q["depth"], t_p1=_t_p_first(c, q)), q["delay"])
            for q, c in zip(quakes, cores)]
    t_end = max(q["delay"] + c["t_end"] for q, c in zip(quakes, cores))
    nf = int(t_end / dt) + 1

    groups = group_quakes(quakes)                         # 震度速報・地震情報は、近い地震をまとめて1つとして発表
    gid = {i: k for k, g in enumerate(groups) for i in g}
    grp_rt = [None] * len(groups)
    rt_all, m01_all = None, None
    for i, (q, c) in enumerate(zip(quakes, cores)):
        d = int(round(q["delay"] / dt))
        rt = c["rt"]
        sh = np.empty((nf, N_ST), dtype=np.float32)
        sh[:d] = rt[0]                                    # 発生前: 揺れなし（観測点ごとの下限値）
        sh[d:d + len(rt)] = rt
        sh[d + len(rt):] = rt[-1]                         # 計算範囲の終了後は最終フレームを保持
        rt_all = sh if rt_all is None else np.fmax(rt_all, sh)
        grp_rt[gid[i]] = sh if grp_rt[gid[i]] is None else np.fmax(grp_rt[gid[i]], sh)
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
        "max": _t1(vmax),
        "maxCls": cls,
        "maxLabel": label,
        "areas": _area_list(m01_all),
        "tEnd": t_end,
        "tsuAreas": [[a[0], a[1], a[2], a[3]] for a in TSU_AREAS],
        "tsuEnd": max([t["end"] for t in tsus] + [0.0]),
        "info": build_info(quakes, groups, grp_rt, tsus, dt),
        "ett": _eew_tt_payload(nf),
        "quakes": [{
            "lat": q["lat"], "lon": q["lon"], "mag": q["mag"], "depth": q["depth"],
            "delay": q["delay"],
            "P": _clean(c["p_curve"], 2),                 # その地震のローカル時間でのP波到達距離
            "S": _clean(c["s_curve"], 2),
            "tEnd": c["t_end"],
            "src": {"type": c["src"]["type"], "name": c["src"]["name"], "L": round(c["src"]["L"])},
            "tsu": ts,
        } for q, c, ts in zip(quakes, cores, tsus)],
    }
    return out, [(c["tp"], c["rt"], c["t_end"]) for c in cores]


# =====================================================================
# 2c. 津波警報・注意報・津波情報（気象庁の発表内容・発表速度に合わせた簡易再現）
#   ・津波予報区66区分（代表点・海の向き・増幅係数は概略値）ごとに予想高さ・到達予想時刻を計算
#   ・予想高さ: 阿部(1979)型の経験式 H = 10^(Mt-5.65)/Δ をベースに、海の向き・湾の増幅・深さで補正
#   ・到達時間: 沿岸ほど浅くなる海底を仮定した長波の伝播時間（約20分で130km、600km先で約1時間）
#   ・震源が海か（沿岸か）は、周囲の陸上観測点の分布（距離・方位の欠け）から簡易判定
#   ・発表の流れ（実際の運用に準拠。第1報は近地で約3分、遠地・中遠地は地震波の到達後〜到達の約3時間前まで
#     後ろにずれる。観測の取り込みは津波の到達後）:
#       約3分後  第1報（地震波から推定したM。M8以上の巨大地震は「巨大」表現で最大クラスを仮定）
#       約11〜15分後 第2報（Mw確定。巨大→数値に確定、予想高さの引き上げ）
#       約35分後 第3報（沖合・沿岸の観測を反映）
#       以降  津波観測に関する情報（第1波・最大波）、時間経過に伴う切替・解除
#   ・警報等の予想高さは、発表後の続報で下げない（巨大→数値の確定時のみ例外）
# =====================================================================
# (名称, 緯度, 経度, 海の向き[度], 増幅係数)
TSU_AREAS = [
    ("北海道太平洋沿岸東部", 42.98, 144.38, 150, 1.0), ("北海道太平洋沿岸中部", 42.15, 142.77, 180, 1.0),
    ("北海道太平洋沿岸西部", 42.32, 140.97, 160, 0.8), ("北海道日本海沿岸北部", 44.40, 141.65, 270, 0.8),
    ("北海道日本海沿岸南部", 41.87, 140.12, 270, 0.9), ("オホーツク海沿岸", 44.02, 144.27, 20, 0.8),
    ("青森県日本海沿岸", 40.65, 139.93, 270, 0.9), ("青森県太平洋沿岸", 40.53, 141.53, 100, 1.0),
    ("陸奥湾", 40.85, 140.80, 0, 0.3), ("岩手県", 39.64, 141.97, 100, 1.4),
    ("宮城県", 38.32, 141.50, 100, 1.2), ("秋田県", 39.72, 140.05, 280, 0.9),
    ("山形県", 38.92, 139.84, 290, 0.9), ("福島県", 37.05, 141.00, 100, 1.0),
    ("茨城県", 36.35, 140.60, 100, 0.9), ("千葉県九十九里・外房", 35.40, 140.45, 100, 1.0),
    ("千葉県内房", 35.15, 139.83, 240, 0.6), ("東京湾内湾", 35.37, 139.90, 270, 0.3),
    ("伊豆諸島", 33.10, 139.80, 180, 1.0), ("小笠原諸島", 27.10, 142.20, 180, 1.0),
    ("相模湾・三浦半島", 35.20, 139.50, 200, 0.8), ("神奈川県東部", 35.45, 139.65, 150, 0.3),
    ("新潟県上中下越", 37.90, 139.00, 320, 0.9), ("佐渡", 38.00, 138.40, 310, 0.9),
    ("富山県", 36.80, 137.20, 20, 0.8), ("石川県能登", 37.40, 137.20, 330, 1.2),
    ("石川県加賀", 36.40, 136.40, 300, 0.9), ("福井県", 36.00, 136.10, 300, 0.9),
    ("静岡県", 34.60, 138.20, 200, 1.1), ("愛知県外海", 34.70, 137.20, 190, 0.9),
    ("伊勢・三河湾", 34.85, 136.90, 200, 0.4), ("三重県南部", 34.00, 136.40, 150, 1.3),
    ("淡路島南部", 34.20, 134.80, 150, 0.9), ("大阪府", 34.60, 135.40, 250, 0.4),
    ("兵庫県北部", 35.60, 134.60, 340, 0.7), ("兵庫県瀬戸内海沿岸", 34.70, 135.00, 200, 0.3),
    ("和歌山県", 33.50, 135.70, 160, 1.2), ("鳥取県", 35.55, 134.20, 340, 0.8),
    ("島根県出雲・石見", 35.40, 132.70, 330, 0.8), ("隠岐", 36.20, 133.30, 340, 0.8),
    ("岡山県", 34.55, 133.95, 190, 0.3), ("広島県", 34.30, 132.50, 200, 0.3),
    ("山口県日本海沿岸", 34.40, 131.10, 330, 0.8), ("山口県瀬戸内海沿岸", 34.00, 131.50, 170, 0.4),
    ("香川県", 34.35, 134.00, 20, 0.3), ("愛媛県宇和海沿岸", 33.10, 132.50, 230, 1.0),
    ("愛媛県瀬戸内海沿岸", 33.90, 132.90, 340, 0.3), ("徳島県", 33.80, 134.60, 120, 1.2),
    ("高知県", 33.40, 133.50, 190, 1.4), ("福岡県瀬戸内海沿岸", 33.85, 130.95, 100, 0.3),
    ("福岡県日本海沿岸", 33.70, 130.40, 340, 0.6), ("有明海", 33.00, 130.40, 230, 0.2),
    ("佐賀県北部", 33.45, 129.95, 320, 0.5), ("長崎県西方", 32.75, 129.70, 270, 0.7),
    ("壱岐・対馬", 34.40, 129.30, 300, 0.7), ("熊本県天草灘沿岸", 32.40, 130.00, 270, 0.7),
    ("大分県瀬戸内海沿岸", 33.40, 131.60, 60, 0.4), ("大分県豊後水道沿岸", 32.90, 132.00, 150, 1.0),
    ("宮崎県", 31.90, 131.50, 100, 1.2), ("鹿児島県東部", 31.40, 131.00, 120, 1.0),
    ("種子島・屋久島地方", 30.50, 131.00, 110, 1.0), ("奄美群島・トカラ列島", 28.40, 129.50, 120, 1.0),
    ("鹿児島県西部", 31.30, 130.20, 250, 0.7), ("沖縄本島地方", 26.30, 127.80, 110, 1.0),
    ("大東島地方", 25.85, 131.25, 100, 1.0), ("宮古島・八重山地方", 24.40, 124.20, 150, 1.0),
]
_TSU_N = len(TSU_AREAS)
_TSU_LAT = np.array([a[1] for a in TSU_AREAS])
_TSU_LON = np.array([a[2] for a in TSU_AREAS])
_TSU_FAC = np.array([a[3] for a in TSU_AREAS], dtype=float)
_TSU_AMP = np.array([a[4] for a in TSU_AREAS])
TSU_LVN = {1: "津波予報", 2: "津波注意報", 3: "津波警報", 4: "大津波警報"}
_TSU_RANK = {"": 0, "若干の海面変動": 0, "1m": 1, "高い": 2, "3m": 2, "5m": 3, "10m": 4, "10m超": 5, "巨大": 6}
_TSU_MSG = {
    4: "大津波警報を発表しました。巨大な津波が襲来します。ただちに高台や避難ビルなど安全な場所へ避難してください。",
    3: "津波警報を発表しました。津波が襲来します。ただちに海岸から離れ、高台や避難ビルなど安全な場所へ避難してください。",
    2: "津波注意報を発表しました。海の中や海岸付近は危険です。海の中にいる人はただちに海から上がって、海岸から離れてください。",
    1: "若干の海面変動が予想されますが、被害の心配はありません。",
}


def _tsu_grade(h):
    """予想高さ[m] → (階級 0〜4, 表示区分)。大津波警報は3m超、津波警報は1m超〜3m、注意報は0.2〜1m"""
    if h > 10.0: return 4, "10m超"
    if h > 5.0: return 4, "10m"
    if h > 3.0: return 4, "5m"
    if h > 1.0: return 3, "3m"
    if h >= 0.2: return 2, "1m"
    if h >= 0.05: return 1, "若干の海面変動"
    return 0, ""


def _sea_factor(lat0, lon0, src_type):
    """震源が海域（沿岸）である度合い 0〜1。陸上観測点が25km以内に無ければ海域。
    沿岸では30km以内の観測点が存在する方位（8分割）の数が少ない（片側にしか陸が無い）ほど海寄りとみなす。"""
    kx = 111.32 * math.cos(math.radians(lat0))
    dx = (ST_LON - lon0) * kx
    dy = (ST_LAT - lat0) * 110.57
    d = np.hypot(dx, dy)
    land = np.asarray(ST_LAND, dtype=bool)
    dn = float(np.min(np.where(land, d, np.inf)))
    if dn >= 25.0:
        f = 1.0
    else:
        m = land & (d <= 30.0)
        ang = np.degrees(np.arctan2(dx[m], dy[m])) % 360.0
        c = len(set((ang // 45).astype(int).tolist()))
        f = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.7, 5: 0.35}.get(c, 0.0)
    if src_type == "海溝型":
        f = max(f, 1.0)
    return f, dn


# ---------------------------------------------------------------------
# 津波の伝播経路（実際の海路をたどる）
#   粗い格子（0.15度≒15km）の海陸マスクを観測点の分布から作り、震源から海上だけを
#   ダイクストラ法で伝播させる。波速は √(g·h)、水深hは海岸からの距離で近似（沿岸ほど浅く遅い）。
#   ・陸を回り込む経路は長くなるため、到達が遅れ、波高も下がる（回折・拡散による減衰）
#   ・太平洋側の地震では、津軽海峡・対馬海峡などを回り込んだ日本海側にだけ、遅れて小さな津波が届く
#   ・湾の奥など格子の海から孤立した予報区は、最寄りの到達可能な海域から余分な距離を足して扱う
#   メモリは格子約4万点（数百KB）程度。震源ごとの経路は小さなキャッシュに保持。
# ---------------------------------------------------------------------
_TG_LON0, _TG_LAT0, _TG_STEP = 120.0, 20.0, 0.15
_TG_NX, _TG_NY = 241, 177                      # 経度120〜156 / 緯度20〜46.4
_TG = None
_TSU_PATH_CACHE = {}
# 山地で観測点が疎な所からの「漏れ」を防ぐための陸の骨格（本州・北海道・九州・四国・紀伊・能登・佐渡の脊梁）
_LAND_SPINES = [
    [(140.9, 41.3), (140.9, 40.0), (140.4, 38.8), (139.6, 37.3), (138.7, 36.3), (137.5, 35.9), (136.3, 35.4),
     (135.3, 35.1), (134.2, 35.0), (133.0, 34.8), (131.8, 34.4), (131.0, 34.1)],
    [(140.3, 41.6), (140.9, 42.4), (141.5, 43.0), (142.4, 43.7), (143.6, 43.7), (144.7, 43.7), (145.2, 43.5)],
    [(141.3, 44.6), (142.0, 44.0), (142.8, 43.0), (143.1, 42.2)],
    [(130.9, 33.6), (131.2, 32.6), (130.9, 31.5)], [(132.6, 33.5), (133.5, 33.8), (134.3, 33.9)],
    [(135.2, 35.0), (135.6, 34.4), (136.0, 34.1)], [(136.7, 37.1), (137.2, 37.4)], [(138.3, 38.1), (138.5, 38.3)],
]
# 津軽海峡（本州と北海道の間の狭い海峡）は陸の穴埋めで塞がれないよう、海として明示
_SEA_CARVE = [(140.05, 140.85, 41.27, 41.60), (140.85, 141.40, 41.55, 41.80)]


def _tg_idx(lon, lat):
    return (int(round((lat - _TG_LAT0) / _TG_STEP)), int(round((lon - _TG_LON0) / _TG_STEP)))


def _tsu_grid():
    """海陸マスク・波速・格子寸法を一度だけ作る"""
    global _TG
    if _TG is not None:
        return _TG
    nx, ny, st = _TG_NX, _TG_NY, _TG_STEP
    land = np.zeros((ny, nx), dtype=bool)
    ix = np.rint((ST_LON - _TG_LON0) / st).astype(np.int64)
    iy = np.rint((ST_LAT - _TG_LAT0) / st).astype(np.int64)
    ok = np.asarray(ST_LAND, dtype=bool) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    land[iy[ok], ix[ok]] = True
    for sp in _LAND_SPINES:                              # 脊梁線をラスタライズ
        for (x1, y1), (x2, y2) in zip(sp[:-1], sp[1:]):
            n = int(max(abs(x2 - x1), abs(y2 - y1)) / (st * 0.5)) + 1
            for k in range(n + 1):
                j, i = _tg_idx(x1 + (x2 - x1) * k / n, y1 + (y2 - y1) * k / n)
                if 0 <= j < ny and 0 <= i < nx:
                    land[j, i] = True

    def dil(m):
        o = m.copy()
        o[1:, :] |= m[:-1, :]; o[:-1, :] |= m[1:, :]; o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
        o[1:, 1:] |= m[:-1, :-1]; o[:-1, :-1] |= m[1:, 1:]; o[1:, :-1] |= m[:-1, 1:]; o[:-1, 1:] |= m[1:, :-1]
        return o

    def ero(m):
        return ~dil(~m)

    land = ero(dil(land))                                # 閉包: 観測点の隙間を埋め、海岸線は保つ
    for x0, x1, y0, y1 in _SEA_CARVE:
        j0, i0 = _tg_idx(x0, y0)
        j1, i1 = _tg_idx(x1, y1)
        land[j0:j1 + 1, i0:i1 + 1] = False
    lat_row = _TG_LAT0 + np.arange(ny) * st
    cx = st * 111.32 * np.cos(np.radians(lat_row))       # 各行の格子の東西幅[km]
    cy = st * 110.57
    # 海岸からの距離[km]（チャンファー距離変換。陸は0）
    INF = 1e9
    d = np.where(land, 0.0, INF).tolist()
    cxl = cx.tolist()
    for j in range(ny):
        row, up = d[j], (d[j - 1] if j > 0 else None)
        for i in range(nx):
            v = row[i]
            if v == 0.0:
                continue
            if i > 0 and row[i - 1] + cxl[j] < v: v = row[i - 1] + cxl[j]
            if up is not None:
                if up[i] + cy < v: v = up[i] + cy
                if i > 0 and up[i - 1] + math.hypot(cxl[j], cy) < v: v = up[i - 1] + math.hypot(cxl[j], cy)
                if i < nx - 1 and up[i + 1] + math.hypot(cxl[j], cy) < v: v = up[i + 1] + math.hypot(cxl[j], cy)
            row[i] = v
    for j in range(ny - 1, -1, -1):
        row, dn = d[j], (d[j + 1] if j < ny - 1 else None)
        for i in range(nx - 1, -1, -1):
            v = row[i]
            if v == 0.0:
                continue
            if i < nx - 1 and row[i + 1] + cxl[j] < v: v = row[i + 1] + cxl[j]
            if dn is not None:
                if dn[i] + cy < v: v = dn[i] + cy
                if i < nx - 1 and dn[i + 1] + math.hypot(cxl[j], cy) < v: v = dn[i + 1] + math.hypot(cxl[j], cy)
                if i > 0 and dn[i - 1] + math.hypot(cxl[j], cy) < v: v = dn[i - 1] + math.hypot(cxl[j], cy)
            row[i] = v
    dl = np.minimum(np.array(d), 800.0)
    depth_m = np.clip(dl * 28.0, 25.0, 5000.0)           # 海岸から離れるほど深い（20kmで約560m、180kmで5000m）
    spd = np.sqrt(9.8 * depth_m) / 1000.0                # 波速[km/s]
    _TG = {"land": land, "spd": spd, "cx": cx, "cy": cy}
    return _TG


def _tsu_trace(lat0, lon0):
    """震源から各予報区への (到達秒, 実効伝播距離km, 到来方位[度], 到達可能)。結果は小さなキャッシュに保持"""
    key = (round(lat0, 1), round(lon0, 1))
    if key in _TSU_PATH_CACHE:
        return _TSU_PATH_CACHE[key]
    import heapq
    g = _tsu_grid()
    land, spd, cxr, cy = g["land"], g["spd"], g["cx"], g["cy"]
    ny, nx = land.shape
    st = _TG_STEP
    # 震源に最も近い海の格子（格子の外なら縁に寄せ、その距離は遠地津波として0.2km/sで加算）
    j0 = min(max(int(round((lat0 - _TG_LAT0) / st)), 0), ny - 1)
    i0 = min(max(int(round((lon0 - _TG_LON0) / st)), 0), nx - 1)
    src = None
    for r in range(0, 12):
        best = None
        for j in range(max(j0 - r, 0), min(j0 + r, ny - 1) + 1):
            for i in range(max(i0 - r, 0), min(i0 + r, nx - 1) + 1):
                if max(abs(j - j0), abs(i - i0)) != r or land[j, i]:
                    continue
                dd = (j - j0) ** 2 + (i - i0) ** 2
                if best is None or dd < best[0]:
                    best = (dd, j, i)
        if best:
            src = best[1:]
            break
    if src is None:
        src = (j0, i0)
    sj, si = src
    clat, clon = _TG_LAT0 + sj * st, _TG_LON0 + si * st
    ex = float(hubeny_km(lat0, lon0, clat, clon))
    T = np.full(ny * nx, np.inf)
    D = np.zeros(ny * nx)
    P = np.full(ny * nx, -1, dtype=np.int64)
    flat_land = land.ravel().tolist()
    flat_spd = spd.ravel().tolist()
    Tl, Dl, Pl = T.tolist(), D.tolist(), P.tolist()
    s0 = sj * nx + si
    Tl[s0] = ex / 0.2
    Dl[s0] = ex
    heap = [(Tl[s0], s0)]
    cxl = cxr.tolist()
    nb = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    while heap:
        t, u = heapq.heappop(heap)
        if t > Tl[u]:
            continue
        j, i = divmod(u, nx)
        for dj, di in nb:
            jj, ii = j + dj, i + di
            if jj < 0 or jj >= ny or ii < 0 or ii >= nx:
                continue
            v = jj * nx + ii
            if flat_land[v]:
                continue
            ln = math.hypot(di * cxl[j], dj * cy)
            nt = t + ln / (0.5 * (flat_spd[u] + flat_spd[v]))
            if nt < Tl[v]:
                Tl[v] = nt
                Dl[v] = Dl[u] + ln
                Pl[v] = u
                heapq.heappush(heap, (nt, v))
    T, D, P = np.array(Tl), np.array(Dl), np.array(Pl)
    n = _TSU_N
    tm = np.full(n, np.inf)
    de = np.full(n, np.inf)
    fb = np.zeros(n)
    bay = _TSU_AMP <= 0.45                                # 湾・内海の予報区は、湾口の外まで広く探す
    for a in range(n):
        R = 1.0 if bay[a] else 0.45
        rc = int(math.ceil(R / st))
        aj, ai = _tg_idx(_TSU_LON[a], _TSU_LAT[a])
        best = None
        for j in range(max(aj - rc, 0), min(aj + rc, ny - 1) + 1):
            for i in range(max(ai - rc, 0), min(ai + rc, nx - 1) + 1):
                v = j * nx + i
                if flat_land[v] or not math.isfinite(Tl[v]):
                    continue
                dd = math.hypot((j - aj) * cy, (i - ai) * cxl[min(max(aj, 0), ny - 1)])
                if dd > R * 111.0:
                    continue
                sc = Tl[v] + dd / 0.05
                if best is None or sc < best[0]:
                    best = (sc, v, dd)
        if best is None:
            continue
        _, v, dd = best
        tm[a] = Tl[v] + dd / 0.05
        de[a] = Dl[v] + dd
        u = v                                             # 到来方位: 経路を数格子さかのぼった向き
        for _ in range(3):
            if Pl[u] < 0:
                break
            u = Pl[u]
        j1, i1 = divmod(u, nx)
        j2, i2 = divmod(v, nx)
        vx = (i2 - i1) * cxl[j2]
        vy = (j2 - j1) * cy
        fb[a] = (math.degrees(math.atan2(vx, vy)) + 180.0) % 360.0 if (vx or vy) else _TSU_FAC[a]
    out = (tm, de, fb)
    if len(_TSU_PATH_CACHE) >= 8:
        _TSU_PATH_CACHE.pop(next(iter(_TSU_PATH_CACHE)))
    _TSU_PATH_CACHE[key] = out
    return out


def _tsu_model(mt, lat0, lon0, depth, sea_f):
    """津波予報区ごとの (予想高さ[m], 到達予想までの秒数, 実効伝播距離km)。
    伝播は海路に沿って計算し、陸を大きく回り込む経路ほど距離（＝減衰）を加算する。"""
    dfac = 1.0 if depth <= 30.0 else max(0.0, 1.0 - (depth - 30.0) / 70.0)
    tm, path, fb = _tsu_trace(lat0, lon0)
    kx = 111.32 * math.cos(math.radians(lat0))
    straight = np.hypot((_TSU_LON - lon0) * kx, (_TSU_LAT - lat0) * 110.57)
    reach = np.isfinite(path)
    p = np.where(reach, path, 1e6)
    d_eff = np.maximum(p, 30.0) + 4.0 * np.maximum(p - 1.15 * straight, 0.0)    # 迂回による減衰
    ff = np.clip(0.65 + 0.35 * np.cos(np.radians(fb - _TSU_FAC)), 0.3, 1.0)     # 到来方向に海が開いているほど大きい
    h = 10.0 ** (mt - 5.65) / d_eff * sea_f * dfac * ff * _TSU_AMP
    h = np.where(reach, np.minimum(h, 40.0), 0.0)
    t = np.where(reach, tm, 1e7)
    return h, t, d_eff


TSU_NEAR_P_S = 90.0      # 日本の観測網（陸上）にP波が届くまでがこれ以内なら近地地震（第1報は約3分）


def _tsu_first_time(t_p1, t_arr, rng):
    """津波情報の第1報の発表時刻[秒]（地震発生から）。固定の3分ではなく、震源の遠さと津波の到達で変える。
      近地: 約3分。
      中遠地: 地震波が日本の観測網に届いてから約10分の解析後（揺れの到達より前には出さない）。
      遠地（地震波の到達が10分超）: 約30分以降（遠地地震に関する情報）。
      津波の到達までさらに時間がある場合は、到達の約3時間前まで待って発表する。"""
    j = int(rng.integers(-15, 16))
    if t_p1 <= TSU_NEAR_P_S:
        return 180.0 + j
    t = max(180.0 + j, t_p1 + 600.0 + 2 * j)
    if t_p1 > 600.0:
        t = max(t, 1800.0 + 20 * j)
    t = max(t, t_arr - 10800.0 + 30 * j)
    return float(min(t, max(t_arr - 300.0, 180.0 + j)))


def tsunami_for_quake(lat0, lon0, mag, depth, t_p1=0.0):
    """1つの地震の津波警報等・津波観測の時系列（その地震の発生を0秒とするローカル時間）
    t_p1: 陸上観測点に最初にP波が届く時刻[秒]（近地か遠地かの判定と第1報の発表時刻に使う）"""
    src = classify_source(mag, depth, lat0, lon0)
    sea_f, _ = _sea_factor(lat0, lon0, src["type"])
    rng = np.random.default_rng(int(abs(lat0 * 1000 + lon0 * 100 + mag * 10 + depth)) % (2 ** 31))
    res = {"reports": [], "obs": [], "end": 0.0}
    n = _TSU_N
    h_true, tm, deff = _tsu_model(mag, lat0, lon0, depth, sea_f)
    if float(h_true.max()) < 0.05:
        if mag >= 6.0:                                            # 震源に関する情報の付加文（津波の心配なし）
            t_none = 150.0 if t_p1 <= TSU_NEAR_P_S else t_p1 + 480.0   # 遠地は揺れ（地震波）の到達後
            res["reports"].append({"t": t_none, "no": 1, "kind": "none", "mag": round(mag, 1), "giant": False,
                                   "text": "この地震による津波の心配はありません。", "areas": []})
            res["end"] = max(600.0, t_none + 450.0)
        return res

    near = t_p1 <= TSU_NEAR_P_S
    giant = mag >= 8.0 and near                                   # 近地のM8以上は「巨大」表現（Mを決め切れない）。遠地は数値で発表
    m1 = (min(mag + 0.5, 9.2) if giant
          else min(mag, 7.9) - 0.15 * max(0.0, mag - 7.0) + float(rng.normal(0, 0.15)))
    m2 = mag + float(rng.normal(0, 0.05))
    eps = np.exp(rng.normal(0, 0.25, n)) * math.exp(float(rng.normal(0, 0.15)))   # 実際の津波の予想とのずれ
    # 第1報は震源・規模のデータベース検索による予測: 安全側に余裕を見るため大きめで、予報区ごとにばらつく
    H1 = _tsu_model(m1, lat0, lon0, depth, sea_f)[0] * 1.2 * np.exp(rng.normal(0, 0.3, n))
    H2 = _tsu_model(m2, lat0, lon0, depth, sea_f)[0]
    safe = 1.35 if mag >= 8.0 else 1.0                            # M8以上は警報等の予想高さを実際より高めに見積もる（安全側）
    Ht = H2 * eps                                                 # 実際に来襲する津波の高さ
    t_true = tm * (1.0 + rng.normal(0, 0.06, n))
    sig = tm[h_true >= 0.2]
    if not len(sig):
        sig = tm[h_true >= 0.05]
    t_arr = float(sig.min())                                      # 最初の津波到達（予想）
    t1 = _tsu_first_time(t_p1, t_arr, rng)                        # 第1報の発表時刻（近地約3分・遠地は後ろへ）
    stages = [(t1, H1, m1, giant, 0.0),
              (t1 + (480.0 if mag >= 7.0 else 720.0) + int(rng.integers(0, 120)),
               H2 * safe * np.exp(rng.normal(0, 0.1, n)), m2, False, 0.0)]
    lH2, lHt = np.log(np.maximum(H2 * safe, 1e-6)), np.log(np.maximum(Ht, 1e-6))
    for tk, w in ((1500.0, 0.45), (2400.0, 0.65), (3600.0, 0.8), (5400.0, 0.9)):
        # 沖合・沿岸の観測を取り込むほど実際の津波に近づく: 予想の拡大（引き上げ・追加）も縮小（切替）も起こる
        Hk = np.exp((1.0 - w) * lH2 + w * lHt + rng.normal(0, 0.08, n))
        # 観測の取り込みは、第1報からの経過か、津波が実際に到達してからの経過の遅い方
        tk2 = max(t1 + (tk - 180.0), t_arr + (tk - 1200.0)) + int(rng.integers(-120, 121))
        stages.append((max(tk2, stages[-1][0] + 120.0), Hk, m2, False, w))

    def build(H, gflag):
        d = {}
        for i in range(n):
            lv, lb = _tsu_grade(float(H[i]))
            if lv:
                if gflag:                                         # M8以上の第1報: 大津波警報=巨大、津波警報=高い、津波注意報=表記なし
                    lb = {4: "巨大", 3: "高い", 2: ""}.get(lv, lb)
                d[i] = [lv, lb]
        if any(v[0] >= 2 for v in d.values()):                   # 警報・注意報があれば予報(若干の変動)は省く
            d = {i: v for i, v in d.items() if v[0] >= 2}
        return d

    cur, no, ever = {}, 0, set()
    for si, (ts_, H, mm, gflag, w) in enumerate(stages):
        nd = build(H, gflag)
        ups, dns = False, {}
        if si == 1 and giant:
            new = nd                                              # 巨大 → 数値に確定（下がることもある）
            ups = True
        else:
            new = {i: list(v) for i, v in cur.items()}
            for i, v in nd.items():
                if (i not in new or v[0] > new[i][0]
                        or (v[0] == new[i][0] and _TSU_RANK[v[1]] > _TSU_RANK[new[i][1]])):
                    new[i] = list(v)
                    ups = True
            if si >= 2:                                           # 観測が予想を下回る予報区は1段階ずつ切り替える
                for i in list(new):
                    lv = new[i][0]
                    nl = nd[i][0] if i in nd else 0
                    if lv >= 3 and nl < lv and t_true[i] <= ts_ + 600.0:
                        new[i] = [lv - 1, {3: "3m", 2: "1m"}[lv - 1]]
                        dns[i] = (lv, lv - 1)
            if any(v[0] >= 2 for v in new.values()):
                new = {i: v for i, v in new.items() if v[0] >= 2}
        if not new or new == cur:
            continue
        no += 1
        top = max(v[0] for v in new.values())
        if no == 1:
            kind, text = "issue", _TSU_MSG[top]
        elif dns and not ups:
            kind = "switch"
            text = "".join(TSU_LVN[o] + "を" + TSU_LVN[nl] + "に切り替えます。"
                           for o, nl in sorted(set(dns.values()), reverse=True))
        else:
            kind, text = "update", _TSU_MSG[top]
        res["reports"].append({
            "t": float(ts_), "no": no, "kind": kind,
            "mag": round(mm, 1), "giant": bool(gflag), "text": text,
            "areas": [{"i": i, "lv": v[0], "hl": v[1], "ts": round(float(tm[i]), 1),
                       "imm": bool(tm[i] <= ts_ + 60.0)} for i, v in sorted(new.items())]})
        ever |= set(new)
        cur = new
    if not cur:
        return res

    # 時間経過に伴う切替・解除（最大波の後、階級ごとに数時間かけて下げる。30分刻みでまとめて発表）
    T = 720.0
    ev = []
    for i, v in cur.items():
        t, l = float(t_true[i]) + 3 * T + 300.0, v[0]
        scale = 2.0 if mag >= 8.0 else 1.0                       # 巨大地震ほど長く続く
        jt = float(np.exp(rng.normal(0, 0.15)))
        while l > 0:
            t += {1: 7200.0, 2: 21600.0, 3: 21600.0, 4: 28800.0}[l] * scale * jt   # 予報2時間・注意報6時間・警報6時間・大津波警報8時間（M8以上は2倍）
            nl = {4: 3, 3: 2, 2: 0, 1: 0}[l]
            ev.append((t, i, nl))
            l = nl
    by = {}
    for t, i, nl in sorted(ev):
        by.setdefault(int(t // 1800) * 1800 + 1800, []).append((i, nl))
    for tb in sorted(by):
        old = {i: v[0] for i, v in cur.items()}
        phr = set()
        for i, nl in by[tb]:
            phr.add((old[i], nl))
            if nl == 0:
                cur.pop(i, None)
            else:
                cur[i] = [nl, {3: "3m", 2: "1m"}[nl]]
        no += 1
        txt = "".join(TSU_LVN[o] + ("を解除します。" if nl == 0 else "を" + TSU_LVN[nl] + "に切り替えます。")
                      for o, nl in sorted(phr, reverse=True))
        res["reports"].append({
            "t": float(tb), "no": no, "kind": "switch" if cur else "cancel",
            "mag": round(m2, 1), "giant": False, "text": txt,
            "areas": [{"i": i, "lv": v[0], "hl": v[1], "ts": round(float(tm[i]), 1), "imm": False}
                      for i, v in sorted(cur.items())]})

    # 津波観測に関する情報（第1波・最大波）。発表された予報区ごとに、到達後に波が繰り返し押し寄せる。
    # 波は押し・引きが交互に来る。第1波は小さめで、最大波は第2波以降（震源から遠い海岸ほど後ろの波）。
    # 波の間隔は地震の規模で決まる（M7級で十数分、M9級で30分前後）。巨大地震では数時間にわたり観測が続く。
    # m の要素: [時刻, 累積最大波高, 波の番号, 押し(+1)/引き(-1), その波の波高]
    per0 = min(3300.0, 900.0 * 10.0 ** (0.4 * (mag - 7.5)))      # 押しから次の押しまでの周期[秒]（M7.5以下は15分、M9で約55分）
    for i in sorted(ever):
        if Ht[i] < 0.1:
            continue
        tt, prev, m = float(t_true[i]), 0.0, []
        d0 = 1 if rng.random() < 0.7 else -1                      # 第1波が押し波か引き波か
        per = per0 * float(np.exp(rng.normal(0, 0.2))) * (1.3 if _TSU_AMP[i] <= 0.45 else 1.0)   # 湾・内海は周期が長い
        far = float(np.clip((float(deff[i]) - 200.0) / 1200.0, 0.0, 1.0))   # 0: 震源に近い 〜 1: 遠い（1400km以上）
        kmax = 3 + int(round(3 * far)) + (1 if mag >= 8.5 else 0)
        kp = int(rng.integers(2, kmax + 2))                       # 最大波の番号（近い海岸は2〜3波目、遠い海岸は最大7〜8波目）
        nw = kp + 5 + (3 if mag >= 8.0 else 0)                    # 観測する波の数（最大波のあと波高が十分に小さくなるまで）
        for k in range(1, nw + 1):
            if k <= kp:
                e = max(0.15, 1.0 - 0.3 * (kp - k))
            else:
                e = max(0.15, 1.0 - 0.17 * (k - kp))
            e *= (0.55 if k == 1 else 1.0)
            hw = round(float(Ht[i]) * e * (1.0 + float(rng.normal(0, 0.04))), 2)
            prev = max(hw, prev)
            m.append([round(tt + 60.0 + (k - 1) * per / 2.0, 1), prev, k, d0 if k % 2 == 1 else -d0, hw])
        res["obs"].append({"a": i, "m": m, "pk": m[-1][0]})
    res["end"] = max([r["t"] for r in res["reports"]] + [o["pk"] for o in res["obs"]]) + 900.0
    return res


def _shift_tsu(ts, delay):
    """津波情報の時刻を全体の時間軸へdelay秒ずらしたコピー"""
    return {
        "reports": [dict(r, t=round(r["t"] + delay, 1),
                         areas=[dict(a, ts=round(a["ts"] + delay, 1)) for a in r["areas"]]) for r in ts["reports"]],
        "obs": [{"a": o["a"], "m": [[round(w[0] + delay, 1)] + list(w[1:]) for w in o["m"]], "pk": round(o["pk"] + delay, 1)}
                for o in ts["obs"]],
        "end": round(ts["end"] + delay, 1) if ts["reports"] else 0.0,
    }


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


@app.get("/api/tsunami_geojson")
def api_tsunami_geojson():
    """津波予報区の海岸線（任意）。data/tsunami_areas.geojson があれば返す（無ければ404）"""
    p = DATA_DIR / "tsunami_areas.geojson"
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
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Quakecast</title>
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
  #runFab{position:absolute;z-index:1300;left:50%;bottom:172px;transform:translateX(-50%);padding:10px 24px;
    border-radius:999px;font-size:15px;font-weight:800;background:var(--acc);border:2px solid #ffffff55;color:#04101f;
    box-shadow:0 4px 14px #000a;cursor:pointer;white-space:nowrap}
  #runFab:hover{filter:brightness(1.1)}
  #runFab:disabled{opacity:.6;cursor:wait}
  #runFab.idle{animation:runpulse 1.6s infinite}
  @keyframes runpulse{50%{box-shadow:0 0 0 9px #3b9cff44,0 4px 14px #000a}}
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
  #simClk{position:absolute;z-index:1100;right:10px;bottom:84px;background:rgba(14,20,34,.82);border:1px solid #2c3a57;
    border-radius:8px;padding:5px 9px;color:#e6ecf8;text-align:right;font-variant-numeric:tabular-nums;backdrop-filter:blur(3px)}
  #simClk #clkT{font-size:19px;font-weight:700;letter-spacing:.5px;cursor:pointer}
  #simClk #clkD{font-size:10.5px;color:#9fb0cf}
  #simClk .clkCtl{display:none;margin-top:4px;gap:4px;align-items:center;justify-content:flex-end}
  #simClk.edit .clkCtl{display:flex}
  #simClk input{background:#0e1422;color:#e6ecf8;border:1px solid #2c3a57;border-radius:4px;font-size:11px;padding:2px 3px}
  #simClk button{background:#2c3a57;color:#e6ecf8;border:0;border-radius:4px;font-size:11px;padding:3px 6px;cursor:pointer}
  #bar{position:absolute;z-index:1000;left:50%;bottom:26px;transform:translateX(-50%);
    width:min(760px,calc(100% - 20px));background:var(--panel);border:1px solid var(--line);
    border-radius:12px;padding:10px 12px}
  #ctl{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
  #ctl #seek{order:-1;flex:0 0 100%;width:100%;margin:0 0 2px}           /* シークバーは凡例と同じ幅で1行目に */
  #ctl button{padding:6px 10px;min-width:44px;text-align:center}
  #ctl select{flex:0 0 96px;width:96px;padding:5px 6px;font-size:13px;text-align:center;text-align-last:center}
  #tlabel{flex:1;font-size:13px;min-width:64px;text-align:left;margin-left:6px;font-variant-numeric:tabular-nums}
  #legend{margin-top:8px}
  #legend canvas{width:100%;height:12px;border-radius:6px;display:block}
  #ticks{display:flex;justify-content:space-between;font-size:10.5px;color:var(--sub);margin-top:2px}
  .epi{color:#ff4d4d;font-size:26px;font-weight:900;text-shadow:0 0 4px #000,0 0 8px #000;
    line-height:26px;text-align:center}
  .foot{font-size:10.5px;color:var(--sub);line-height:1.55;margin-top:14px;border-top:1px solid var(--line);padding-top:8px}
  #rightcol{position:absolute;z-index:1000;right:4px;top:56px;width:300px;max-height:calc(100% - 200px);
    overflow:auto;display:flex;flex-direction:column;gap:8px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 10px;font-size:12px;line-height:1.5}
  .card h3{margin:0 0 5px;font-size:11.5px;color:var(--sub);font-weight:600}
  .card .sub{color:var(--sub);font-size:11px;margin-top:3px}
  #rank td.no{color:var(--sub);width:14px}
  .eewcard{border-width:2px;padding:5px 8px}
  .eewcard .eh{display:flex;align-items:center;gap:6px;min-height:24px;margin-bottom:5px}
  .eewcard .ett{flex:1;min-width:0;font-size:17px;color:#fff;font-weight:800;line-height:1.3;white-space:normal;text-shadow:0 0 4px #0008}
  .eewcard .tiles{display:grid;grid-template-columns:1.15fr 1fr .9fr;gap:5px}
  .eewcard .tile{background:#0005;border-radius:6px;padding:2px 3px 1px;text-align:center;height:48px;overflow:hidden}
  .eewcard .tile .k{font-size:9.5px;color:#cfd8ea;line-height:1.25}
  .eewcard .tile .v{font-size:22px;font-weight:800;line-height:1.25;font-variant-numeric:tabular-nums;white-space:nowrap}
  .eewcard .eewDep{font-size:19px;line-height:1.5}
  .eewcard .eewDep small{font-size:11px;margin-left:2px;font-weight:600}
  .eewcard .chipbig{display:inline-block;min-width:42px;border-radius:7px;padding:0 6px;font-size:20px;line-height:1.3;color:#fff;text-shadow:0 0 3px #000a}
  .eewcard .pos{height:16px;font-size:11.5px;line-height:16px;font-variant-numeric:tabular-nums;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
  .eewcard .warn{margin-top:5px;padding:4px 8px;border-radius:6px;background:#000b;border-left:5px solid #ffd23b;color:#fff3b0;font-size:14.5px;font-weight:800;line-height:1.4;white-space:normal}
  .eewcard .warn:empty{display:none}
  .eewcard .pos{margin-top:4px}
  .eewcard .pos:empty{display:none}
  .eewcard .plumTag{display:inline-block;margin:0 6px;padding:0 7px;border-radius:6px;background:#ffd23b;color:#111;font-size:.95em;font-weight:900;letter-spacing:.06em;text-shadow:none;box-shadow:0 0 6px #ffd23b99;vertical-align:baseline}
  .eewcard details.histbox{margin-top:2px}
  .eewcard details.histbox>summary{cursor:pointer;font-size:10.5px;color:#cfd8ea;font-weight:600;user-select:none}
  .eewcard table.hist{margin-top:0}
  .eewcard table.hist td{border-bottom:none;font-size:10.5px;padding:0 4px;color:var(--sub)}
  .eewcard.lv1{background:#12304fee;border-color:#3b9cff}
  .eewcard.lv2{background:#7a1010ee;border-color:#ff4d4d;animation:eewb 1s infinite}
  .eewcard.lv3{background:#2a0a3eee;border-color:#c04bff;animation:eewb .6s infinite}
  @keyframes eewb{50%{box-shadow:0 0 18px 4px #ff3b3b99}}
  #eewcards{display:flex;flex-direction:column;gap:8px}
  .tsucard{border-width:2px;padding:6px 8px}
  .tsucard .th{display:flex;align-items:center;gap:6px;margin-bottom:3px}
  .tsucard .tt{flex:1;min-width:0;font-size:11.5px;color:#cfd8ea;font-weight:600}
  .tsucard .tb{font-size:14px;font-weight:800;padding:1px 8px;border-radius:6px;background:#0006;white-space:nowrap}
  .tsucard .tm{font-size:11px;color:#cfd8ea}
  .tsucard .tx{font-size:11px;line-height:1.4;margin:3px 0}
  .tsucard .tw{max-height:210px;overflow:auto;margin-top:2px}
  .tsucard table{width:100%;border-collapse:collapse}
  .tsucard td{font-size:11px;padding:1px 3px;border-bottom:1px solid #ffffff18;white-space:nowrap}
  .tsucard td:first-child{white-space:normal;min-width:5em}
  .tsucard .tw{overflow-x:auto}
  .tsucard .chip{padding:0 5px;border-radius:5px;font-weight:700;font-size:10.5px}
  .tsucard.tl0{background:#1a2030ee;border-color:#3b4a66}
  .tsucard.tl1{background:#12304fee;border-color:#3b9cff}
  .tsucard.tl2{background:#4a3f08ee;border-color:#ffd23b}
  .tsucard.tl3{background:#7a1010ee;border-color:#ff4d4d;animation:eewb 1s infinite}
  .tsucard.tl4{background:#2a0a3eee;border-color:#c04bff;animation:eewb .6s infinite}
  #mxIco{position:absolute;z-index:1000;right:314px;bottom:226px;display:none;flex-direction:column;align-items:center;gap:3px}
  #mxIco .mk{font-size:11px;font-weight:700;color:#fff;text-shadow:0 0 3px #000,0 0 6px #000}
  #mxIco .mxb{width:78px;height:78px;border-radius:14px;display:flex;align-items:center;justify-content:center;border:2px solid rgba(255,255,255,.85);box-shadow:0 2px 10px rgba(0,0,0,.5);font-weight:900}
  #mxIco .n{font-size:46px;line-height:1}
  #mxIco .s{font-size:22px;line-height:1;align-self:flex-end;margin-bottom:13px;margin-left:1px}
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
  /* 画面切替タブ（強震モニタ / 地震情報） */
  #tabbar{position:fixed;z-index:2100;top:10px;right:58px;display:flex;background:var(--panel);
    border:1px solid var(--line);border-radius:999px;padding:3px;backdrop-filter:blur(6px)}
  #tabbar button{border:none;background:none;border-radius:999px;padding:6px 14px;font-size:13px;font-weight:700;
    color:var(--sub);position:relative;white-space:nowrap}
  #tabbar button.on{background:var(--acc);color:#04101f}
  #tabbar .badge{position:absolute;top:-5px;right:-3px;min-width:18px;height:18px;line-height:18px;border-radius:9px;
    background:#ff4d4d;color:#fff;font-size:11px;padding:0 5px;display:none;text-align:center}
  #infoView{display:none;position:fixed;inset:0;z-index:2000;background:var(--bg);flex-direction:column}
  #infoHead{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:10px 280px 10px 16px;border-bottom:1px solid var(--line)}
  #infoHead .it{font-size:16px;font-weight:800;margin-right:6px}
  #infoHead label{font-size:12px;color:var(--sub);display:flex;align-items:center;gap:5px}
  #infoHead input{width:auto}
  #iTime{font-size:13px;font-variant-numeric:tabular-nums;color:var(--sub);min-width:150px}
  #infoMain{flex:1;min-height:0;display:flex}
  #infoMapCol{flex:1 1 55%;min-width:0;position:relative;border-right:1px solid var(--line)}
  #imap{position:absolute;inset:0;background:#0b1220}
  #imLegend{position:absolute;z-index:1000;left:10px;bottom:10px;display:flex;gap:3px;align-items:center;padding:5px 7px;
    background:var(--panel);border:1px solid var(--line);border-radius:8px;font-size:11px;color:var(--sub)}
  #imLegend .chip{min-width:0;padding:1px 5px;font-size:11px}
  .imk{display:flex;align-items:center;justify-content:center;width:100%;height:100%;border:1px solid #0b1220;border-radius:5px;
    font-size:11px;font-weight:800;line-height:1;box-shadow:0 0 0 1px #0008}
  #infoBody{flex:1 1 45%;min-width:0;overflow:auto;padding:14px 16px 40px}
  .itw{max-height:340px;overflow:auto}
  .itbl{width:100%;border-collapse:collapse;font-size:12.5px}
  .itbl th,.itbl td{padding:5px 12px;border-bottom:1px solid #222b3b;text-align:left}
  .itbl th{position:sticky;top:0;background:var(--panel);font-size:11px;color:var(--sub);font-weight:700}
  .itbl .num{text-align:right;font-variant-numeric:tabular-nums}
  #infoBody .wrap{max-width:860px;margin:0 auto;display:flex;flex-direction:column;gap:14px}
  .iempty{color:var(--sub);font-size:13px;line-height:1.7;text-align:center;margin-top:60px}
  .igc{border:1px solid var(--line);border-radius:12px;background:var(--panel);overflow:hidden}
  .igh{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:10px 14px;background:#1d2636;border-bottom:1px solid var(--line)}
  .igh .gt{font-size:15px;font-weight:800}
  .igh .gs{flex:1;min-width:0;font-size:12.5px;color:#cfd8ea;line-height:1.5}
  .igh .gm{font-size:11px;color:#ffd23b}
  .ir{padding:10px 14px;border-bottom:1px solid #222b3b;font-size:13px;line-height:1.6}
  .ir:last-child{border-bottom:none}
  .ir.new{animation:inew 1.6s ease-out}
  @keyframes inew{0%{background:#3b9cff55}100%{background:transparent}}
  .ir .rh{display:flex;align-items:baseline;gap:8px;margin-bottom:4px}
  .ir .rh b{font-size:13.5px}
  .ir .rh .rt{margin-left:auto;font-size:11.5px;color:var(--sub);font-variant-numeric:tabular-nums}
  .ir .rh .ru{font-size:11px;color:#ffd23b}
  .ir .im{font-size:12.5px;color:#e6ebf5;margin:2px 0}
  .ir .ic{display:flex;gap:8px;align-items:flex-start;margin:3px 0}
  .ir .ic .chip{flex:none;min-width:64px}
  .ir .ia{flex:1;min-width:0;font-size:12.5px;color:#dfe6f3}
  .ir .ia i{font-style:normal;color:var(--sub);font-size:11px}
  .ir .itsu{margin-top:4px;font-size:12px;color:#9be7ff}
  .ir .inote{margin-top:4px;font-size:11px;color:var(--sub)}
  .ir .bl{font-size:12.5px;margin:2px 0;color:#e6ebf5}
  .ir .im{display:block}
  .ir .ia b{color:#cfd8ea;font-weight:700;margin-left:2px}
  .ir .ia .pf{display:inline-block;margin-right:10px}
  .eewcard .eh{height:auto;min-height:24px}
  .eewcard .ett{white-space:normal;overflow:visible;line-height:1.3}
  /* iPad など中間幅: 左の設定パネル（幅310px）と下の再生バー・実行ボタン・HUDが重ならないよう、パネルの右側へ寄せる */
  @media (min-width:701px) and (max-width:1400px){
    #bar{left:330px;right:10px;transform:none;width:auto;bottom:calc(14px + env(safe-area-inset-bottom,0px))}
    #runFab{left:calc(330px + (100% - 340px) / 2);bottom:calc(164px + env(safe-area-inset-bottom,0px))}
    #panel{max-height:calc(100% - 20px - env(safe-area-inset-bottom,0px))}
  }
  @media (min-width:701px) and (max-width:1100px){
    #hud{top:54px;left:330px;transform:none}
    #rightcol{top:100px;max-height:calc(100% - 300px)}
  }
  @media (max-width:700px){
    #bar{bottom:calc(14px + env(safe-area-inset-bottom,0px))}
    #tabbar{top:50px;left:10px;right:auto}
    #mxIco{right:auto;left:10px;bottom:226px}
    #infoView{padding-top:92px}
    #infoHead{padding-right:16px}
    #infoMain{flex-direction:column}
    #infoMapCol{flex:none;height:42%;border-right:none;border-bottom:1px solid var(--line)}
    #infoBody{flex:1 1 auto}
    #panel{width:calc(100% - 20px);max-height:62%;top:92px;display:none}
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
<button id="toggle">設定</button>
<button id="runFab" class="idle" title="現在の地震設定でシミュレーションを実行（Enterキーでも可）">► シミュレーション実行</button>

<div id="panel" class="open">
  <h1>Quakecast</h1>

  <h2>① 地震（複数指定できます）</h2>
  <div id="qlist"></div>
  <button id="qadd" class="sec">＋ 地震を追加</button>
  <div class="sl" style="display:flex;flex-wrap:wrap;gap:5px;margin-top:6px;align-items:center">
    <select id="slSel" style="flex:1;min-width:110px"></select>
    <button id="slSave" class="sec" style="width:auto;margin:0;padding:5px 8px;font-size:12px">保存</button>
    <button id="slLoad" class="sec" style="width:auto;margin:0;padding:5px 8px;font-size:12px">読込</button>
    <button id="slDel" class="sec" style="width:auto;margin:0;padding:5px 8px;font-size:12px">削除</button>
    <button id="slExp" class="sec" style="width:auto;margin:0;padding:5px 8px;font-size:12px">↓ ファイル</button>
    <button id="slImp" class="sec" style="width:auto;margin:0;padding:5px 8px;font-size:12px">↑ ファイル</button>
    <input id="slFile" type="file" accept=".json,application/json" style="display:none">
  </div>
  <div class="hint">選択中の地震の震源は、地図をクリックして設定できます。「発生（秒後）」で時間差を付けられます。観測点ごとの震度は、全地震の揺れのうち最大のものを表示します。緊急地震速報は地震ごとに独立して発表します。</div>

  <label style="display:flex;align-items:center;gap:6px;margin-top:10px"><input id="showPred" type="checkbox" checked style="width:auto"> 予想震度を地図に薄く塗る</label>
  <label style="display:flex;align-items:center;gap:6px;margin-top:6px"><input id="showObs" type="checkbox" style="width:auto"> その時点で観測された最大震度を表示（各観測点の最大値を保持）</label>
  <label style="display:flex;align-items:center;gap:6px;margin-top:6px"><input id="muOn" type="checkbox" style="width:auto"> 震度計が未入電となる可能性（計測震度5.5以上の観測点が、震度が大きいほど高い確率で「未入電(5弱以上)」になる。確率は低め：6.5で約1割）</label>
  <div style="margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,.12)">
    <label style="display:flex;align-items:center;gap:6px"><input id="sndOn" type="checkbox" checked style="width:auto"> 効果音（緊急地震速報・震度速報／地震情報・津波・最大震度の更新）</label>
    <label style="display:flex;align-items:center;gap:6px;margin-top:6px"><input id="spkOn" type="checkbox" checked style="width:auto"> 読み上げ（ブラウザ標準の日本語音声）</label>
    <label style="display:flex;align-items:center;gap:6px;margin-top:6px">音量 <input id="sndVol" type="range" min="0" max="1" step="0.05" value="0.7" style="flex:1;width:auto"> <button id="sndTest" class="sec" style="width:auto;margin:0;padding:3px 8px;font-size:12px">テスト</button></label>
    <div class="hint">再生中に発表された情報だけ鳴ります（シーク・巻き戻し時は鳴りません）。ブラウザの仕様で、画面を一度操作するまでは音が出ない場合があります。</div>
  </div>
  <label style="display:flex;align-items:center;gap:6px;margin-top:6px">海底震度計
    <select id="seaMode" style="width:auto"><option value="thin" selected>薄く表示（数字なし）</option><option value="off">非表示</option><option value="normal">通常表示</option></select></label>
  <div class="hint">黄色の○＋＝EEW推定震源／✕＝設定した震源（番号は地震番号）／地図の薄い塗り＝予想震度</div>

  <h2>② プリセット（選択中の地震に適用）</h2>
  <div class="presets" id="presets"></div>

  <button id="run" class="primary">► シミュレーション実行</button>
  <div id="status"></div>

  <div id="result" style="display:none">
    <div id="srcinfo" class="hint"></div>
    <h2 id="areasH">最終的な区域別最大震度（シミュレーション結果・上位）</h2>
    <table id="areas"></table>
  </div>

  <div class="foot">
    距離減衰式による簡易推定です。実際の強震モニタ（実観測波形）や気象庁の発表とは異なります。防災・避難判断には使用しないでください。<br>
    参考作品: 「リアルタイム地震ビューアー」(kotoho7様作)。効果音・読み上げの場面構成を参考に、音はブラウザで新規に合成しています。<br>
    海底観測点（data/seafloor.csv）は初期設定では薄く表示します（設定で非表示／通常表示に変更可）。区域別集計・警報対象・最大震度には含めません。<br>
    本プロジェクトは JQuake のような強震モニタ・地震津波情報等総合配信サイトの再現を目指しています。津波情報は簡易推定です。<br>
    地図: Tiles © Esri
  </div>
</div>

<div id="audioAsk" style="position:fixed;inset:0;z-index:5000;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.72);padding:16px">
  <div style="max-width:420px;width:100%;background:#1b2231;color:#e8edf7;border:1px solid rgba(255,255,255,.18);border-radius:12px;padding:18px 18px 16px;box-shadow:0 8px 30px rgba(0,0,0,.5);font-size:14px;line-height:1.6">
    <div style="font-size:16px;font-weight:700;margin-bottom:8px">効果音・読み上げを使いますか？</div>
    <div>地震情報の発表に合わせて、効果音と日本語の読み上げが流れます。</div>
    <ul style="margin:8px 0 10px;padding-left:20px;color:#c4cde0;font-size:13px">
      <li>緊急地震速報や津波警報の音は<b>大きく鳴ることがあります</b>。周囲や音量にご注意ください。</li>
      <li>読み上げはシミュレーションの内容です。<b>実際の防災・避難判断には使用しないでください</b>。</li>
      <li>音が出ない場合は、画面を一度タップしてください。</li>
      <li>選択は、設定パネルからいつでも変更できます。</li>
    </ul>
    <div style="display:flex;gap:8px">
      <button id="askOn" style="flex:1;padding:9px 6px;font-size:14px;font-weight:700;border:0;border-radius:8px;background:#2f7cf6;color:#fff;cursor:pointer">音声あり</button>
      <button id="askOff" style="flex:1;padding:9px 6px;font-size:14px;font-weight:700;border:1px solid rgba(255,255,255,.3);border-radius:8px;background:transparent;color:#e8edf7;cursor:pointer">音声なし</button>
    </div>
  </div>
</div>
<div id="audioHint" style="display:none;position:fixed;z-index:4000;top:96px;left:50%;transform:translateX(-50%);background:#2f7cf6;color:#fff;font-size:13px;font-weight:700;padding:8px 14px;border-radius:20px;box-shadow:0 4px 14px rgba(0,0,0,.45);cursor:pointer;max-width:90%;text-align:center">効果音がオフです。ここをタップして有効にする</div>
<div id="tabbar">
  <button id="tabMon" class="on">強震モニタ</button>
  <button id="tabInfo">地震情報<span class="badge" id="tabBadge">0</span></button>
</div>

<div id="infoView">
  <div id="infoHead">
    <span class="it">震度速報・地震情報</span>
    <button id="iRun" class="sec" style="width:auto;margin:0;padding:6px 12px">► シミュレーション実行</button>
    <button id="iRestart" title="最初に戻る">|◄</button>
    <button id="iPlay" title="再生/一時停止">►</button>
    <span id="iTime">—</span>
    <label><input id="infoAuto" type="checkbox" checked>更新時に自動で切り替え（強震モニタ表示中）</label>
  </div>
  <div id="infoMain">
    <div id="infoMapCol"><div id="imap"></div><div id="imLegend"></div></div>
    <div id="infoBody"></div>
  </div>
</div>

<div id="hud">
  <span><span class="lbl">経過</span><b id="hT">0.0</b> 秒</span>
  <span><span class="lbl">P波</span><b id="hP">-</b> km</span>
  <span><span class="lbl">S波</span><b id="hS">-</b> km</span>
  <span><span class="lbl">現在の最大</span><b id="hM">-</b></span>
</div>

<div id="mxIco"><div class="mk">最大震度</div><div class="mxb"></div></div>
<div id="rightcol">
  <div id="obs" class="card"><span class="ok">観測最大震度</span><span class="ov" id="obsV"></span><span class="oa" id="obsA"></span></div>
  <div id="eewcards"></div>
  <div id="tsucard" class="card tsucard tl0" style="display:none"></div>
  <div id="rank" class="card">
    <h3>リアルタイム震度ランキング（区域別）</h3>
    <table id="rankTbl"><tr><td style="color:var(--sub)">シミュレーション実行後に表示</td></tr></table>
  </div>
</div>

<div id="simClk" title="クリックで時刻を変更"><div id="clkT">--:--:--</div><div id="clkD"></div>
    <div class="clkCtl"><input type="datetime-local" step="1" id="clkIn"><button id="clkNow" type="button">現在時刻</button></div></div>
  <div id="bar">
  <div id="ctl">
    <button id="restart" title="最初に戻る">|◄</button>
    <button id="play" title="再生/一時停止">►</button>
    <input id="seek" type="range" min="0" max="0" step="1" value="0" disabled>
    <span id="tlabel">0.0 s</span>
    <select id="speed" title="再生速度">
      <option value="0.5">×0.5</option><option value="1" selected>×1</option>
      <option value="2">×2</option><option value="5">×5</option><option value="10">×10</option>
    </select>
    <select id="tsuSpeed" title="地震動の計算範囲を過ぎた後（津波フェーズ）の再生速度">
      <option value="1">津波×1</option><option value="5">津波×5</option><option value="10">津波×10</option><option value="30">津波×30</option><option value="60" selected>津波×60</option><option value="300">津波×300</option><option value="1800">津波×1800</option>
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
function shindoCls(x){
  x=x+1e-6; const th=[0.5,1.5,2.5,3.5,4.5,5.0,5.5,6.0,6.5]; let k=0;
  for(let i=0;i<th.length;i++) if(x>=th[i]) k=i+1; return k;
}
// 色テーブル: 0以上は震度階級ごとの固定色（10色）、0未満は従来どおり連続グラデーション
const LUT=[]; for(let k=0;k<LUT_N;k++){
  const v=Math.round((LUT_MIN+k*LUT_STEP)*10)/10;
  if(v>=0){ const c=CLASS_RGB[shindoCls(v)]; LUT.push('rgb('+c[0]+','+c[1]+','+c[2]+')'); }
  else{ const c=colorAt(v); LUT.push('rgb('+c[0]+','+c[1]+','+c[2]+')'); }
}
// 値→色テーブル番号（切り捨て: 4.46 が 4.5=震度5弱の色にならないように）
function lutK(v){ let k=Math.floor((v-LUT_MIN)/LUT_STEP+1e-4); return k<0?0:(k>=LUT_N?LUT_N-1:k); }
// 計測震度の表示（小数第2位以下切り捨て。6.46 を「6.5」と表示しない）
const fx1=v=>(Math.floor(v*10+1e-4)/10).toFixed(1);
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

/* ---------- 地震セットの保存・読込（localStorage＋JSONファイル） ---------- */
const SL_KEY='quakeSets_v1';
function slAll(){ try{ const o=JSON.parse(localStorage.getItem(SL_KEY)||'{}'); return (o&&typeof o==='object')?o:{}; }catch(_){ return {}; } }
function slPut(o){ try{ localStorage.setItem(SL_KEY,JSON.stringify(o)); return true; }catch(_){ $('status').textContent='保存できませんでした（ブラウザの保存が無効）'; return false; } }
function slClean(a){                                    // 入力検証（最大5件・範囲に丸める）
  if(!Array.isArray(a)) return null;
  const out=[];
  for(const q of a.slice(0,MAX_Q)){
    const lat=+q.lat, lon=+q.lon, mag=+q.mag, dep=+q.depth, dl=+(q.delay||0);
    if(![lat,lon,mag,dep,dl].every(isFinite)) continue;
    out.push({lat:clampN(lat,-90,90),lon:clampN(lon,-180,180),mag:clampN(mag,0,15),depth:clampN(dep,0,699),delay:Math.round(clampN(dl,0,300))});
  }
  return out.length?out:null;
}
function slRefresh(sel){
  const o=slAll(), names=Object.keys(o), box=$('slSel'); box.innerHTML='';
  if(!names.length){ const op=document.createElement('option'); op.value=''; op.textContent='（保存なし）'; box.appendChild(op); return; }
  names.forEach(n=>{ const op=document.createElement('option'); op.value=n; op.textContent=n+'（'+o[n].length+'件）'; box.appendChild(op); });
  if(sel&&o[sel]) box.value=sel;
}
function slApply(list){
  QS=list.map(q=>({...q})); selQ=0; renderQuakes(); syncMarkers();
  const q=QS[0]; if(q) map.panTo([q.lat,q.lon]);
}
$('slSave').onclick=()=>{
  const o=slAll(), cur=$('slSel').value;
  const nm=(prompt('保存名を入力（同じ名前は上書き）',cur||('地震セット'+(Object.keys(o).length+1)))||'').trim();
  if(!nm) return;
  o[nm]=QS.map(q=>({lat:q.lat,lon:q.lon,mag:q.mag,depth:q.depth,delay:q.delay}));
  if(slPut(o)){ slRefresh(nm); $('status').textContent='「'+nm+'」を保存しました（'+QS.length+'件）'; }
};
$('slLoad').onclick=()=>{
  const o=slAll(), nm=$('slSel').value, l=slClean(o[nm]);
  if(!l){ $('status').textContent='読み込める保存データがありません'; return; }
  slApply(l); $('status').textContent='「'+nm+'」を読み込みました（'+l.length+'件）';
};
$('slDel').onclick=()=>{
  const o=slAll(), nm=$('slSel').value; if(!nm||!o[nm]) return;
  if(!confirm('「'+nm+'」を削除しますか？')) return;
  delete o[nm]; if(slPut(o)) slRefresh();
};
$('slExp').onclick=()=>{
  const blob=new Blob([JSON.stringify({version:1,quakes:QS},null,1)],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='quakes.json'; a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),2000);
};
$('slImp').onclick=()=>$('slFile').click();
$('slFile').addEventListener('change',async()=>{
  const f=$('slFile').files[0]; $('slFile').value=''; if(!f) return;
  try{
    const j=JSON.parse(await f.text()), l=slClean(Array.isArray(j)?j:j.quakes);
    if(!l) throw new Error('形式が不正です');
    slApply(l); $('status').textContent='ファイルから '+l.length+'件を読み込みました';
  }catch(e){ $('status').textContent='読込エラー: '+e.message; }
});
slRefresh();

const PRESETS=[
  {n:'阪神・淡路(1995)',lat:34.60,lon:135.04,m:7.3,d:16},
  {n:'東北沖(2011)',lat:38.10,lon:142.86,m:9.0,d:24},
  {n:'熊本(2016)',lat:32.74,lon:130.76,m:7.3,d:12},
  {n:'能登半島(2024)',lat:37.49,lon:137.27,m:7.6,d:15},
  {n:'南海トラフ(想定)',lat:33.00,lon:136.50,m:9.0,d:20},
  {n:'首都直下(想定)',lat:35.68,lon:139.77,m:7.3,d:30},
  {n:'小笠原諸島西方沖地震(2015・深発)',lat:27.9,lon:140.7,m:8.1,d:682}
];
PRESETS.forEach(p=>{
  const b=document.createElement('button'); b.textContent=p.n;
  b.onclick=()=>{
    const q=QS[selQ]; q.lat=p.lat; q.lon=p.lon; q.mag=p.m; q.depth=p.d;     // 発生時刻（秒後）は据え置き
    renderQuakes(); syncMarkers(); map.panTo([p.lat,p.lon]);
  };
  $('presets').appendChild(b);
});
{ const b=document.createElement('button'); b.textContent='複数地震の例(南海トラフ+首都直下)';
  b.onclick=()=>{
    QS=[{lat:33.00,lon:136.50,mag:9.0,depth:20,delay:0},{lat:35.68,lon:139.77,mag:7.3,depth:30,delay:20}];
    selQ=0; renderQuakes(); syncMarkers(); map.setView([34.6,138.0],6);
  };
  $('presets').appendChild(b); }
$('toggle').onclick=()=>$('panel').classList.toggle('open');
if(window.innerWidth<=700) $('panel').classList.remove('open');

/* ---------- 観測点 ---------- */
let N=0, SLON=null, SLAT=null, WX=null, WY=null, OX=0, OY=0, SC=256, PX=null, PY=null, VALS=null, viewDirty=true, AREA=null, ANAMES={}, AM=null, SEA=null, SEAIDX=[];
fetch('/api/stations').then(r=>r.json()).then(j=>{
  SLON=Float32Array.from(j.lon); SLAT=Float32Array.from(j.lat); N=SLON.length;
  AREA=Int32Array.from(j.area); ANAMES=j.anames; SEA=Uint8Array.from(j.sea||new Array(N).fill(0)); SEAIDX=[]; for(let i=0;i<N;i++) if(SEA[i]) SEAIDX.push(i); let mx=0; for(const a of AREA) if(a>mx) mx=a; AM=new Float32Array(mx+1);
  WX=new Float32Array(N); WY=new Float32Array(N);   // 正規化メルカトル座標(0..1)。再投影を足し算と掛け算だけで行うため最初に1回だけ計算
  for(let i=0;i<N;i++){ const sn=Math.sin(SLAT[i]*Math.PI/180); WX[i]=(SLON[i]+180)/360; WY[i]=0.5-Math.log((1+sn)/(1-sn))/(4*Math.PI); }
  PX=new Float32Array(N); PY=new Float32Array(N); VALS=new Float32Array(N).fill(-3.1); viewDirty=true; loadGeo(); loadTsuGeo();
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
  SC=256*Math.pow(2,map.getZoom()); const p0=map.latLngToContainerPoint([0,0]); OX=p0.x-0.5*SC; OY=p0.y-0.5*SC;
  if(WX) for(let i=0;i<N;i++){ PX[i]=OX+WX[i]*SC; PY[i]=OY+WY[i]*SC; }   // 以前は観測点ごとに latLngToContainerPoint を呼んでいた（毎フレーム数千回）
  viewDirty=false;
}
function circlePts(lat0,lon0,rkm,n){
  const R=6371.0, d=rkm/R, la=lat0*Math.PI/180, lo=lon0*Math.PI/180, pts=[];
  for(let i=0;i<=n;i++){
    const th=i/n*2*Math.PI;
    const lat=Math.asin(Math.sin(la)*Math.cos(d)+Math.cos(la)*Math.sin(d)*Math.cos(th));
    const lon=lo+Math.atan2(Math.sin(th)*Math.sin(d)*Math.cos(la),Math.cos(d)-Math.sin(la)*Math.sin(lat));
    const sn=Math.sin(lat); pts.push({x:OX+((lon*180/Math.PI+180)/360)*SC, y:OY+(0.5-Math.log((1+sn)/(1-sn))/(4*Math.PI))*SC});
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
let PEAK=null, PEAKV=null, peakF=-1;   // 観測点ごとの「その時点までの最大震度」(×10)
let CUMV=null, CUMI=null;   // フレームごとの「その時点までの最大震度(×10)」と観測点
let CUMV2=null, CUMI2=null; // 同上（未入電になった観測点を除く）
let MUF=null, MUEND=null, MUPK=null, FX=[], MUFIRST=1e9, MUI=-1;   // 観測点ごとの未入電になるフレーム(-1=ならない)／最初の未入電の時刻と観測点
const MU_PMAX=0.2;   // 未入電確率の上限係数（計測震度5.5で0、6.0で約6%、6.5で約10%、7.0で約13%）。大きくするほど未入電が増える
const muOn=()=>{ const e=document.getElementById('muOn'); return !!(e&&e.checked); };
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

const BKT=Array.from({length:LUT_N},()=>[]);
function draw(ts){
  const s=map.getSize(); ctx.clearRect(0,0,s.x,s.y);
  if(!N) return;
  updatePoints();   // 毎フレーム再投影（ズーム/パン中も地図とずれない）
  // 津波情報への切り替え: 地震動の計算範囲が終わり津波フェーズに入ったら、地震情報（観測点の震度・予想震度・
  // P/S波・EEW・ランキング・観測最大震度）は消して、津波の表示だけにする。巻き戻せば地震情報に戻る。
  const clr=!!(sim && sim.hasTsu && tcur>=sim.tSeis);
  let fcur=0; if(sim){ fcur=Math.round(tcur/sim.dt); if(fcur<0)fcur=0; if(fcur>sim.nf-1)fcur=sim.nf-1; }
  const muEn=!!(sim&&MUF&&muOn());
  if(sim) valuesAt(tcur);
  else { const tk=Math.floor((ts||0)/160); for(let i=0;i<N;i++) VALS[i]=-3.1+(frac(Math.sin(i*12.9898+tk*78.233)*43758.5453)-0.5)*0.14; }

  let DV=VALS;                                         // 地図に描く値（既定は現在値）
  if(sim && $('showObs').checked){                     // 各観測点の「その時点までの最大震度」を保持して表示
    if(!PEAK||PEAK.length!==N){ PEAK=new Int8Array(N).fill(-128); PEAKV=new Float32Array(N); peakF=-1; }
    const nf=sim.nf; let f=Math.round(tcur/sim.dt); if(f<0)f=0; if(f>nf-1)f=nf-1;
    if(f<peakF){ PEAK.fill(-128); peakF=-1; }          // 巻き戻し時は最初から積み直す
    for(let g=peakF+1;g<=f;g++){ const o=g*N; for(let i=0;i<N;i++){ const x=Q[o+i]; if(x>PEAK[i]) PEAK[i]=x; } }
    peakF=f;
    for(let i=0;i<N;i++) PEAKV[i]=PEAK[i]*0.1;
    DV=PEAKV;
  }
  const z=map.getZoom(), r0=Math.max(2.0,(z-3.6)*0.95);
  const buckets=BKT; for(let k=0;k<LUT_N;k++) buckets[k].length=0;   // 配列は使い回す（毎フレームの確保とGCを避ける）
  let vmax=-99, vmaxI=-1; AM.fill(-99);
  const muList=[];                                     // 未入電になっている観測点
  for(let i=0;i<N;i++){
    const v=VALS[i];
    const isSea=!!(SEA&&SEA[i]);
    const mu=muEn&&!isSea&&MUF[i]>=0&&fcur>=MUF[i];     // 未入電の観測点
    if(!isSea){                                          // 海底観測点は最大・ランキングに含めない
      if(mu){ if(fcur<MUEND[i]) muList.push(i); }   // 未入電の表示は発生から20〜30秒で消える（集計からは除外したまま）
      else{ if(v>vmax){ vmax=v; vmaxI=i; } const ar=AREA[i]; if(v>AM[ar]) AM[ar]=v; }
    }
    if(isSea && seaMode!=='normal') continue;           // 海底は「薄く/非表示」のとき通常描画から外す
    if(mu) continue;                                     // 未入電は下で別に描く
    buckets[lutK(DV[i])].push(i);
  }
  // EEWの予想震度で地図を薄く塗りつぶす（観測点ごとの予想震度の階級色。高い階級を上に重ねる）
  if(sim && !clr && $('showPred').checked){ const pv=mergedPred(); if(pv) drawPredFill(pv,z); }
  // 海底震度計: 薄く（数字・縁取りなし、小さめ）。陸上の観測点より先に描いて下に敷く
  if(!clr && seaMode==='thin' && SEAIDX.length){
    ctx.save(); ctx.globalAlpha=0.3;
    for(const i of SEAIDX){
      const v=DV[i]; const k=lutK(v);
      const rr=v>=0.49?Math.max(2.5,Math.min(16,5.5+(z-4)*1.2)*0.5):r0*0.7;
      ctx.fillStyle=LUT[k]; ctx.beginPath(); ctx.arc(PX[i],PY[i],rr,0,6.2832); ctx.fill();
    }
    ctx.restore();
  }
  // 震度1以上の観測点は「円の中に震度階級の数字」で表示（ズームに応じて円を拡大）
  const rn=Math.min(16,5.5+(z-4)*1.2), showTxt=rn>=6.5;
  ctx.textAlign='center'; ctx.textBaseline='middle';
  for(let k=0;k<LUT_N;k++){
    const b=buckets[k]; if(clr||!b.length) continue;
    const v=LUT_MIN+k*LUT_STEP;
    const num=v>=0.49, cl=num?shindoCls(v+0.01):0;
    const zero=!num&&v>=-0.05;                          // 震度0（計測震度0〜0.5未満）: 震度1以上と同じ大きさの灰色の円（数字なし）
    const r=num?rn*(cl>=5?1.2:1.0):(zero?rn:r0*(v<0?0.85:1.0));
    ctx.fillStyle=zero?'#7f8896':LUT[k]; ctx.beginPath();
    for(let j=0;j<b.length;j++){ const i=b[j]; ctx.moveTo(PX[i]+r,PY[i]); ctx.arc(PX[i],PY[i],r,0,6.2832); }
    ctx.fill();
    if(num||zero){
      ctx.strokeStyle='rgba(255,255,255,.85)'; ctx.lineWidth=1; ctx.stroke();
      if(num&&showTxt){
        ctx.font='bold '+Math.round(r*(cl>=5?1.0:1.25))+'px sans-serif';
        ctx.fillStyle=(cl===2||cl===3)?'#111':'#fff';
        const t=NUM_LABEL[cl];
        for(let j=0;j<b.length;j++){ const i=b[j]; ctx.fillText(t,PX[i],PY[i]+0.5); }
      }
    }
  }
  if(!clr&&muList.length){                               // 未入電: 5弱と同じ色で数字なしの円
    const r=rn*1.2; ctx.fillStyle=clsColor(5); ctx.beginPath();
    for(const i of muList){ ctx.moveTo(PX[i]+r,PY[i]); ctx.arc(PX[i],PY[i],r,0,6.2832); }
    ctx.fill(); ctx.strokeStyle='rgba(255,255,255,.85)'; ctx.lineWidth=1; ctx.stroke();
  }
  if(!clr&&FX.length){                                   // 最大震度更新時の発光エフェクト（約3.5秒）
    const now=performance.now(); ctx.save();
    for(let k=FX.length-1;k>=0;k--){
      const e=FX[k], age=(now-e.t0)/1000; if(age>3.5){ FX.splice(k,1); continue; }
      const a=1-age/3.5, x=PX[e.i], y=PY[e.i], col=clsColor(e.c), r0=rn*1.2;
      ctx.globalAlpha=0.55*a; ctx.shadowColor=col; ctx.shadowBlur=26; ctx.fillStyle=col;
      ctx.beginPath(); ctx.arc(x,y,r0*(1.5+0.35*Math.sin(age*9)),0,6.2832); ctx.fill();
      ctx.shadowBlur=0; ctx.lineWidth=3; ctx.strokeStyle='#fff';
      for(let j=0;j<2;j++){ const p=(age*0.9+j*0.5)%1; ctx.globalAlpha=a*(1-p); ctx.beginPath(); ctx.arc(x,y,r0+p*46,0,6.2832); ctx.stroke(); }
    }
    ctx.restore();
  }
  const setDisp=(el,v)=>{ if(el._d!==v){ el.style.display=v; el._d=v; } };
  setDisp($('rank'),clr?'none':''); setDisp($('eewcards'),clr?'none':'');
  if(!clr) updateRank(ts);
  if(sim){
    if(!clr) updateEew();
    { const tsuS=tsuState(tcur); drawTsuCoast(tsuS,z); drawTsuBars(tsuS,z); updateTsu(ts,tsuS); }
    if(!clr) sim.quakes.forEach(q=>{                    // 各地震の実際のP波・S波（発生後のみ）
      const tl=tcur-q.delay; if(tl<0) return;
      drawCircle(q.lat,q.lon,interp(q.P,tl),'rgba(90,200,255,.95)',1.6);
      drawCircle(q.lat,q.lon,interp(q.S,tl),'rgba(255,95,95,.98)',2.2);
    });
    if(!clr) eewCurs.forEach(cur=>{
      if(!cur) return;
      if(sim.ett && cur.t0!=null){                     // EEW推定震源から見たP波・S波（推定、薄く破線）
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
    $('hP').textContent=clr?'—':hv('P');
    $('hS').textContent=clr?'—':hv('S');
    const hm=$('hM'); const m=Math.max(vmax,-3);
    if(clr) $('obs').style.display='none';              // updateObs が display を直接触るため setDisp は使わない
    else { const nf=sim.nf; let f=Math.round(tcur/sim.dt); if(f<0)f=0; if(f>nf-1)f=nf-1;
      if(CUMV&&CUMV.length===nf){ if(muEn) updateObs(CUMV2[f]*0.1,CUMI2[f],f>=MUFIRST); else updateObs(CUMV[f]*0.1,CUMI[f],false); } else updateObs(-9,-1,false); }
    updateMxIcon(vmax,!clr,muList.length>0);            // その瞬間の最大震度（陸上観測点の現在値の最大）
    if(clr){ hm.textContent='—'; hm.style.color='#93a0b8'; }
    else if(m<0.5){ hm.textContent='—'; hm.style.color='#93a0b8'; }
    else { hm.textContent=fx1(m)+'（'+CLASS_LABEL[shindoCls(m)]+'）'; hm.style.color=clsColor(shindoCls(m)); }
  }
}

let seaMode='thin';
try{ const m=localStorage.getItem('seaMode'); if(m==='off'||m==='thin'||m==='normal') seaMode=m; }catch(_){}
$('seaMode').value=seaMode;
$('seaMode').addEventListener('change',()=>{ seaMode=$('seaMode').value; try{ localStorage.setItem('seaMode',seaMode); }catch(_){} });
let lastRank=0, eewCurs=[], eewCards=[];
function updateRank(ts){
  if(!sim||!AM||(ts||0)-lastRank<250) return; lastRank=ts||0;
  const arr=[]; for(let a=0;a<AM.length;a++){ if(AM[a]>=0.5) arr.push([a,AM[a]]); }
  arr.sort((p,q)=>q[1]-p[1]);
  let h='';
  arr.slice(0,10).forEach((e,i)=>{
    const c=shindoCls(e[1]);
    h+='<tr><td class="no">'+(i+1)+'</td><td>'+(ANAMES[e[0]]||('区域'+e[0]))+'</td><td class="num">'+fx1(e[1])+
       '</td><td class="num"><span class="chip" style="background:'+clsColor(c)+'">'+CLASS_LABEL[c]+'</span></td></tr>';
  });
  $('rankTbl').innerHTML=h||'<tr><td style="color:#93a0b8">震度1以上の観測なし</td></tr>';
}
const EEW_LBL=['待機中','予報','警報','特別警報'];
function eewSet(el,html){ if(el._h!==html){ el.innerHTML=html; el._h=html; } }
function eewTxt(el,t){ if(el._h!==t){ el.textContent=t; el.title=t; el._h=t; } }
// その時点で観測された最大震度（陸上観測点）。チェックボックスで表示/非表示
// その瞬間の最大震度のアイコン（EEW欄の左隣。「最大震度を表示」の設定とは無関係に常に出す）
function updateMxIcon(m,show,mu){
  const el=$('mxIco');
  if(!show){ if(el._d!=='none'){ el.style.display='none'; el._d='none'; } return; }
  if(el._d!=='flex'){ el.style.display='flex'; el._d='flex'; }
  const c=(mu&&(m<0.5||shindoCls(m)<5))?-2:(m<0.5?-1:shindoCls(m));   // -2: 未入電（5弱以上が未入電で、入電済みの最大が5弱未満のとき）
  if(el._k===c) return; el._k=c;
  const box=el.querySelector('.mxb');
  if(c===-2){ box.style.background=clsColor(5); box.style.color='#fff'; box.innerHTML=''; return; }
  if(c<0){ box.style.background='#3a4357'; box.style.color='#93a0b8'; box.innerHTML='<span class="n">-</span>'; return; }
  const L=CLASS_LABEL[c], n=L.charAt(0), s=L.length>1?L.slice(1):'';
  box.style.background=clsColor(c); box.style.color=(c===2||c===3)?'#111':'#fff';
  box.innerHTML='<span class="n">'+n+'</span>'+(s?'<span class="s">'+s+'</span>':'');
}
function updateObs(m,i,mu){
  const box=$('obs'), on=!!sim && $('showObs').checked;
  box.style.display=on?'flex':'none'; if(!on) return;
  if(mu&&(m<0.5||i<0||shindoCls(m)<5)){
    eewSet($('obsV'),'<span class="chip" style="background:'+clsColor(5)+';color:#fff">未入電</span>');
    eewTxt($('obsA'),MUI>=0?(ANAMES[AREA[MUI]]||('区域'+AREA[MUI])):''); return; }
  if(m<0.5||i<0){ eewSet($('obsV'),'<span style="color:#93a0b8">—</span>'); eewTxt($('obsA'),''); return; }
  const c=shindoCls(m), tc=(c===2||c===3)?'#111':'#fff';
  eewSet($('obsV'),'<span class="chip" style="background:'+clsColor(c)+';color:'+tc+'">'+CLASS_LABEL[c]+'</span> '+fx1(m));
  eewTxt($('obsA'),ANAMES[AREA[i]]||('区域'+AREA[i]));
}
// 地震の数だけ緊急地震速報カードを作る（報の履歴は初期状態で閉じる）
function buildEewCards(n){
  const box=$('eewcards'); box.innerHTML=''; eewCards=[]; eewCurs=new Array(n).fill(null);
  for(let i=0;i<n;i++){
    const c=document.createElement('div'); c.className='card eewcard lv0';
    c.innerHTML='<div class="eh"><span class="ett r-ett">緊急地震速報'+(n>1?'　<span style="color:'+QCOL[i%QCOL.length]+'">●</span>地震'+(i+1):'')+'</span>'
      +'</div>'
      +'<div class="tiles">'
      +'<div class="tile"><div class="k">予想最大震度</div><div class="v r-max"><span class="chipbig" style="background:#3a4357">-</span></div></div>'
      +'<div class="tile"><div class="k">マグニチュード</div><div class="v r-mag">-</div></div>'
      +'<div class="tile"><div class="k">深さ</div><div class="v eewDep r-dep">-</div></div></div>'
      +'<div class="pos r-pos">シミュレーション実行後に表示</div>'
      +'<div class="warn r-warn"></div>'
      +'<details class="histbox"><summary>報の履歴</summary><table class="hist r-hist"></table></details>';
    box.appendChild(c);
    const els={}; ['ett','max','mag','dep','pos','warn','hist'].forEach(k=>els[k]=c.querySelector('.r-'+k));
    eewCards.push({el:c, r:k=>els[k]});
  }
}
buildEewCards(1);
function updateEew(){ sim.quakes.forEach((qk,i)=>updateEewCard(qk,i)); }
// 見出し: 緊急地震速報(予報/警報/特別警報)　三陸沖で地震　#報数
function eewTitle(i,cur,lvmax){
  const n=sim?sim.quakes.length:1;
  const mk=n>1?'　<span style="color:'+QCOL[i%QCOL.length]+'">●</span>地震'+(i+1):'';
  if(!cur) return '緊急地震速報'+mk;
  return '緊急地震速報('+EEW_LBL[lvmax]+')'+(cur.plum?'<span class="plumTag">PLUM報</span>':'　')+escH(cur.epiName||'震源付近')+'で地震　#'+cur.no+(cur.final?'(最終)':'')+mk;
}
function updateEewCard(qk,i){
  const C=eewCards[i]; if(!C) return;
  const card=C.el, R=C.r, e=qk.eew;
  let cur=null, lvmax=0;
  if(e){ for(let k=0;k<e.reports.length;k++){ const r=e.reports[k]; if(r.t>tcur) break; cur=r; if(r.lv>lvmax) lvmax=r.lv; } }
  eewCurs[i]=cur;
  if(!cur){
    card.className='card eewcard lv0';
    eewSet(R('ett'),eewTitle(i,null,0));
    eewSet(R('max'),'<span class="chipbig" style="background:#3a4357">-</span>');
    eewSet(R('mag'),'-'); eewSet(R('dep'),'-');
    eewTxt(R('pos'),e?(tcur<qk.delay?'発生前（'+qk.delay+'秒後に発生）':'観測点でP波を検知中…'):'発表基準に達しませんでした');
    eewTxt(R('warn'),''); eewSet(R('hist'),'');
    return;
  }
  card.className='card eewcard lv'+lvmax;
  eewSet(R('ett'),eewTitle(i,cur,lvmax));
  const tc=(cur.maxCls===2||cur.maxCls===3)?'#111':'#fff';
  eewSet(R('max'),'<span class="chipbig" style="background:'+clsColor(cur.maxCls)+';color:'+tc+'">'+cur.maxLabel+'</span>');
  eewSet(R('mag'),'M'+cur.mag.toFixed(1));
  eewSet(R('dep'),cur.depth+'<small>km</small>');
  eewTxt(R('pos'),'');                                   // 震源名・座標の行は出さない（発表前後の状態表示のみ使用）
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

/* ---------- 津波警報・注意報・津波情報 ---------- */
const TSU_LV=['','津波予報','津波注意報','津波警報','大津波警報'];
const TSU_COL=['#555','#6cc8ff','#ffd23b','#ff3030','#c04bff'];
const TSU_MSG={
  4:'大津波警報を発表しました。巨大な津波が襲来します。ただちに高台や避難ビルなど安全な場所へ避難してください。',
  3:'津波警報を発表しました。津波が襲来します。ただちに海岸から離れ、高台や避難ビルなど安全な場所へ避難してください。',
  2:'津波注意報を発表しました。海の中や海岸付近は危険です。海の中にいる人はただちに海から上がって、海岸から離れてください。',
  1:'若干の海面変動が予想されますが、被害の心配はありません。'};
let simClock0=Date.now();
const pad2=n=>('0'+n).slice(-2);
// 時計: clkOrigin=ユーザーが指定した起点時刻(ms)。null なら現在時刻。シミュレーション中は simClock0+経過時間を表示
let clkOrigin=null, clkSet0=Date.now();
const WD=['日','月','火','水','木','金','土'];
function clkLocalStr(ms){ const d=new Date(ms); return d.getFullYear()+'-'+pad2(d.getMonth()+1)+'-'+pad2(d.getDate())+'T'+pad2(d.getHours())+':'+pad2(d.getMinutes())+':'+pad2(d.getSeconds()); }
function clkNowMs(){ if(sim) return simClock0+tcur*1000; return clkOrigin===null?Date.now():clkOrigin+(Date.now()-clkSet0); }
function clkRefresh(){ const el=document.getElementById('clkT'); if(!el) return; const d=new Date(clkNowMs());
  el.textContent=pad2(d.getHours())+':'+pad2(d.getMinutes())+':'+pad2(d.getSeconds());
  document.getElementById('clkD').textContent=d.getFullYear()+'/'+pad2(d.getMonth()+1)+'/'+pad2(d.getDate())+'('+WD[d.getDay()]+')'+(clkOrigin!==null?' ・設定':''); }
function clkSetTo(ms){ if(ms===null){ clkOrigin=null; if(sim) simClock0=Date.now()-tcur*1000; }
  else { if(sim){ simClock0=ms-tcur*1000; clkOrigin=simClock0; } else { clkOrigin=ms; clkSet0=Date.now(); } }
  clkRefresh(); }
setInterval(clkRefresh,250);
document.addEventListener('DOMContentLoaded',()=>{
  const box=document.getElementById('simClk'), inp=document.getElementById('clkIn');
  document.getElementById('clkT').onclick=()=>{ box.classList.toggle('edit'); inp.value=clkLocalStr(clkNowMs()); };
  inp.onchange=()=>{ const ms=Date.parse(inp.value); if(!isNaN(ms)) clkSetTo(ms); };
  document.getElementById('clkNow').onclick=()=>{ clkSetTo(null); inp.value=clkLocalStr(Date.now()); };
  clkRefresh(); });
function clk(t){ const d=new Date(simClock0+t*1000); return pad2(d.getHours())+':'+pad2(d.getMinutes())+':'+pad2(d.getSeconds()); }
function fmtT(t){ if(t<600) return t.toFixed(1)+' s'; const s=Math.floor(t), h=Math.floor(s/3600), m=Math.floor(s%3600/60), x=s%60;
  return (h?h+':'+pad2(m):m)+':'+pad2(x); }
function tToS(t){ if(!sim.tsuExt) return Math.round(t*10);
  if(t<=sim.tSeis) return Math.round(t/sim.tSeis*1000);
  return Math.round(1000+(t-sim.tSeis)/(sim.tEnd-sim.tSeis)*1000); }
function sToT(s){ if(!sim.tsuExt) return s/10;
  if(s<=1000) return s/1000*sim.tSeis;
  return sim.tSeis+(s-1000)/1000*(sim.tEnd-sim.tSeis); }
function tsuState(t){
  // 報の番号は地震ごとに1から振られているため、複数の地震があると「第1報」の表示が
  // 更新されない。全地震の報を発表時刻順に通し番号(gno)で数え直し、最新の報を last とする。
  const list=new Map(), obs=new Map(); let last=null, any=false, gno=0;
  for(let qi=0;qi<sim.quakes.length;qi++){
    const qk=sim.quakes[qi];
    const T=qk.tsu; if(!T||!T.reports.length) continue; any=true;
    let cur=null;
    for(const r of T.reports){
      if(r.t>t) break; cur=r; gno++;
      if(!last||r.t>=last.t) last=Object.assign({},r,{qi:qi});
    }
    if(cur){
      for(const a of cur.areas){ const e=list.get(a.i);
        if(!e) list.set(a.i,{i:a.i,lv:a.lv,hl:a.hl,ts:a.ts,imm:a.imm});
        else{ if(a.lv>e.lv){ e.lv=a.lv; e.hl=a.hl; } if(a.ts<e.ts) e.ts=a.ts; e.imm=e.imm||a.imm; } }
    }
    for(const o of T.obs){
      if(t<o.m[0][0]) continue;
      let e=obs.get(o.a); if(!e){ e={ft:o.m[0][0],mx:0,mt:0,fin:false,segs:[]}; obs.set(o.a,e); }
      const vis=o.m.filter(x=>x[0]<=t), lv=vis[vis.length-1];
      if(lv && lv[1]>e.mx){ e.mx=lv[1]; e.mt=lv[0]; e.segs=vis.map(x=>x[1]);
        let mw=vis[0]; for(const x of vis){ if(x[1]>=lv[1]-1e-9){ mw=x; break; } }
        e.mw=mw; e.cur=lv; e.f1=o.m[0]; }   // mw: 最大波（第N波・押し/引き）、cur: 最大を更新した波、f1: 第1波   // segs: 観測ごとの累積最大波高（棒グラフの段）
      if(t>=o.pk) e.fin=true;
    }
  }
  const arr=Array.from(list.values()); arr.sort((p,q)=>q.lv-p.lv||p.ts-q.ts);
  return {list:arr,obs,last,any,gno};
}
/* ---------- 津波予報区の海岸線（data/tsunami_areas.geojson があるときだけ使用） ----------
   予報区名で TSU_AREAS と突き合わせ、海岸線に沿って色を付ける。
   名前が一致しない予報区・ファイルが無い場合は、従来の代表点マーカーで描く。 */
let TSUGEO_RAW=null, TSUGEO=null;
function loadTsuGeo(){
  fetch('/api/tsunami_geojson').then(r=>r.ok?r.json():null).then(gj=>{ if(gj&&gj.features) TSUGEO_RAW=gj; }).catch(()=>{});
}
function buildTsuGeo(){
  const norm=s=>String(s||'').normalize('NFKC').replace(/[\s・･·、,，]/g,'');
  const AR=sim.tsuAreas, byName={}; AR.forEach((a,i)=>{ byName[norm(a[0])]=i; });
  const KEYS=['name','Name','NAME','area_name','AREA_NAME','regionName','Area','area','津波予報区名','予報区名','名称'];
  const linesOf=g=>g.type==='LineString'?[g.coordinates]:g.type==='MultiLineString'?g.coordinates
    :g.type==='Polygon'?g.coordinates:g.type==='MultiPolygon'?[].concat(...g.coordinates):[];
  const out={}, left=[], skipped=[]; let hit=0;
  for(const f of TSUGEO_RAW.features){
    const p=f.properties||{}, g=f.geometry;
    let id=null, nm='';
    for(const k of KEYS){ if(p[k]!=null){ if(!nm) nm=String(p[k]); if(byName[norm(p[k])]!=null){ id=byName[norm(p[k])]; break; } } }
    const lines=g?linesOf(g).filter(l=>l.length>=2):[];
    if(!lines.length){ skipped.push(nm||'(名称なし)'); continue; }
    if(id==null){ left.push({nm:nm,lines:lines}); continue; }
    (out[id]=out[id]||[]).push(...lines); hit++;
  }
  // 名前が一致しなかった予報区は、(1)名称の部分一致 (2)代表点に最も近い余りの海岸線 の順で割り当てる
  const fuzzy=[];
  for(let i=0;i<AR.length;i++){
    if(out[i]) continue;
    const n=norm(AR[i][0]);
    const k=left.findIndex(L=>{ const m=norm(L.nm); return m&&(m.includes(n)||n.includes(m)); });
    if(k>=0){ out[i]=left[k].lines; fuzzy.push(AR[i][0]+' ← '+left[k].nm+'（名称の部分一致）'); left.splice(k,1); }
  }
  const dkm=(i,lines)=>{ const la=AR[i][1], lo=AR[i][2], kx=111.32*Math.cos(la*Math.PI/180); let b=1e9;
    for(const l of lines) for(const c of l){ const d=Math.hypot((c[0]-lo)*kx,(c[1]-la)*110.57); if(d<b) b=d; } return b; };
  const pairs=[];
  for(let i=0;i<AR.length;i++){ if(out[i]) continue; left.forEach((L,k)=>pairs.push([dkm(i,L.lines),i,k])); }
  pairs.sort((a,b)=>a[0]-b[0]);
  const usedI=new Set(), usedK=new Set();
  for(const [d,i,k] of pairs){
    if(d>60||usedI.has(i)||usedK.has(k)) continue;
    out[i]=left[k].lines; usedI.add(i); usedK.add(k);
    fuzzy.push(AR[i][0]+' ← '+left[k].nm+'（最寄り '+Math.round(d)+'km）');
  }
  TSUGEO=out; TSU_ANCH={}; TSUW={}; TSUPC={};
  console.log('tsunami_areas.geojson: '+hit+'/'+TSUGEO_RAW.features.length+' features matched by name, '+Object.keys(out).length+'/'+AR.length+' areas have a coastline');
  if(fuzzy.length) console.log('tsunami_areas.geojson: 名称が違うが割り当てた予報区:\n  '+fuzzy.join('\n  '));
  const noGeo=AR.map(a=>a[0]).filter((n,i)=>!out[i]);
  if(noGeo.length) console.warn('tsunami_areas.geojson: 海岸線が見つからない予報区（代表点マーカーで表示）:',noGeo);
  const unused=left.filter((L,k)=>!usedK.has(k)).map(L=>L.nm);
  if(unused.length) console.warn('tsunami_areas.geojson: どの予報区にも対応しなかった feature:',unused);
  if(skipped.length) console.warn('tsunami_areas.geojson: 線・面の形状が無く使えなかった feature:',skipped);
}

/* ---------- 津波観測値の棒グラフ（観測地点から上へ積み上がる） ---------- */
let TSU_ANCH={}, TSUW={}, TSUPC={};   // TSUW: 予報区ごとの海岸線の世界座標、TSUPC: 画面座標（地図が動かない間は使い回す）
function tsuWorld(i){
  let w=TSUW[i]; if(w) return w;
  w=(TSUGEO[i]||[]).map(ln=>{ const a=new Float32Array(ln.length*2);
    for(let k=0;k<ln.length;k++){ const sn=Math.sin(ln[k][1]*Math.PI/180); a[2*k]=(ln[k][0]+180)/360; a[2*k+1]=0.5-Math.log((1+sn)/(1-sn))/(4*Math.PI); }
    return a; });
  return (TSUW[i]=w);
}
function tsuAnchor(i){                                   // 観測地点 [lat,lon]: 海岸線があればその中ほど、無ければ代表点
  if(TSU_ANCH[i]) return TSU_ANCH[i];
  const a=sim.tsuAreas[i]; let r=[a[1],a[2]];
  const L=TSUGEO&&TSUGEO[i];
  if(L&&L.length){ let b=L[0]; for(const l of L) if(l.length>b.length) b=l; const m=b[Math.floor(b.length/2)]; r=[m[1],m[0]]; }
  return (TSU_ANCH[i]=r);
}
const tsuBarColor=h=>h>=3?TSU_COL[4]:h>=1?TSU_COL[3]:h>=0.2?TSU_COL[2]:TSU_COL[1];
const tsuBarPx=h=>h<=0?0:(h<=6?Math.max(3,h*13):78+Math.log2(h/6)*14);   // 6mまで13px/m、それ以上は対数で圧縮
function drawTsuBars(st,z){
  if(!st.obs.size) return;
  if(TSUGEO_RAW && !TSUGEO) buildTsuGeo();
  const bw=Math.max(7,Math.min(14,(z-3)*2.2));
  const items=[];
  for(const [i,o] of st.obs){
    if(!(o.mx>0)||!o.segs||!o.segs.length) continue;
    items.push([i,o,map.latLngToContainerPoint(tsuAnchor(i))]);
  }
  items.sort((p,q)=>p[2].y-q[2].y);                      // 画面の下（手前）の棒を上に重ねる
  ctx.save(); ctx.textAlign='center'; ctx.textBaseline='bottom'; ctx.font='bold 11px sans-serif';
  for(const [i,o,p] of items){
    const x=Math.round(p.x-bw/2), y0=p.y;
    let prev=0;
    for(const h of o.segs){                              // 観測のたびに1段ずつ積み上がる（棒全体の色は現在の最大波高の階級）
      if(h<=prev) continue;
      const ya=y0-tsuBarPx(prev), yb=y0-tsuBarPx(h);
      ctx.fillStyle=tsuBarColor(o.mx); ctx.fillRect(x,yb,bw,ya-yb);   // 段ごとではなく、棒全体を現在の最大波高の色にする
      ctx.strokeStyle='rgba(0,0,0,.85)'; ctx.lineWidth=1; ctx.strokeRect(x+.5,yb+.5,bw-1,Math.max(ya-yb-1,0));
      prev=h;
    }
    ctx.fillStyle='#fff'; ctx.beginPath(); ctx.arc(p.x,y0,2.5,0,6.2832); ctx.fill();
    ctx.strokeStyle='#000'; ctx.lineWidth=1; ctx.stroke();
    if(z>=5.5){
      const ty=y0-tsuBarPx(o.mx)-2, txt=o.mx.toFixed(1)+'m';
      ctx.lineWidth=3; ctx.strokeStyle='rgba(0,0,0,.85)'; ctx.strokeText(txt,p.x,ty);
      ctx.fillStyle='#fff'; ctx.fillText(txt,p.x,ty);
    }
  }
  ctx.restore();
}
function drawTsuCoast(st,z){
  if(!st.list.length) return;
  if(TSUGEO_RAW && !TSUGEO) buildTsuGeo();
  if(!TSUGEO){ drawTsu(st,z); return; }
  const rest=[];
  ctx.save(); ctx.lineCap='round'; ctx.lineJoin='round'; ctx.textBaseline='middle';
  const lw=Math.max(3,Math.min(7,(z-3)*1.2));
  const placed=[];
  for(const e of st.list){
    const lines=TSUGEO[e.i];
    if(!lines){ rest.push(e); continue; }
    let ce=TSUPC[e.i]; const vk=OX+','+OY+','+SC;
    if(!ce||ce.k!==vk){                                  // 地図が動いたときだけ投影し直す（以前は毎フレーム全頂点を変換していた）
      const paths=[]; let best=null, bestN=0;
      for(const a of tsuWorld(e.i)){
        const n=a.length>>1, pts=[]; let lx=-1e9, ly=-1e9;
        for(let k=0;k<n;k++){
          const x=OX+a[2*k]*SC, y=OY+a[2*k+1]*SC;
          if(k===0||k===n-1||Math.abs(x-lx)+Math.abs(y-ly)>=2){ pts.push({x:x,y:y}); lx=x; ly=y; }
        }
        if(pts.length<2) continue;
        paths.push(pts);
        if(n>bestN){ bestN=n; best=pts; }
      }
      ce=TSUPC[e.i]={k:vk,paths:paths,best:best};
    }
    const paths=ce.paths, best=ce.best;
    if(!paths.length) continue;
    ctx.beginPath();
    for(const pts of paths){ ctx.moveTo(pts[0].x,pts[0].y); for(let k=1;k<pts.length;k++) ctx.lineTo(pts[k].x,pts[k].y); }
    ctx.strokeStyle='rgba(0,0,0,.75)'; ctx.lineWidth=lw+3; ctx.stroke();
    ctx.strokeStyle=TSU_COL[e.lv]; ctx.lineWidth=lw; ctx.stroke();
    if(z>=5.5 && best){                                  // ラベルは最長の海岸線の中ほどに表示
      const o=st.obs.get(e.i), a=sim.tsuAreas[e.i], m=best[Math.floor(best.length/2)];
      const txt=(z>=7?a[0]+' ':'')+e.hl+(o&&o.mx>0?' ／観測'+o.mx.toFixed(1)+'m':'');
      ctx.font='bold 11px sans-serif'; const w=ctx.measureText(txt).width;
      let tx=m.x+8, ty=m.y;
      for(const q of placed){ if(Math.abs(q[0]-tx)<w+6 && Math.abs(q[1]-ty)<13) ty=q[1]+13; }   // ラベル同士の重なりを避ける
      placed.push([tx,ty]);
      ctx.textAlign='left'; ctx.lineWidth=3; ctx.strokeStyle='rgba(0,0,0,.85)'; ctx.strokeText(txt,tx,ty);
      ctx.fillStyle=o&&o.mx>0?'#9be7ff':TSU_COL[e.lv]; ctx.fillText(txt,tx,ty);
    }
  }
  ctx.restore();
  if(rest.length) drawTsu({list:rest,obs:st.obs},z);
}
function drawTsu(st,z){
  if(!st.list.length) return;
  const AR=sim.tsuAreas;
  ctx.save(); ctx.lineCap='round'; ctx.textAlign='left'; ctx.textBaseline='middle';
  for(const e of st.list){
    const a=AR[e.i], p=map.latLngToContainerPoint([a[1],a[2]]);
    const az=a[3]*Math.PI/180, th=(a[3]+90)*Math.PI/180, len=Math.max(10,Math.min(26,(z-2)*4));
    const ox=Math.sin(az)*7, oy=-Math.cos(az)*7, dx=Math.sin(th)*len, dy=-Math.cos(th)*len;
    ctx.beginPath(); ctx.moveTo(p.x+ox-dx,p.y+oy-dy); ctx.lineTo(p.x+ox+dx,p.y+oy+dy);
    ctx.strokeStyle='rgba(0,0,0,.75)'; ctx.lineWidth=9; ctx.stroke();
    ctx.strokeStyle=TSU_COL[e.lv]; ctx.lineWidth=5.5; ctx.stroke();
    if(z>=6){
      const o=st.obs.get(e.i);
      const txt=(z>=8?a[0]+' ':'')+e.hl+(o&&o.mx>0?' ／観測'+o.mx.toFixed(1)+'m':'');
      ctx.font='bold 11px sans-serif'; ctx.lineWidth=3; ctx.strokeStyle='rgba(0,0,0,.85)';
      const tx=p.x+ox*2.2+(Math.sin(az)>=0?4:-4-ctx.measureText(txt).width), ty=p.y+oy*2.2;
      ctx.strokeText(txt,tx,ty); ctx.fillStyle=o&&o.mx>0?'#9be7ff':TSU_COL[e.lv]; ctx.fillText(txt,tx,ty);
    }
  }
  ctx.restore();
}
let lastTsu=0;
function updateTsu(ts,st){
  const box=$('tsucard');
  if(!st.any){ box.style.display='none'; return; }
  box.style.display='block';
  if((ts||0)-lastTsu<200) return; lastTsu=ts||0;
  let top=0; for(const e of st.list) if(e.lv>top) top=e.lv;
  box.className='card tsucard tl'+top;
  let lab;
  if(top) lab=TSU_LV[top];
  else if(st.last&&st.last.kind==='cancel') lab='解除';
  else if(st.last&&st.last.kind==='none') lab='津波の心配なし';
  else lab='発表前';
  let h='<div class="th"><span class="tt">津波警報・注意報・津波情報</span><span class="tb" style="color:'+(top===2?'#ffd23b':'#fff')+'">'+lab+'</span>'
    +(st.last?'<span class="tm">第'+st.gno+'報</span>':'')+'</div>';
  if(st.last){
    const r=st.last, many=sim.quakes.length>1;
    // 発令・続報の文面は、全地震を合わせた現在の最高階級に合わせる（地震ごとの文面だと他の地震の警報と食い違うため）
    const txt=((r.kind==='issue'||r.kind==='update')&&top)?TSU_MSG[top]:r.text;
    h+='<div class="tm">発表 '+clk(r.t)+(many?'　地震'+(r.qi+1):'')+(r.giant?'　※巨大地震（M8を超える）':'　M'+r.mag.toFixed(1))+'</div>'
      +'<div class="tx">'+txt+'</div>';
  }else h+='<div class="tx" style="color:#93a0b8">地震発生から約3分で第1報を発表します</div>';
  if(st.list.length){
    h+='<div class="tw"><table><tr style="color:#93a0b8"><td>予報区</td><td></td><td>予想高さ</td><td>到達予想</td><td>観測</td></tr>';
    for(const e of st.list.slice(0,40)){
      const o=st.obs.get(e.i);
      const arr=e.imm?'ただちに来襲':clk(e.ts).slice(0,5);
      const dW=x=>(x&&x[3]<0)?'引き':'押し';
      const ob=o&&o.mx>0?(o.f1?'第1波'+dW(o.f1)+' '+clk(o.f1[0]).slice(0,5)+'／':'')+'最大'+o.mx.toFixed(1)+'m（第'+(o.mw?o.mw[2]:1)+'波'+dW(o.mw)+' '+clk(o.mw?o.mw[0]:o.mt).slice(0,5)+'）'+(o.fin?'':' 観測中'):'';
      h+='<tr><td>'+sim.tsuAreas[e.i][0]+'</td><td><span class="chip" style="background:'+TSU_COL[e.lv]+';color:'+(e.lv===2?'#111':'#fff')+'">'+TSU_LV[e.lv].replace('津波','')+'</span></td>'
        +'<td>'+e.hl+'</td><td>'+arr+'</td><td style="color:#9be7ff">'+ob+'</td></tr>';
    }
    h+='</table></div>';
  }
  eewSet(box,h);
}

/* ---------- 地震情報タブ（震度速報・地震情報）----------
   強震モニタとは独立した画面。シミュレーションの時刻(tcur)に合わせて、発表済みの情報だけを表示する。
   近くで同時に起きた地震はサーバー側で1つにまとめられている（sim.info.groups）。 */
const INFO_KIND={miu:'震度に関する情報（未入電観測点の更新）',sokuho:'震度速報',gensen:'震源に関する情報',gs:'震源・震度に関する情報',kakuchi:'各地の震度に関する情報',genkou:'顕著な地震の震源要素更新のお知らせ'};
const TSU_TXT=['この地震による津波の心配はありません。','この地震により、日本の沿岸では若干の海面変動があるかもしれませんが、被害の心配はありません。','津波警報等（大津波警報・津波警報あるいは津波注意報）を発表中です。','この地震による津波の影響については、現在調査中です。'];
let curTab='mon', autoTimer=null, autoRet=null, infoSig='', infoCnt=0, infoPrev=null, infoUnread=0, seekStamp=-1e9, lastInfo=0;
let imap=null, imLayer=null, imFitted=false, imState=null, AREAC=null, NAME2ID=null;
try{ if(localStorage.getItem('infoAuto')==='0') $('infoAuto').checked=false; }catch(_){}
$('infoAuto').addEventListener('change',()=>{ try{ localStorage.setItem('infoAuto',$('infoAuto').checked?'1':'0'); }catch(_){} });
function renderBadge(){ const b=$('tabBadge'); b.textContent=infoUnread; b.style.display=(infoUnread>0&&curTab!=='info')?'block':'none'; }
function showTab(t,auto){
  if(!auto){ clearTimeout(autoTimer); autoRet=null; }   // 手動で切り替えたら、自動で戻さない
  curTab=t;
  $('infoView').style.display=t==='info'?'flex':'none';
  $('tabMon').classList.toggle('on',t==='mon'); $('tabInfo').classList.toggle('on',t==='info');
  if(t==='info'){ infoUnread=0; ensureInfoMap(); }
  renderBadge();
}
/* ---------- 地震情報タブの地図 ---------- */
// 区域の代表点: 区域内の陸上観測点の平均位置に最も近い観測点（海上や他区域に落ちないようにする）
function areaCentroids(){
  if(AREAC||!AREA||!N) return AREAC;
  const sx={}, sy={}, c={};
  for(let i=0;i<N;i++){ if(SEA&&SEA[i]) continue; const a=AREA[i]; sx[a]=(sx[a]||0)+SLON[i]; sy[a]=(sy[a]||0)+SLAT[i]; c[a]=(c[a]||0)+1; }
  const best={};
  for(let i=0;i<N;i++){
    if(SEA&&SEA[i]) continue; const a=AREA[i];
    const d=Math.pow(SLON[i]-sx[a]/c[a],2)+Math.pow(SLAT[i]-sy[a]/c[a],2);
    if(!best[a]||d<best[a].d) best[a]={d:d,p:[SLAT[i],SLON[i]]};
  }
  AREAC={}; for(const a in best) AREAC[a]=best[a].p;
  NAME2ID={}; for(const k in ANAMES) NAME2ID[ANAMES[k]]=k;
  return AREAC;
}
function ensureInfoMap(){
  if(!imap){
    imap=L.map('imap',{zoomSnap:0.25,minZoom:4,maxZoom:11,worldCopyJump:false,zoomAnimation:false,markerZoomAnimation:false,fadeAnimation:false}).setView([36.3,137.6],5.5);
    L.layerGroup([esri('Canvas/World_Dark_Gray_Base',ATTR_BASE),esri('Canvas/World_Dark_Gray_Reference','')]).addTo(imap);
    imLayer=L.layerGroup().addTo(imap);
    let lg='<span>震度</span>'; for(let c=1;c<=9;c++) lg+=chipHtml2(c);
    $('imLegend').innerHTML=lg;
  }
  setTimeout(()=>{ imap.invalidateSize(); drawInfoMap(); },0);
}
function chipHtml2(c){ return '<span class="chip" style="background:'+clsColor(c)+';color:'+((c===2||c===3)?'#111':'#fff')+'">'+CLASS_LABEL[c]+'</span>'; }
window.addEventListener('resize',()=>{ if(imap&&curTab==='info') imap.invalidateSize(); });
// 公開済みの情報から、地図と表に出す内容（区域ごとの震度・震源）をまとめる
function infoState(vis){
  const gs=sim.info.groups, areas=new Map(), epis=[];
  gs.forEach((G,gi)=>{
    const rel=G.releases.slice(0,vis[gi]); if(!rel.length) return;
    let ms=null, ar=null;
    for(let k=rel.length-1;k>=0;k--){
      if(!ms&&rel[k].mem) ms=rel[k].mem;
      if(!ar&&rel[k].areas) ar=rel[k];
      if(ms&&ar) break;
    }
    (ms||[]).forEach(m=>epis.push({m:m,gi:gi}));
    if(ar) ar.areas.forEach(a=>{
      const o=areas.get(a[0]), n=a.length>3?a[3]:null;
      if(!o||a[1]>o.c||(a[1]===o.c&&a[2]>o.v)) areas.set(a[0],{c:a[1],v:a[2],n:n});
    });
  });
  return {areas:areas,epis:epis};
}
function tableHtml(st){
  const rows=Array.from(st.areas.entries()).sort((a,b)=>b[1].c-a[1].c||b[1].v-a[1].v||(a[0]<b[0]?-1:1));
  if(!rows.length) return '';
  const hasN=rows.some(r=>r[1].n!=null);
  return '<div class="igc"><div class="igh"><span class="gt">各地の震度（区域別）</span><span class="gs">'+rows.length+'区域（最新の発表内容）</span></div>'
    +'<div class="itw"><table class="itbl"><tr><th>震度</th><th>区域</th><th class="num">計測震度</th>'+(hasN?'<th class="num">観測点数</th>':'')+'</tr>'
    +rows.map(r=>'<tr><td>'+chipHtml(r[1].c)+'</td><td>'+escH(r[0])+'</td><td class="num">'+fx1(r[1].v)+'</td>'
      +(hasN?'<td class="num">'+(r[1].n!=null?r[1].n:'')+'</td>':'')+'</tr>').join('')
    +'</table></div></div>';
}
function drawInfoMap(){
  if(!imap) return;
  imLayer.clearLayers();
  const st=imState; if(!st||!areaCentroids()) return;
  const pts=[];
  Array.from(st.areas.entries()).sort((a,b)=>a[1].c-b[1].c).forEach(r=>{
    const nm=r[0], o=r[1], id=NAME2ID[nm], c=id!=null?AREAC[id]:null; if(!c) return;
    const ic=L.divIcon({className:'',html:'<div class="imk" style="background:'+clsColor(o.c)+';color:'+((o.c===2||o.c===3)?'#111':'#fff')+'">'+CLASS_LABEL[o.c]+'</div>',iconSize:[o.c>=5?30:24,18],iconAnchor:[o.c>=5?15:12,9]});
    L.marker(c,{icon:ic,keyboard:false,zIndexOffset:o.c*10}).bindTooltip(escH(nm)+'　震度'+CLASS_LABEL[o.c]+(o.n!=null?'（'+o.n+'点）':'')).addTo(imLayer);
    pts.push(c);
  });
  st.epis.forEach(e=>{
    const m=e.m;
    L.marker([m.lat,m.lon],{icon:epiIcon(e.gi),keyboard:false,zIndexOffset:1000})
      .bindTooltip(escH(m.epi)+'　'+magTxt(m)+'　深さ'+(m.depth<1?'ごく浅い':'約'+m.depth+'km')).addTo(imLayer);
    pts.push([m.lat,m.lon]);
  });
  if(!imFitted&&pts.length){ imFitted=true; imap.fitBounds(L.latLngBounds(pts).pad(0.25),{maxZoom:8,animate:false}); }
}
$('tabMon').onclick=()=>showTab('mon',false);
$('tabInfo').onclick=()=>showTab('info',false);
// 情報が更新されたら、強震モニタ表示中なら数秒だけ地震情報タブへ切り替えて元に戻す
function autoSwitch(){
  if(!$('infoAuto').checked) return;
  if(curTab==='info' && autoRet==null) return;          // 手動で情報タブを見ているときは切り替えない
  if(curTab==='mon'){ autoRet='mon'; showTab('info',true); }
  clearTimeout(autoTimer);                              // 続けて更新があれば、戻るのを延ばす
  autoTimer=setTimeout(()=>{ if(autoRet){ const r=autoRet; autoRet=null; showTab(r,true); } },5000);
}
$('iRun').onclick=()=>$('runFab').click();
$('iPlay').onclick=()=>$('play').click();
$('iRestart').onclick=()=>$('restart').click();
$('restart').addEventListener('click',()=>{ seekStamp=performance.now(); });
$('seek').addEventListener('input',()=>{ seekStamp=performance.now(); });
const escH=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function chipHtml(c){ return '<span class="chip" style="background:'+clsColor(c)+';color:'+((c===2||c===3)?'#111':'#fff')+'">震度'+CLASS_LABEL[c]+'</span>'; }
function jdt(t){ const d=new Date(simClock0+t*1000); return d.getDate()+'日'+d.getHours()+'時'+pad2(d.getMinutes())+'分'; }
function jfull(t){ const d=new Date(simClock0+t*1000); return (d.getMonth()+1)+'月'+d.getDate()+'日 '+d.getHours()+'時'+pad2(d.getMinutes())+'分'; }
function llTxt(m){ return (m.lat>=0?'北緯':'南緯')+Math.abs(m.lat).toFixed(1)+'度、'+(m.lon>=0?'東経':'西経')+Math.abs(m.lon).toFixed(1)+'度'; }
function magTxt(m){ return m.gst?'M8以上の巨大地震と推定':(m.ge?'M'+m.mag.toFixed(1)+'程度以上':'M'+m.mag.toFixed(1)); }
// 気象庁の発表文に近い形: 「○日○時○分ころ、地震がありました。」＋震源地（緯度経度）・深さ・規模
function memHtml(ms){
  return ms.map(m=>'<div class="bl">'+jdt(m.delay)+'ころ、地震がありました。</div>'
    +'<div class="im">震源地は、<b>'+escH(m.epi)+'</b>（'+llTxt(m)+'）で、震源の深さは'+(m.depth<1?'ごく浅く':'約'+m.depth+'km')
    +'、地震の規模（マグニチュード）は<b>'+(m.gst?'M8以上の巨大地震':(m.ge?(m.mag.toFixed(1)+'程度以上'):m.mag.toFixed(1)))+'</b>と推定されます。</div>').join('');
}
// 震度別（震度の大きい順）に区域名を並べる（震度速報の形式）
function areasHtml(areas){
  const by={}; areas.forEach(a=>{ (by[a[1]]=by[a[1]]||[]).push(a); });
  return Object.keys(by).map(Number).sort((p,q)=>q-p).map(c=>'<div class="ic">'+chipHtml(c)+'<div class="ia">'
    +by[c].map(a=>escH(a[0])).join('、')+'</div></div>').join('');
}
// 震度別 → 都道府県別 → 区域（各地の震度に関する情報の形式。detail=true で計測震度・観測点数も付ける）
function areasByPref(areas,detail){
  const by={}; areas.forEach(a=>{ (by[a[1]]=by[a[1]]||[]).push(a); });
  return Object.keys(by).map(Number).sort((p,q)=>q-p).map(c=>{
    const pf=new Map();
    by[c].forEach(a=>{ const p=a[4]||'その他'; if(!pf.has(p)) pf.set(p,[]); pf.get(p).push(a); });
    let h='';
    pf.forEach((list,p)=>{
      h+='<span class="pf"><b>'+escH(p)+'</b> '+list.map(a=>{
        const nm=(a[4]&&a[0].indexOf(a[4])===0&&a[0].length>a[4].length)?a[0].slice(a[4].length):a[0];
        return escH(nm)+(detail&&a[3]!=null?' <i>('+fx1(a[2])+'／'+a[3]+'点)</i>':'');
      }).join('、')+'</span>';
    });
    return '<div class="ic">'+chipHtml(c)+'<div class="ia">'+h+'</div></div>';
  }).join('');
}
// 未入電の観測点を震度情報に反映する: 発表時点で未入電の区域は、入電している観測点だけで震度を出し直し、
// 数時間後（2.5〜4時間）に「未入電だった観測点の震度」を発表して元の値に更新する
function muInfoPatch(v){
  if(!v.info||!MUF||!muOn()) return;
  const dt=v.dt, nf=v.nf, fl=[]; for(let i=0;i<N;i++) if(MUF[i]>=0) fl.push(i);
  if(!fl.length) return;
  const n2id={}; for(const k in ANAMES) n2id[ANAMES[k]]=parseInt(k);
  const aset=new Set(fl.map(i=>AREA[i])), AX={}, byA={};
  for(let i=0;i<N;i++){ if(SEA&&SEA[i]) continue; const a=AREA[i]; if(aset.has(a)) (byA[a]=byA[a]||[]).push(i); }
  aset.forEach(a=>{ const arr=new Int16Array(nf), st=byA[a]||[]; let cm=-999;
    for(let f=0;f<nf;f++){ const o=f*N; for(const i of st){ if(MUF[i]>=0&&f>=MUF[i]) continue; const x=Q[o+i]; if(x>cm) cm=x; } arr[f]=cm; }
    AX[a]=arr; });
  const fr=t=>{ const f=Math.round(t/dt); return f<0?0:(f>nf-1?nf-1:f); };
  const gs=v.info.groups, grp=new Map(), recT=new Map();
  fl.forEach(i=>{ const tf=MUF[i]*dt; let gi=0; gs.forEach((G,k)=>{ if(G.t0<=tf+1) gi=k; }); grp.set(i,gi);
    recT.set(gi,Math.max(recT.get(gi)||0,tf+9000+Math.random()*5400)); });
  gs.forEach((G,gi)=>{
    G.releases.forEach(r=>{
      const f=fr(r.t), act=fl.filter(i=>grp.get(i)===gi&&f>=MUF[i]);
      if(!act.length) return;
      if(r.areas){
        r._a0=r.areas;
        const as=new Set(act.map(i=>AREA[i])), minc=r.k==='sokuho'?3:(r.minc||1), rows=[];
        r.areas.forEach(row=>{ const id=n2id[row[0]];
          if(id!=null&&as.has(id)){ const nv=AX[id][f]*0.1, c=nv<0.5?0:shindoCls(nv); if(c<minc) return;
            row=row.slice(); row[1]=c; row[2]=Math.floor(nv*10)/10; }
          rows.push(row); });
        rows.sort((p,q)=>q[2]-p[2]); r.areas=rows;
      }
      if(r.mx){ const m2=CUMV2[f]*0.1, c=m2<0.5?0:shindoCls(m2); r.mx=[c,CLASS_LABEL[c]]; }
      r.mu=Array.from(new Set(act.map(i=>ANAMES[AREA[i]]||('区域'+AREA[i]))));
    });
    const tr=recT.get(gi); if(!tr) return;
    const best=new Map();
    fl.filter(i=>grp.get(i)===gi).forEach(i=>{ const nm=ANAMES[AREA[i]]||('区域'+AREA[i]), x=MUPK[i]; if(!best.has(nm)||x>best.get(nm)) best.set(nm,x); });
    const rec=Array.from(best.entries()).map(e=>[e[0],shindoCls(e[1]*0.1),Math.floor(e[1])/10]).sort((p,q)=>q[2]-p[2]);
    let lastK=null; for(let k=G.releases.length-1;k>=0;k--){ if(G.releases[k].k==='kakuchi'){ lastK=G.releases[k]; break; } }
    const c=CUMV[fr(tr)]*0.1, cc=c<0.5?0:shindoCls(c);
    G.releases.forEach(r=>{ if(r.t>=tr) { r.mu=null; } });
    G.releases.push({t:Math.round(tr*10)/10,k:'miu',upd:true,mx:[cc,CLASS_LABEL[cc]],mem:G.members,rec:rec,areas:lastK?(lastK._a0||lastK.areas):[]});
    G.releases.sort((p,q)=>p.t-q.t);
    v.info.end=Math.max(v.info.end,tr+20);
  });
}
function relHtml(r,isNew,t0){
  let b='';
  const mxLine=r.mx&&r.mx[0]>0?'<div class="im">最大震度'+CLASS_LABEL[r.mx[0]]+'を観測しています。</div>':'';
  const tsu=r.tsu!=null?'<div class="itsu">'+TSU_TXT[r.tsu]+'</div>':'';
  if(r.k==='sokuho') b='<div class="bl">'+jdt(t0)+'ころ、地震がありました。</div>'+areasHtml(r.areas)
    +'<div class="inote">震度3以上を観測した地域をお知らせします。今後の情報にご注意ください。</div>';
  else if(r.k==='gensen') b=memHtml(r.mem)+tsu;
  else if(r.k==='gs') b=memHtml(r.mem)+mxLine+tsu+'<div class="inote">'+(r.minc>=3?'震度3以上':'震度1以上')+'を観測した地域</div>'+areasByPref(r.areas,false);
  else if(r.k==='genkou'){
    const ch=r.mem.map((m,i)=>{ const o=r.old&&r.old[i]; if(o==null) return ''; if(o===-1) return 'マグニチュードを「M8以上の巨大地震」から'+m.mag.toFixed(1)+'に更新します。';
      return Math.abs(o-m.mag)>=0.05?'マグニチュードを'+o.toFixed(1)+'から'+m.mag.toFixed(1)+'に更新します。':''; }).filter(x=>x).join('');
    b=memHtml(r.mem)+'<div class="im">'+(ch||'震源要素を精査しました。マグニチュード・震源の位置に変更はありません。')+'</div>'+mxLine+tsu;
  }
  else if(r.k==='miu') b=memHtml(r.mem)+'<div class="im">震度5弱以上未入電となっていた観測点の震度を確認しました。</div>'
    +(r.rec||[]).map(e=>'<div class="ic">'+chipHtml(e[1])+'<div class="ia">'+escH(e[0])+'（計測震度 '+fx1(e[2])+'）</div></div>').join('')+mxLine;
  else b=memHtml(r.mem)+mxLine+'<div class="inote">震度1以上を観測した地域（計測震度／観測点数）</div>'+areasByPref(r.areas,true);
  if(r.mu&&r.mu.length&&r.k!=='miu') b+='<div class="inote" style="color:#ffb54a">震度5弱以上未入電の観測点があります：'+r.mu.map(escH).join('、')+'</div>';
  return '<div class="ir'+(isNew?' new':'')+'"><div class="rh"><b>'+INFO_KIND[r.k]+'</b>'+(r.upd?'<span class="ru">更新</span>':'')
    +'<span class="rt">'+jfull(r.t)+' 気象庁発表</span></div>'+b+'</div>';
}
function groupHtml(gi,n,prevN){
  const G=sim.info.groups[gi], rel=G.releases.slice(0,n), last=rel[rel.length-1];
  let ms=null; for(let k=rel.length-1;k>=0;k--) if(rel[k].mem){ ms=rel[k].mem; break; }
  let sub='震源の情報を待っています…';
  if(ms){ const mm=ms.reduce((p,q)=>q.mag>p.mag?q:p); sub=escH(mm.epi)+'　M'+mm.mag.toFixed(1)+(ms.length>1?'　ほか'+(ms.length-1)+'件':''); }
  const many=sim.info.groups.length>1;
  let h='<div class="igc"><div class="igh"><span class="gt">地震'+(many?' '+(gi+1):'')+'</span><span class="gs">発生 '+clk(G.t0)+'頃　'+sub
    +(G.members.length>1?'<br><span class="gm">近接して発生した'+G.members.length+'つの地震を、一つの地震としてまとめて発表しています</span>':'')+'</span>'
    +(last.mx&&last.mx[0]>0?chipHtml(last.mx[0]):'')+'</div>';
  for(let k=rel.length-1;k>=0;k--) h+=relHtml(rel[k],prevN!=null&&k>=prevN,G.t0);
  return h+'</div>';
}
function updateInfo(ts){
  if(curTab==='info'){
    $('iPlay').textContent=playing?'❚❚':'►';
    $('iTime').textContent=sim?clk(tcur)+'（経過 '+fmtT(tcur)+'）':'—';
  }
  if((ts||0)-lastInfo<150) return; lastInfo=ts||0;
  const box=$('infoBody');
  if(!sim||!sim.info){
    if(infoSig!=='empty'){ box.innerHTML='<div class="iempty">強震モニタ画面でシミュレーションを実行すると、ここに震度速報・地震情報が発表されます。<br>（実行ボタンはこの画面の上にもあります）</div>'; infoSig='empty'; imState=null; drawInfoMap(); }
    return;
  }
  const gs=sim.info.groups;
  const vis=gs.map(g=>{ let n=0; for(const r of g.releases){ if(r.t>tcur) break; n++; } return n; });
  const total=vis.reduce((a,b)=>a+b,0), grew=total>infoCnt;
  if(grew && playing && ts-seekStamp>800){                // 再生中に新しい情報が発表された（シークや巻き戻しは除く）
    if(curTab!=='info') infoUnread+=total-infoCnt;
    renderBadge(); autoSwitch();
  }
  infoCnt=total;
  const sig=vis.join(',');
  if(sig===infoSig) return;
  infoSig=sig;
  const prev=(grew&&infoPrev&&infoPrev.length===vis.length)?infoPrev:null;
  const order=gs.map((g,i)=>i).filter(i=>vis[i]>0).sort((a,b)=>gs[b].t0-gs[a].t0);   // 新しい地震を上に
  imState=infoState(vis);
  box.innerHTML='<div class="wrap">'+tableHtml(imState)+(order.length?order.map(i=>groupHtml(i,vis[i],prev?prev[i]:null)).join('')
    :'<div class="iempty">まだ情報は発表されていません。</div>')+'</div>';
  infoPrev=vis.slice();
  drawInfoMap();
}

/* ---------- 効果音・読み上げ ----------
   場面の構成は「リアルタイム地震ビューアー」(kotoho7様作)を参考にしているが、音は Web Audio で新規に合成したもの。
   読み上げはブラウザ標準の日本語音声。再生中(playing)に tcur が進んだ区間で発表された情報だけを鳴らす。 */
let aCtx=null, aMaster=null, aPrev=-0.001, aMaxCls=-1, aObs=new Map(), aBusyUntil=0, aBusyPrio=0, aVoice=null, aHintOn=false;
const aOn=()=>$('sndOn').checked, aSp=()=>$('spkOn').checked, aVol=()=>{ const v=parseFloat($('sndVol').value); return isFinite(v)?v:0; };
function aSave(){ try{ localStorage.setItem('kmAudio',JSON.stringify({s:aOn(),k:aSp(),v:aVol()})); }catch(e){} }
try{ const sv=JSON.parse(localStorage.getItem('kmAudio')||'null');
  if(sv){ $('sndOn').checked=!!sv.s; $('spkOn').checked=!!sv.k; if(isFinite(sv.v)) $('sndVol').value=sv.v; } }catch(e){}
function aHint(on){ if(aHintOn===on) return; aHintOn=on; $('audioHint').style.display=on?'block':'none'; }
// ブラウザは「タップ/クリックの瞬間」でないと音を出せないため、操作のたびに有効化を試みる（スマホ対応）
function audioUnlock(){
  try{
    if(navigator.audioSession) navigator.audioSession.type='playback';    // iPhone: マナーモードでも効果音を鳴らす
    if(!aCtx){ const AC=window.AudioContext||window.webkitAudioContext; if(!AC) return;
      aCtx=new AC(); aMaster=aCtx.createGain();
      const cp=aCtx.createDynamicsCompressor(); aMaster.connect(cp); cp.connect(aCtx.destination);   // 重なっても音割れしにくくする
      aCtx.onstatechange=()=>{ if(aCtx.state==='running') aHint(false); }; }
    aMaster.gain.value=1.4*aVol();
    aPrimeSpeech();
    if(aCtx.state!=='running'){
      const pr=aCtx.resume(); if(pr&&pr.then) pr.then(()=>{ if(aCtx.state==='running') aHint(false); }).catch(()=>{});
      const b=aCtx.createBuffer(1,1,22050), s=aCtx.createBufferSource(); s.buffer=b; s.connect(aCtx.destination); s.start(0);   // iOS用: 無音を鳴らして出力を開始
    }
  }catch(e){}
}
['pointerdown','pointerup','touchstart','touchend','click','keydown'].forEach(ev=>document.addEventListener(ev,audioUnlock,{passive:true}));
$('audioHint').addEventListener('click',audioUnlock);
$('sndVol').addEventListener('input',()=>{ if(aMaster) aMaster.gain.value=1.4*aVol(); aSave(); });
$('sndOn').addEventListener('change',aSave);
$('spkOn').addEventListener('change',()=>{ aSave(); if(!aSp()){ if(window.speechSynthesis) window.speechSynthesis.cancel(); } else audioUnlock(); });
// 起動時に音声のON/OFFを選ばせる（選択はブラウザに保存し、次回からは聞かない）
(function(){
  let asked=false; try{ asked=localStorage.getItem('kmAudioAsked')==='1'; }catch(e){}
  const dlg=$('audioAsk');
  if(asked){ dlg.style.display='none'; return; }
  $('sndOn').checked=false; $('spkOn').checked=false;          // 選ぶまでは音声なし
  const pick=on=>{ $('sndOn').checked=on; $('spkOn').checked=on; dlg.style.display='none';
    try{ localStorage.setItem('kmAudioAsked','1'); }catch(e){} aSave(); audioUnlock(); };
  $('askOn').onclick=()=>pick(true); $('askOff').onclick=()=>pick(false);
})();
function aTone(f0,t0,d,type,vol,f1){
  const o=aCtx.createOscillator(), g=aCtx.createGain(), t=aCtx.currentTime+t0;
  o.type=type||'sine'; o.frequency.setValueAtTime(f0,t);
  if(f1) o.frequency.linearRampToValueAtTime(f1,t+d);
  g.gain.setValueAtTime(0.0001,t); g.gain.exponentialRampToValueAtTime(vol,t+0.012);
  g.gain.exponentialRampToValueAtTime(0.0001,t+d);
  o.connect(g); g.connect(aMaster); o.start(t); o.stop(t+d+0.03);
}
function aBell(f,t0,d,vol){ aTone(f,t0,d,'sine',vol); aTone(f*2.76,t0,d*0.5,'sine',vol*0.35); aTone(f*5.4,t0,d*0.25,'sine',vol*0.15); }
// 各効果音は鳴らす長さ[秒]を返す
// 少しずらした2本ののこぎり波を重ねて「うなり」を出し、緊迫感のある硬い音にする
function aHarsh(f,t0,d,vol,f1){ aTone(f,t0,d,'sawtooth',vol,f1); aTone(f*1.013,t0,d,'sawtooth',vol,f1?f1*1.013:0); }
const SFX={
  // 緊急地震速報（予報）: 高い2音を素早く3回
  eewFcst(){ for(let k=0;k<3;k++){ aHarsh(1397,k*0.34,0.1,0.13); aHarsh(1047,k*0.34+0.13,0.1,0.13); } return 1.1; },
  // 緊急地震速報（警報）: 不協和な2音の高速連打＋低音の脈動を4セット
  eewWarn(){ for(let k=0;k<4;k++){ const b=k*0.9;
      for(let j=0;j<5;j++) aHarsh(j%2?1245:1760,b+j*0.1,0.085,0.15);
      aTone(98,b,0.7,'sawtooth',0.2,87); } return 3.6; },
  // 予想震度の更新: 短い3連
  eewUpd(){ for(let k=0;k<3;k++) aHarsh(2093,k*0.09,0.06,0.12); return 0.35; },
  // 震度速報・地震情報: 低めの短調の下降音＋重い打音
  infoNew(){ aHarsh(784,0,0.22,0.13); aHarsh(622,0.24,0.22,0.13); aHarsh(523,0.48,0.5,0.13); aTone(65,0,0.6,'sine',0.35,45); return 1.1; },
  infoUpd(){ aHarsh(622,0,0.16,0.12); aHarsh(784,0.18,0.3,0.12); aTone(65,0,0.4,'sine',0.3,45); return 0.6; },
  // 最大震度の更新: 震度が大きいほど、速く・多く・高く（5弱以上は低音付き）
  newInt(c){ const n=Math.min(2+c,8), gap=c>=5?0.075:0.1, base=440*Math.pow(2,Math.min(c,9)/8);
    for(let k=0;k<n;k++) aHarsh(base*(k%2?1.26:1),k*gap,gap*0.8,c>=5?0.15:0.11);
    if(c>=5) aTone(90,0,n*gap,'sawtooth',0.2,70); return n*gap+0.1; },
  // 津波情報（予報）: 低いゴング
  tsuInfo(){ aBell(330,0,1.4,0.35); aBell(247,0.6,1.6,0.35); return 2.0; },
  // 津波注意報: 緊張感のある2音サイレン
  tsuAdv(){ for(let k=0;k<3;k++){ aHarsh(740,k*0.8,0.38,0.14); aHarsh(554,k*0.8+0.4,0.38,0.14); } return 2.4; },
  // 津波警報: 速い上下サイレン
  tsuWarn(){ for(let k=0;k<5;k++){ aHarsh(500,k*0.7,0.35,0.15,1000); aHarsh(1000,k*0.7+0.35,0.35,0.15,500); }
    aTone(82,0,3.5,'sawtooth',0.2); return 3.5; },
  // 大津波警報: さらに速い高音サイレン＋重低音
  tsuMajor(){ for(let k=0;k<10;k++){ aHarsh(600,k*0.4,0.2,0.15,1500); aHarsh(1500,k*0.4+0.2,0.2,0.15,600); }
    aTone(65,0,4,'square',0.2,55); return 4.0; },
  // 津波観測: 重い鐘を2回
  tsuObs(){ aBell(392,0,1.4,0.4); aBell(311,0.5,1.6,0.4); aTone(65,0,0.8,'sine',0.3,50); return 2.0; },
  tsuClear(){ aBell(659,0,1.2,0.35); aBell(523,0.5,1.2,0.35); aBell(392,1.0,1.8,0.35); return 2.8; },   // 津波警報等の解除・切替（発表時とは別の音）
  cancel(){ aHarsh(523,0,0.25,0.11); aHarsh(415,0.28,0.25,0.11); aHarsh(330,0.56,0.45,0.11); return 1.0; }
};
function audioPlay(name,prio,arg){
  if(!aOn()||!SFX[name]) return;
  if(!aCtx) audioUnlock();
  if(!aCtx) return;
  if(aCtx.state!=='running'){ try{ aCtx.resume(); }catch(e){} aHint(true); return; }   // 音を出せない間は溜めずに捨て、タップを促す
  const now=aCtx.currentTime;
  if(now<aBusyUntil && prio<aBusyPrio) return;          // もっと重要な音が鳴っている間は鳴らさない
  aBusyPrio=prio; aBusyUntil=now+SFX[name](arg);
}
// 読み上げ。keep=true は話している最中でも順番待ちにする。長い文は「。」で区切って順に話す（途中で切れるのを防ぐ）
// 読み上げの読み間違い対策: 読みを間違えやすい語を、ひらがなに置き換えてから読み上げる（長い語から順に置換）
const YOMI=[
 ['大津波警報','おおつなみけいほう'],['三八上北','さんぱちかみきた'],['九十九里','くじゅうくり'],['外房','そとぼう'],['内房','うちぼう'],
 ['東京都23区','とうきょうとにじゅうさんく'],['23区','にじゅうさんく'],
 ['石狩','いしかり'],['空知','そらち'],['後志','しりべし'],['胆振','いぶり'],['日高','ひだか'],['渡島','おしま'],['檜山','ひやま'],['上川','かみかわ'],
 ['留萌','るもい'],['宗谷','そうや'],['網走','あばしり'],['十勝','とかち'],['釧路','くしろ'],['根室','ねむろ'],['内浦湾','うちうらわん'],['噴火湾','ふんかわん'],
 ['色丹島','しこたんとう'],['択捉島','えとろふとう'],['国後島','くなしりとう'],
 ['津軽','つがる'],['下北','しもきた'],['庄内','しょうない'],['村山','むらやま'],['最上','もがみ'],['置賜','おきたま'],['浜通り','はまどおり'],['中通り','なかどおり'],['会津','あいづ'],
 ['三陸','さんりく'],['陸奥湾','むつわん'],['房総','ぼうそう'],['茨城','いばらき'],['栃木','とちぎ'],['群馬','ぐんま'],['埼玉','さいたま'],['神奈川','かながわ'],
 ['多摩','たま'],['富士五湖','ふじごこ'],['三浦半島','みうらはんとう'],['相模湾','さがみわん'],['伊豆諸島','いずしょとう'],['伊豆','いず'],['小笠原','おがさわら'],['駿河湾','するがわん'],['遠州灘','えんしゅうなだ'],
 ['上越','じょうえつ'],['中越','ちゅうえつ'],['下越','かえつ'],['佐渡','さど'],['能登','のと'],['加賀','かが'],['嶺北','れいほく'],['嶺南','れいなん'],['飛騨','ひだ'],['美濃','みの'],
 ['伊勢','いせ'],['三河','みかわ'],['若狭','わかさ'],['淡路島','あわじしま'],['丹後','たんご'],['但馬','たじま'],['和歌山','わかやま'],['兵庫','ひょうご'],['熊野灘','くまのなだ'],['紀伊','きい'],
 ['鳥取','とっとり'],['島根','しまね'],['隠岐','おき'],['出雲','いずも'],['石見','いわみ'],['瀬戸内','せとうち'],['愛媛','えひめ'],['東予','とうよ'],['中予','ちゅうよ'],['南予','なんよ'],
 ['宇和海','うわかい'],['豊後水道','ぶんごすいどう'],['豊後','ぶんご'],['土佐湾','とさわん'],['伊予灘','いよなだ'],['安芸灘','あきなだ'],['周防灘','すおうなだ'],['響灘','ひびきなだ'],
 ['筑豊','ちくほう'],['筑後','ちくご'],['北九州','きたきゅうしゅう'],['有明','ありあけ'],['島原半島','しまばらはんとう'],['壱岐','いき'],['対馬','つしま'],['五島','ごとう'],
 ['阿蘇','あそ'],['球磨','くま'],['天草灘','あまくさなだ'],['天草','あまくさ'],['芦北','あしきた'],['日向灘','ひゅうがなだ'],['薩摩','さつま'],['大隅','おおすみ'],['十島村','としまむら'],
 ['甑島','こしきしま'],['種子島','たねがしま'],['屋久島','やくしま'],['奄美','あまみ'],['錦江湾','きんこうわん'],
 ['久米島','くめじま'],['大東島','だいとうじま'],['宮古島','みやこじま'],['石垣島','いしがきじま'],['与那国島','よなぐにじま'],['西表島','いりおもてじま'],['八重山','やえやま'],['本島','ほんとう'],
 ['北東部','ほくとうぶ'],['北西部','ほくせいぶ'],['南東部','なんとうぶ'],['南西部','なんせいぶ'],['北中部','ほくちゅうぶ'],['中北部','ちゅうほくぶ'],['中南部','ちゅうなんぶ'],['中東部','ちゅうとうぶ'],['中西部','ちゅうせいぶ'],
 ['北部','ほくぶ'],['南部','なんぶ'],['東部','とうぶ'],['西部','せいぶ'],['中部','ちゅうぶ'],['山沿い','やまぞい'],['平野部','へいやぶ'],['沿岸','えんがん'],['内陸','ないりく'],['近海','きんかい']
].sort((a,b)=>b[0].length-a[0].length);
function aYomi(t){ let s=String(t); for(const p of YOMI) if(s.indexOf(p[0])>=0) s=s.split(p[0]).join(p[1]); return s; }
const aUtt=[]; let aSpeakAt=0, aPrimed=false;    // aUtt: 話している途中でブラウザに破棄されないよう参照を保持する
function aPrimeSpeech(){            // iPhone等: 最初の操作の中で一度だけ無音の読み上げを行い、以後の自動読み上げを許可させる
  if(aPrimed||!aSp()||!window.speechSynthesis) return; aPrimed=true;
  try{ const u=new SpeechSynthesisUtterance(' '); u.volume=0; u.lang='ja-JP'; window.speechSynthesis.speak(u); }catch(e){}
}
// 読み上げ。keep=true は話している最中でも順番待ちにする。長い文は「。」で区切って順に話す（途中で切れるのを防ぐ）
function audioSpeak(text,prio,keep){
  if(!aSp()||!window.speechSynthesis||!text) return;
  try{
    const ss=window.speechSynthesis;
    let busy=ss.speaking||ss.pending;
    if(busy&&Date.now()-aSpeakAt>180000){ ss.cancel(); aUtt.length=0; busy=false; }   // 固まったときの復帰
    if(!aVoice) aVoice=(ss.getVoices()||[]).find(v=>/^ja/i.test(v.lang))||null;
    const parts=[]; let cur='';
    aYomi(text).split('。').forEach(s=>{ if(!s) return; s+='。'; if(cur&&(cur+s).length>90){ parts.push(cur); cur=''; } cur+=s; });
    if(cur) parts.push(cur);
    const go=()=>{
      try{ ss.resume(); }catch(e){}
      parts.forEach(t=>{ const u=new SpeechSynthesisUtterance(t); u.lang='ja-JP'; u.rate=1.05; u.volume=Math.min(1,aVol()*1.2);
        if(aVoice) u.voice=aVoice;
        u.onend=u.onerror=()=>{ const i=aUtt.indexOf(u); if(i>=0) aUtt.splice(i,1); };
        aUtt.push(u); ss.speak(u); });
      aSpeakAt=Date.now();
    };
    if(prio>=90){ ss.cancel(); aUtt.length=0; setTimeout(go,80); }     // cancel 直後の speak は無視されることがあるため少し待つ
    else if(busy&&!keep) return;
    else go();
  }catch(e){}
}
// 最大震度が上がったとき、その震度を観測した観測点を数秒間、光が広がるように強調する
function aFx(t,c){
  if(!sim||!CUMI||CUMI.length!==sim.nf) return;
  let f=Math.round(Math.min(t,sim.tSeis)/sim.dt); if(f<0) f=0; if(f>sim.nf-1) f=sim.nf-1;
  const arr=(muOn()&&CUMI2&&CUMI2.length===sim.nf)?CUMI2:CUMI, i=arr[f];
  if(i>=0){ FX.push({i:i,c:c,t0:performance.now()}); if(FX.length>8) FX.shift(); }
}
function aMaxClsAt(t){
  if(!sim||!CUMV||CUMV.length!==sim.nf) return -1;
  let f=Math.round(Math.min(t,sim.tSeis)/sim.dt); if(f<0) f=0; if(f>sim.nf-1) f=sim.nf-1;
  const m=((muOn()&&CUMV2&&CUMV2.length===sim.nf)?CUMV2:CUMV)[f]*0.1; return m<0.5?0:shindoCls(m);
}
function aMxCls(r){ if(r.mx&&r.mx[0]>0) return r.mx[0]; let m=0; (r.areas||[]).forEach(a=>{ if(a[1]>m) m=a[1]; }); return m; }
const SPK_CLS=['0','1','2','3','4','ごじゃく','ごきょう','ろくじゃく','ろっきょう','7'];     // 震度の読み（5弱=ごじゃく など）
const aSh=c=>SPK_CLS[c]!==undefined?SPK_CLS[c]:String(c);
function aClockTxt(t){ const d=new Date(simClock0+t*1000); return d.getHours()+'時'+d.getMinutes()+'分'; }
function aList(a,cap){ return a.length>cap?a.slice(0,cap).join('、')+'ほか':a.join('、'); }
function aMainMem(r){ const ms=r.mem; return (ms&&ms.length)?ms.reduce((p,q)=>q.mag>p.mag?q:p):null; }
function aMagTxt(m){ return m.gst?'8以上の巨大地震と推定':m.mag.toFixed(1)+(m.ge?'程度以上':''); }
function aDepTxt(m){ return m.depth<1?'ごく浅く':'約'+Math.round(m.depth)+'キロ'; }
function aTsuTxt1(r){ return (r.tsu!=null&&TSU_TXT[r.tsu])?TSU_TXT[r.tsu].replace('。','。'):''; }
// 震度別の地域（震度の大きい順）。minc 以上の階級を最大 maxLines 階級まで読む
function aAreaLines(r,minc,maxLines,cap,lead){
  const by={}; (r.areas||[]).forEach(a=>{ (by[a[1]]=by[a[1]]||[]).push(a[0]); });
  const cs=Object.keys(by).map(Number).filter(c=>c>=minc).sort((p,q)=>q-p).slice(0,maxLines);
  return cs.map(c=>'震度'+aSh(c)+'、'+aList(by[c],cap)+'。').join('');
}
// 「何時何分ごろ、○○を震源とする最大震度○の地震が発生しました。地震の規模を示すマグニチュードは○、△△、△△で最大震度○を観測しました。」
function aQuakeTxt(r,t0){
  const m=aMainMem(r), mc=aMxCls(r), ar=r.areas||[];
  let top=0; ar.forEach(a=>{ if(a[1]>top) top=a[1]; });
  const names=ar.filter(a=>a[1]===top).map(a=>a[0]);
  let s=aClockTxt(m?m.delay:t0)+'ごろ、'+(m?m.epi+'を震源とする':'')+(mc>0?'最大震度'+aSh(mc)+'の':'')+'地震が発生しました。';
  if(m) s+='地震の規模を示すマグニチュードは'+aMagTxt(m)+(names.length?'、'+aList(names,8)+'で最大震度'+aSh(top)+'を観測しました。':'です。');
  return s;
}
const INFO_SPK={
  miu:(r)=>'未入電だった観測点の震度を更新します。'+(r.rec||[]).slice(0,3).map(e=>e[0]+'で震度'+aSh(e[1])).join('、')+'。',
  sokuho:(r,t0)=>'震度速報。'+aClockTxt(t0)+'ごろ、強い地震がありました。'+aAreaLines(r,3,3,8),
  gensen:(r,t0)=>{ const m=aMainMem(r); return '震源に関する情報。'+aClockTxt(m?m.delay:t0)+'ごろ、'+(m?m.epi+'を震源とする':'')+'地震がありました。'
    +(m?'震源の深さは'+aDepTxt(m)+'、マグニチュードは'+aMagTxt(m)+'です。':'')+aTsuTxt1(r); },
  gs:(r,t0)=>'震源・震度に関する情報。'+aQuakeTxt(r,t0)+aTsuTxt1(r),
  kakuchi:(r,t0)=>{ const a=aQuakeTxt(r,t0), mc=aMxCls(r); const o=aAreaLines(r,3,3,6).replace(/^震度[^、]*、[^。]*。/,'');
    return '各地の震度に関する情報。'+a+(o?'このほか、'+o:''); },
  genkou:(r,t0)=>{ const m=aMainMem(r); return '顕著な地震の震源要素更新のお知らせ。'+(m?m.epi+'を震源とする地震の、震源の深さは'+aDepTxt(m)+'、マグニチュードは'+aMagTxt(m)+'です。':'')+aTsuTxt1(r); }
};
const aHl=h=>String(h).replace(/m/g,'メートル');
// 津波: 大津波警報・津波警報の地域は全て、予想される高さごとにまとめて読む（注意報のみのときは地域名を読む）
function aTsuTxt(st,top){
  let out='';
  [4,3].forEach(lv=>{
    const by=new Map();
    st.list.filter(e=>e.lv===lv).forEach(e=>{ const h=String(e.hl); if(!by.has(h)) by.set(h,[]); by.get(h).push(sim.tsuAreas[e.i][0]); });
    if(!by.size) return;
    out+=TSU_LV[lv]+'の地域です。';
    by.forEach((names,h)=>{ out+='予想される津波の高さは、'+aHl(h)+'、'+names.join('、')+'。'; });
  });
  if(top===2){ const n=st.list.filter(e=>e.lv===2).map(e=>sim.tsuAreas[e.i][0]); if(n.length) out+='津波注意報の地域は、'+aList(n,8)+'。'; }
  return out;
}
const aTsuLv=m=>(m>=5?1:0)+(m>=3?1:0)+(m>=1?1:0)+(m>0?1:0);
// (t0,t1] の間に起きた「鳴らす出来事」を集める（p が大きいほど重要）
// 警報の対象地域を、読み上げ用の地方名（東北・関東・北陸・甲信…）にまとめる。予想震度の大きい地方から順に並べる
const EEW_SPK_REG={'北海道':'北海道','青森県':'東北','岩手県':'東北','宮城県':'東北','秋田県':'東北','山形県':'東北','福島県':'東北',
  '茨城県':'関東','栃木県':'関東','群馬県':'関東','埼玉県':'関東','千葉県':'関東','東京都':'関東','神奈川県':'関東',
  '新潟県':'北陸','富山県':'北陸','石川県':'北陸','福井県':'北陸','山梨県':'甲信','長野県':'甲信',
  '岐阜県':'東海','静岡県':'東海','愛知県':'東海','三重県':'東海','滋賀県':'近畿','京都府':'近畿','大阪府':'近畿','兵庫県':'近畿','奈良県':'近畿','和歌山県':'近畿',
  '鳥取県':'中国','島根県':'中国','岡山県':'中国','広島県':'中国','山口県':'中国','徳島県':'四国','香川県':'四国','愛媛県':'四国','高知県':'四国',
  '福岡県':'九州北部','佐賀県':'九州北部','長崎県':'九州北部','熊本県':'九州北部','大分県':'九州北部','宮崎県':'九州南部','鹿児島県':'九州南部','沖縄県':'沖縄'};
const EEW_SPK_ORD=['北海道','東北','関東','北陸','甲信','東海','近畿','中国','四国','九州北部','九州南部','沖縄'];
function eewRegionsTxt(areas){
  const best={};
  (areas||[]).forEach(a=>{ const g=EEW_SPK_REG[a.pref]; if(!g) return; if(!(g in best)||a.cls>best[g]) best[g]=a.cls; });
  const ks=Object.keys(best).sort((p,q)=>best[q]-best[p]||EEW_SPK_ORD.indexOf(p)-EEW_SPK_ORD.indexOf(q));
  return ks.join('、');
}
function audioCollect(t0,t1){
  const ev=[]; if(!sim) return ev;
  // 緊急地震速報（初報／警報への格上げ／予想最大震度の更新）
  sim.quakes.forEach(qk=>{
    const e=qk.eew; if(!e) return; let lvm=0, cls=-1, spC=-9, spM=-9, spT=-1e9;   // sp*: 最後に読み上げた予想最大震度・M・時刻
    e.reports.forEach((r,k)=>{
      const inW=r.t>t0&&r.t<=t1;
      const epi=r.epiName||'震源付近';
      if(r.lv>=2&&lvm<2){ if(inW) ev.push({s:'eewWarn',p:90,k:true,x:'緊急地震速報。'+epi+'で地震。'+(eewRegionsTxt(r.areas)?eewRegionsTxt(r.areas)+'では':'')+'強い揺れに警戒してください。'}); spC=r.maxCls; spM=r.mag; spT=r.t; }
      else if(k===0){ if(inW) ev.push({s:'eewFcst',p:70,k:true,x:'緊急地震速報。'+epi+'で地震。'}); spC=r.maxCls; spM=r.mag; spT=r.t; }
      else if(r.maxCls!==spC||Math.abs(r.mag-spM)>=0.3||r.t-spT>=10){
        // 予想最大震度が変わった／予想Mが0.3以上変わった／10秒以上読み上げていない → 現在の推定を読み上げる
        if(inW) ev.push({s:'eewUpd',p:50,x:(r.maxCls>=1?'推定最大震度'+aSh(r.maxCls)+'、':'')+'マグニチュード'+r.mag.toFixed(1)+'。'});
        spC=r.maxCls; spM=r.mag; spT=r.t;
      }
      if(r.lv>lvm) lvm=r.lv; if(r.maxCls>cls) cls=r.maxCls;
    });
  });
  // 震度速報・地震情報
  if(sim.info) sim.info.groups.forEach(g=>g.releases.forEach(r=>{
    if(r.t>t0&&r.t<=t1){ const f=INFO_SPK[r.k];
      ev.push({s:r.upd?'infoUpd':'infoNew',p:r.k==='sokuho'?80:60,k:true,x:(r.upd?'情報を更新します。':'')+(f?f(r,g.t0):'')}); }
  }));
  // 津波警報・注意報・解除・観測
  const tq=sim.quakes.filter(q=>q.tsu&&q.tsu.reports.length);
  if(tq.length){
    const st=tsuState(t1); let top=0; st.list.forEach(e=>{ if(e.lv>top) top=e.lv; });
    tq.forEach(qk=>qk.tsu.reports.forEach(r=>{
      if(r.t<=t0||r.t>t1) return;
      if(r.kind==='none') ev.push({s:'cancel',p:45,k:true,x:String(r.text||'').replace(/<[^>]*>/g,'')});
      else if(r.kind==='cancel'||r.kind==='switch') ev.push({s:'tsuClear',p:45,k:true,x:String(r.text||'').replace(/<[^>]*>/g,'')});   // 解除・切替は警報発表時とは別の音で、文面どおりに読む
      else if(top>0){
        const issue=(r.kind==='issue');
        ev.push({s:['','tsuInfo','tsuAdv','tsuWarn','tsuMajor'][top],p:[0,40,85,95,100][top],k:true,
          x:(issue?TSU_MSG[top]:TSU_LV[top]+'の内容が更新されました。')+aTsuTxt(st,top)});
      }else ev.push({s:'tsuInfo',p:40,k:true,x:String(r.text||'').replace(/<[^>]*>/g,'')});
    }));
    const nobs=[]; let lvUp=false;                      // 観測が更新された予報区を集めて、1回の読み上げにまとめる（階級が上がったときだけ効果音）
    st.obs.forEach((o,a)=>{
      const prev=aObs.get(a)||0;
      if(o.mx>prev+1e-6){
        nobs.push({nm:(sim.tsuAreas[a]||['沿岸'])[0],mx:o.mx,cur:o.cur});
        if(aTsuLv(o.mx)>aTsuLv(prev)) lvUp=true;
        aObs.set(a,o.mx);
      }
    });
    if(nobs.length){
      nobs.sort((p,q)=>q.mx-p.mx); const top4=nobs.slice(0,4);
      ev.push({s:lvUp?'tsuObs':'',p:lvUp?55:35,x:'津波観測に関する情報。'+top4.map(e=>e.nm+'で第'+(e.cur?e.cur[2]:1)+'波、'+((e.cur&&e.cur[3]<0)?'引き':'押し')+'、'+e.mx.toFixed(1)+'メートル').join('、')
        +(nobs.length>top4.length?'、ほか'+(nobs.length-top4.length)+'か所。':'。')});
    }
  }
  // 観測された最大震度が上がった
  const clr=!!(sim.hasTsu&&t1>=sim.tSeis);
  if(!clr){ const c=aMaxClsAt(t1);
    if(c>aMaxCls){ if(c>=1){ ev.push({s:'newInt',a:c,p:65}); aFx(t1,c); } aMaxCls=c; } }
  return ev;
}
function audioResync(t){      // シーク・巻き戻し・一時停止中: 鳴らさずに「発表済み」の状態だけ合わせる
  aPrev=t; aMaxCls=aMaxClsAt(t); aObs=new Map();
  if(sim&&sim.quakes.some(q=>q.tsu&&q.tsu.reports.length)) tsuState(t).obs.forEach((o,a)=>{ aObs.set(a,o.mx); });
}
function aSilence(){          // 読み上げ中・順番待ちの音声をすべて止める
  aUtt.length=0;
  try{ if(window.speechSynthesis){ window.speechSynthesis.cancel(); } }catch(e){}
}
function audioReset(){
  FX.length=0;
  aPrev=-0.001; aMaxCls=-1; aObs=new Map(); aBusyUntil=0; aBusyPrio=0;
  if(window.speechSynthesis) window.speechSynthesis.cancel();
}
function audioStep(ts){
  if(!sim){ aPrev=-0.001; aMaxCls=-1; aObs=new Map(); aHint(false); return; }
  if(!aOn()) aHint(false);
  const t1=tcur;
  if(!playing||t1<aPrev-1e-6||ts-seekStamp<800){ audioResync(t1); return; }
  if(t1<=aPrev) return;
  const ev=audioCollect(aPrev,t1); aPrev=t1;
  if(!ev.length) return;
  ev.sort((p,q)=>q.p-p.p);
  audioPlay(ev[0].s,ev[0].p,ev[0].a);
  const sp=ev.find(e=>e.x); if(sp) audioSpeak(sp.x,sp.p,sp.k);
}
$('sndTest').onclick=()=>{ audioUnlock(); audioPlay('infoNew',100); audioSpeak('音声のテストです。震度速報。最大震度ごじゃくを観測しました。',100); };

function tick(ts){
  if(sim && playing){
    const d=Math.min((ts-lastTs)/1000,0.1);
    const inTsu=tcur>=sim.tSeis;                       // 地震動の計算範囲を過ぎたら津波フェーズの速度で進める
    const lim=limT();                                  // EEWの演算が確定している時刻（これ以降は再生できない）
    const selSp=parseFloat($('speed').value);
    const inInfo=inTsu && sim.info && tcur<(sim.info.end0||sim.info.end);     // 震度速報・地震情報の発表が続く間は、津波フェーズでも速度を上げすぎない
    const speed=inInfo?Math.max(10,selSp*5):(inTsu?parseFloat($('tsuSpeed').value):selSp), done=lim>=sim.tEnd-0.01;
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
    if(tcur>=sim.tEnd){ tcur=sim.tEnd; playing=false; $('play').textContent='►'; }
    $('seek').value=tToS(tcur); $('tlabel').textContent=fmtT(tcur);
  }
  lastTs=ts;
  // 再生中・地図の移動やズーム中・設定やスライダーの操作直後・時刻が動いたときだけ毎フレーム描く。
  // それ以外（一時停止中や待機中）は描画を約4〜6fpsに間引く（以前は何も変わらなくても毎フレーム全観測点を描き直していた）
  const mp=map.latLngToContainerPoint([0,0]), mk=mp.x+','+mp.y+','+map.getZoom();
  if(playing||FX.length||viewDirty||uiDirty||tcur!==lastDrawT||mk!==lastMapKey||ts-lastDrawTs>=(sim?250:160)){
    draw(ts); lastDrawTs=ts; lastDrawT=tcur; lastMapKey=mk; uiDirty=false;
  }
  updateInfo(ts); audioStep(ts); requestAnimationFrame(tick);
}
let lastDrawTs=0, lastDrawT=-1, lastMapKey='', uiDirty=true;
['input','change','click','keyup','pointerup'].forEach(e=>document.addEventListener(e,()=>{ uiDirty=true; },true));
requestAnimationFrame(tick);

$('play').onclick=()=>{
  if(!sim) return;
  if(tcur>=sim.tEnd-0.01) tcur=0;
  playing=!playing; $('play').textContent=playing?'❚❚':'►';
  if(!playing) aSilence();                                   // 停止したら読み上げも止める
};
$('restart').onclick=()=>{ if(!sim) return; aSilence(); tcur=0; $('seek').value=0; $('tlabel').textContent='0.0 s'; };
$('seek').addEventListener('input',()=>{ if(!sim) return; tcur=Math.min(sToT(parseInt($('seek').value)),limT()); $('seek').value=tToS(tcur); $('tlabel').textContent=fmtT(tcur); });

/* ---------- 実行 ---------- */
$('runFab').onclick=()=>{ if(!$('run').disabled) $('run').click(); };
document.addEventListener('keydown',e=>{            // Enterキーでも実行（入力欄の操作中は除く）
  if(e.key==='Enter'&&!e.isComposing&&!/^(INPUT|TEXTAREA|SELECT|BUTTON)$/.test((e.target.tagName||''))) $('runFab').click(); });
$('run').onclick=async()=>{
  const body={quakes:QS.map(q=>({lat:q.lat,lon:q.lon,mag:q.mag,depth:q.depth,delay:q.delay})),
              gz:(typeof DecompressionStream!=='undefined')};
  if(!QS.every(q=>[q.lat,q.lon,q.mag,q.depth,q.delay].every(isFinite))){ $('status').textContent='入力値を確認してください'; return; }
  if(runCtl) runCtl.abort(); const ctl=runCtl=new AbortController();
  $('run').disabled=$('runFab').disabled=true; $('runFab').textContent='計算中…'; $('status').textContent='計算中…'; playing=false; waitFlag=false;
  const handle=async(m)=>{
    if(ctl!==runCtl) return;
    const k=m.k, v=m.v;
    if(k==='base'){                                    // 震度フレーム到着 → すぐ再生開始
      const bin=atob(v.q); let u=new Uint8Array(bin.length);
      for(let i=0;i<bin.length;i++) u[i]=bin.charCodeAt(i);
      if(v.gz) u=new Uint8Array(await new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
      Q=new Int8Array(u.buffer,u.byteOffset,u.length);
      { const nf=v.nf; CUMV=new Int16Array(nf); CUMI=new Int32Array(nf); CUMV2=new Int16Array(nf); CUMI2=new Int32Array(nf);
        MUF=new Int32Array(N).fill(-1); MUEND=new Int32Array(N); MUFIRST=1e9; MUI=-1;
        const pk=new Int16Array(N).fill(-128), on=new Int32Array(N).fill(-1);
        for(let f=0;f<nf;f++){ const o=f*N; for(let i=0;i<N;i++){ const x=Q[o+i]; if(x>pk[i]) pk[i]=x; if(x>=55&&on[i]<0) on[i]=f; } }
        // 未入電: 計測震度5.5以上に達した陸上観測点が、確率 1-0.5^(計測震度-5.5) で未入電になる（6.5で1/2）。抽選は実行ごとに1回
        for(let i=0;i<N;i++){ if((SEA&&SEA[i])||pk[i]<55) continue;
          if(Math.random()<MU_PMAX*(1-Math.pow(0.5,pk[i]/10-5.5))){ MUF[i]=on[i]; MUEND[i]=on[i]+Math.round((20+Math.random()*10)/v.dt); if(on[i]<MUFIRST){ MUFIRST=on[i]; MUI=i; } } }
        let mv=-999, mi=-1, mv2=-999, mi2=-1;   // 陸上観測点の累積最大（CUMV2は未入電の観測点を除く）
        for(let f=0;f<nf;f++){ const o=f*N; for(let i=0;i<N;i++){ if(SEA&&SEA[i]) continue; const x=Q[o+i];
            if(x>mv){ mv=x; mi=i; }
            if(MUF[i]>=0&&f>=MUF[i]) continue;
            if(x>mv2){ mv2=x; mi2=i; } }
          CUMV[f]=mv; CUMI[f]=mi; CUMV2[f]=mv2; CUMI2[f]=mi2; }
        MUPK=pk; }
      muInfoPatch(v);
      PEAK=null; peakF=-1;
      v.quakes.forEach(q=>{ q.eew={reports:[]}; q.eewUpto=0; q.eewDone=false; });
      // 地震動の計算範囲(tSeis)の後ろに、津波・地震情報の発表が終わるまで時間軸を延長する
      v.tSeis=v.tEnd; v.hasTsu=(v.tsuEnd||0)>v.tEnd+1;
      { const ext=Math.max(v.tsuEnd||0,(v.info&&v.info.end)||0); v.tsuExt=ext>v.tEnd+1; if(v.tsuExt) v.tEnd=ext; }
      infoCnt=0; infoSig=''; infoPrev=null; infoUnread=0; imFitted=false; imState=null; drawInfoMap(); renderBadge();
      simClock0=(clkOrigin!==null&&!sim)?clkOrigin:(clkOrigin!==null?clkNowMs()-tcur*1000:Date.now());
      sim=v; tcur=0; audioReset(); buildEewCards(v.quakes.length);
      $('seek').disabled=false; $('seek').max=v.tsuExt?2000:Math.round(v.tEnd*10); $('seek').value=0; $('tlabel').textContent='0.0 s';
      showResult(v);
      $('status').textContent='再生中（EEW演算中）';
      playing=true; $('play').textContent='❚❚'; $('run').disabled=$('runFab').disabled=false;
      $('runFab').classList.remove('idle'); $('runFab').textContent='↻ もう一度実行';
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
  finally{ if(ctl===runCtl){ $('run').disabled=$('runFab').disabled=false; if($('runFab').textContent==='計算中…') $('runFab').textContent=sim?'↻ もう一度実行':'► シミュレーション実行'; } }
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
    tr.innerHTML='<td>'+a.name+'</td><td class="num">'+fx1(a.v)+'</td><td class="num"><span class="chip" style="background:'+clsColor(a.cls)+'">'+a.label+'</span></td>';
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
