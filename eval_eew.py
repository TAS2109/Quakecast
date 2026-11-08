# -*- coding: utf-8 -*-
"""
eval_eew.py — Quakecast の緊急地震速報(EEW)の精度評価ツール

目的: 「気象庁の実際のEEWとほぼ同じ結果が出る」ことを、数値で測れるようにする。
      app.py を変更するたびに実行し、良くなったか悪くなったかを比較する。

置き場所: app.py（または app (24).py など app*.py）と同じフォルダ。data/ も同じ場所にあること。
          eew_reference.json（基準データ）も同じフォルダ。

使い方:
  python eval_eew.py                       # 基準データの全地震を実行
  python eval_eew.py --grid                # M×場所の格子で、報数・最終報時刻を実測値の範囲と比較（遅い）
  python eval_eew.py --only noto2024,iburi2018
  python eval_eew.py --repeat 3 --jobs 4   # 震源を少しずらして3回ずつ。4プロセス並列
  python eval_eew.py --save-baseline v1    # 結果を保存
  python eval_eew.py --compare v1          # 保存した結果との差を表示（改善/悪化）
  python eval_eew.py --list                # 基準データの一覧だけ表示（app.py を読み込まない）

比べているもの（2種類あることに注意）:
  [A] 気象庁の実発表との差  … eew_reference.json に値を入れた項目だけ。これが最終目標。
  [B] シミュレーション内の真値との差 … 震源位置・M・各区域の最大震度（このアプリの震度計算結果）。
      app.py の震度計算が気象庁の実際の揺れと違えば、[B]が良くても[A]は良くならない。切り分けに使う。
"""
import argparse
import base64
import contextlib
import importlib.util
import io
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OBS_TO_CLS = {"1": 1, "2": 2, "3": 3, "4": 4, "5-": 5, "5+": 6, "6-": 7, "6+": 8, "7": 9}
CLS_LABEL = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
NAN = float("nan")
_A = None            # 読み込んだ app モジュール


# ---------------------------------------------------------------------------
# app モジュールの読み込み
# ---------------------------------------------------------------------------
def load_app(path=None, verbose=False):
    if path:
        p = Path(path)
    else:
        cands = [c for c in HERE.glob("app*.py") if c.name != Path(__file__).name]
        if not cands:
            sys.exit("app.py が見つかりません（このスクリプトと同じフォルダに置くか --app で指定）")
        exact = [c for c in cands if c.name == "app.py"]
        p = exact[0] if exact else max(cands, key=lambda c: c.stat().st_mtime)
    if not (p.parent / "data").is_dir():
        sys.exit("%s の隣に data/ フォルダがありません（stations.csv などが必要）" % p)
    spec = importlib.util.spec_from_file_location("quakecast_app", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["quakecast_app"] = mod
    sink = sys.stderr if verbose else io.StringIO()
    with contextlib.redirect_stdout(sink):
        spec.loader.exec_module(mod)
    print("読み込み: %s（観測点 %d）" % (p.name, mod.N_ST), file=sys.stderr)
    return mod


# ---------------------------------------------------------------------------
# 基準データ
# ---------------------------------------------------------------------------
def load_reference(path=None):
    p = Path(path) if path else HERE / "eew_reference.json"
    if not p.exists():
        p = HERE / "data" / "eew_reference.json"
    if not p.exists():
        sys.exit("eew_reference.json が見つかりません")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def grid_cases():
    locs = [("inland", 36.0, 138.0, 10.0), ("offshore", 38.0, 142.5, 30.0), ("west", 34.5, 135.5, 15.0)]
    out = []
    for nm, la, lo, d in locs:
        for m in (4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0):
            out.append({"id": "grid_%s_M%.1f" % (nm, m), "name": "格子 %s M%.1f" % (nm, m),
                        "lat": la, "lon": lo, "mag": m, "depth": d, "plum_only": False, "eew": None,
                        "observed_max": None, "grid": True})
    return out


def in_band(ref, mag):
    for b in (ref.get("envelope") or {}).get("bands", []):
        if b["mag"][0] <= mag <= b["mag"][1]:
            return b
    return None


# ---------------------------------------------------------------------------
# 1地震の評価
# ---------------------------------------------------------------------------
def _nz(x):
    return NAN if x is None else float(x)


def _area_arrays(A):
    nA = int(A.ST_AREA.max()) + 1
    return nA, A.ST_LAND, A.ST_AREA


def area_pred(A, rep, nA):
    pv = np.frombuffer(base64.b64decode(rep["pq"]), dtype=np.int8).astype(np.float64) / 10.0
    ap = np.full(nA, -99.0)
    np.maximum.at(ap, A.ST_AREA[A.ST_LAND], pv[A.ST_LAND])
    return ap


def intensity_scores(A, rep, act_cls, nA):
    """区域ごとの予想震度階級と、シミュレーション上の実際の階級の比較（どちらかが震度3以上の区域が対象）"""
    ap = area_pred(A, rep, nA)
    ex = w1 = n = 0
    bias = []
    for a, ac in act_cls.items():
        pc = A.shindo_class(float(ap[a]))[0] if ap[a] > -90 else 0
        if ac < 3 and pc < 3:
            continue
        n += 1
        ex += (pc == ac)
        w1 += (abs(pc - ac) <= 1)
        bias.append(pc - ac)
    if n == 0:
        return NAN, NAN, NAN
    return ex / n, w1 / n, float(np.mean(bias))


def run_case(c, jitter=0, ref=None):
    A = _A
    t_start = time.time()
    lat, lon, mag, depth = c["lat"] + 0.001 * jitter, c["lon"], c["mag"], c["depth"]
    plum = bool(c.get("plum_only"))
    core = A.quake_core(lat, lon, mag, depth)
    tp, rt, t_end, m01, s_curve = core["tp"], core["rt"], core["t_end"], core["m01"], core["s_curve"]
    reps = []
    for kind, v in A.eew_stream(mag, depth, lat, lon, tp, rt, t_end, plum_only=plum):
        if kind == "report":
            reps.append(v)
    nA, land, st_area = _area_arrays(A)

    # シミュレーション上の実際の揺れ
    act = np.full(nA, -99.0)
    np.maximum.at(act, st_area[land], m01[land])
    act_cls = {a: A.shindo_class(float(act[a]))[0] for a in range(nA) if act[a] > -90}
    sim_max_cls = max(act_cls.values()) if act_cls else 0
    obs_cls = OBS_TO_CLS.get(c.get("observed_max") or "", None)

    fin_tp = tp[np.isfinite(tp)]
    t_det = float(fin_tp.min()) if len(fin_tp) else NAN
    m = {"id": c["id"], "name": c.get("name", c["id"]), "mag": mag, "depth": depth,
         "n_reports": len(reps), "sim_max_cls": sim_max_cls,
         "sim_vs_obs_cls": (sim_max_cls - obs_cls) if obs_cls is not None else NAN, "grid": bool(c.get("grid"))}
    keys = ["t_first_rel", "t_final_rel", "first_to_final", "pos_err_first", "pos_err_10s", "pos_err_final",
            "dep_err_final", "mag_err_first", "mag_err_final", "mag_bias_final", "mag_swing", "mag_reversals",
            "area_exact", "area_within1", "area_bias", "area_exact_10s", "lead_median", "lead_min",
            "env_final_dev", "env_n_dev", "d_n_reports", "d_mag_first", "d_mag_final", "d_pos_first", "d_pos_final",
            "d_depth_first", "d_depth_final", "d_maxcls_first", "d_maxcls_final", "d_final_after_first", "warn_match"]
    for k in keys:
        m[k] = NAN
    for k in ("warn_tp", "warn_fn", "warn_fp", "blind_n", "ev_hit", "ev_miss", "ev_false", "ev_tn"):
        m[k] = 0
    m["warned"] = False
    m["special"] = False

    # 警報の判定は報がなくても行う（報なし＝警報なし）
    warned_ids = set()
    first_warn = {}
    for r in reps:
        for a in r["areas"]:
            warned_ids.add(a["i"])
            first_warn.setdefault(a["i"], r["t"])
    m["warned"] = bool(warned_ids) or any(r["lv"] >= 2 for r in reps)
    m["special"] = any(r["lv"] >= 3 for r in reps)
    act4 = {a for a, ac in act_cls.items() if ac >= 4}
    m["warn_tp"] = len(warned_ids & act4)
    m["warn_fn"] = len(act4 - warned_ids)
    m["warn_fp"] = len(warned_ids - act4)
    should = sim_max_cls >= 5
    m["ev_hit"], m["ev_miss"] = int(should and m["warned"]), int(should and not m["warned"])
    m["ev_false"], m["ev_tn"] = int((not should) and m["warned"]), int((not should) and not m["warned"])

    if reps:
        r0, rl = reps[0], reps[-1]
        m["t_first_rel"] = r0["t"] - t_det
        m["t_final_rel"] = rl["t"] - t_det
        m["first_to_final"] = rl["t"] - r0["t"]

        def at(rel):
            sel = None
            for r in reps:
                if r["t"] - r0["t"] <= rel:
                    sel = r
                else:
                    break
            return sel
        pe = lambda r: float(A._planar_km(r["lat"], r["lon"], lat, lon))
        if not plum:     # PLUM法のみは仮定震源（最初の検知点・M1.0）なので震源の精度は評価しない
            m["pos_err_first"], m["pos_err_final"] = pe(r0), pe(rl)
            r10 = at(10.0)
            m["pos_err_10s"] = pe(r10)
            m["dep_err_final"] = abs(rl["depth"] - depth)
            m["mag_err_first"], m["mag_err_final"] = abs(r0["mag"] - mag), abs(rl["mag"] - mag)
            m["mag_bias_final"] = rl["mag"] - mag
            mg = [r["mag"] for r in reps]
            m["mag_swing"] = float(sum(abs(b - a) for a, b in zip(mg[:-1], mg[1:])))
            m["mag_reversals"] = int(sum(1 for a, b in zip(mg[:-1], mg[1:]) if b < a - 0.25))
        m["area_exact"], m["area_within1"], m["area_bias"] = intensity_scores(A, rl, act_cls, nA)
        m["area_exact_10s"] = intensity_scores(A, at(10.0), act_cls, nA)[0]

        # 警報が地震の揺れ（S波）にどれだけ先行したか（警報対象になった震度4以上の区域）
        dist = A.hubeny_km(lat, lon, A.ST_LAT, A.ST_LON)
        tf = np.arange(len(s_curve)) * 1.0
        ts = np.interp(dist, s_curve, tf, right=np.inf)
        leads = []
        for a in warned_ids & act4:
            idx = np.where((st_area == a) & land)[0]
            if len(idx):
                leads.append(float(ts[idx].min()) - first_warn[a])
        if leads:
            m["lead_median"], m["lead_min"] = float(np.median(leads)), float(min(leads))
            m["blind_n"] = int(sum(1 for x in leads if x <= 0))

        # 実測値の範囲（M帯ごと）との比較
        b = in_band(ref or {}, mag) if ref else None
        if b:
            lo, hi = b["final_rel"]
            m["env_final_dev"] = 0.0 if lo <= m["t_final_rel"] <= hi else (m["t_final_rel"] - (lo if m["t_final_rel"] < lo else hi))
            if b.get("n_reports"):
                lo, hi = b["n_reports"]
                n = len(reps)
                m["env_n_dev"] = 0.0 if lo <= n <= hi else float(n - (lo if n < lo else hi))

        # 気象庁の実発表との差（eew_reference.json に値がある項目だけ）
        j = c.get("eew") or {}
        if j.get("n_reports") is not None:
            m["d_n_reports"] = len(reps) - j["n_reports"]
        if j.get("final_after_first_s") is not None:
            m["d_final_after_first"] = m["first_to_final"] - j["final_after_first_s"]
        for tag, sr in (("first", r0), ("final", rl)):
            jr = j.get(tag) or {}
            if jr.get("mag") is not None and not plum:
                m["d_mag_" + tag] = sr["mag"] - jr["mag"]
            if jr.get("depth") is not None and not plum:
                m["d_depth_" + tag] = sr["depth"] - jr["depth"]
            if jr.get("lat") is not None and jr.get("lon") is not None and not plum:
                m["d_pos_" + tag] = float(A._planar_km(sr["lat"], sr["lon"], jr["lat"], jr["lon"]))
            if jr.get("max_pred") in OBS_TO_CLS:
                m["d_maxcls_" + tag] = sr["maxCls"] - OBS_TO_CLS[jr["max_pred"]]
    j = c.get("eew") or {}
    if j.get("warning_issued") is not None:
        m["warn_match"] = float(bool(j["warning_issued"]) == m["warned"])
    m["mag_traj"] = [[round(r["t"] - t_det, 1), r["mag"], r["depth"], r["maxCls"], r["lv"]] for r in reps] if len(reps) < 200 else []
    m["elapsed_s"] = round(time.time() - t_start, 1)
    return m


def _safe_run(args):
    c, jit, ref = args
    try:
        return run_case(c, jit, ref)
    except Exception as e:      # 1件の失敗で全体を止めない
        import traceback
        return {"id": c["id"] + ("#%d" % jit if jit else ""), "error": "%s: %s" % (type(e).__name__, e),
                "tb": traceback.format_exc()}


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------
# (キー, 表示名, 集計方法, 小さいほど良い か 大きいほど良い か, 小数桁)
SUMMARY = [
    ("pos_err_first", "震央誤差 第1報 [km]", "mean", "low", 1),
    ("pos_err_10s", "震央誤差 第1報+10秒 [km]", "mean", "low", 1),
    ("pos_err_final", "震央誤差 最終報 [km]", "mean", "low", 1),
    ("dep_err_final", "深さ誤差 最終報 [km]", "mean", "low", 1),
    ("mag_err_first", "M誤差(絶対) 第1報", "mean", "low", 2),
    ("mag_err_final", "M誤差(絶対) 最終報", "mean", "low", 2),
    ("mag_bias_final", "M偏り 最終報（＋=過大）", "mean", "zero", 2),
    ("mag_swing", "Mの上下動の総和", "mean", "low", 2),
    ("mag_reversals", "Mが0.25以上下がった回数", "mean", "low", 2),
    ("area_exact", "予想震度の一致率（区域・最終報）", "mean", "high", 3),
    ("area_within1", "予想震度の±1階級以内（区域・最終報）", "mean", "high", 3),
    ("area_bias", "予想震度の偏り（＋=過大）", "mean", "zero", 2),
    ("area_exact_10s", "予想震度の一致率（第1報+10秒）", "mean", "high", 3),
    ("warn_recall", "警報の網羅率（震度4以上の区域）", "derived", "high", 3),
    ("warn_precision", "警報の的中率（対象区域のうち実際に震度4以上）", "derived", "high", 3),
    ("ev_miss", "警報の見逃し（震度5弱以上なのに警報なし）[件]", "sum", "low", 0),
    ("ev_false", "警報の空振り（震度4以下なのに警報）[件]", "sum", "low", 0),
    ("lead_median", "警報の先行時間 中央値 [秒]", "mean", "high", 1),
    ("blind_n", "警報が間に合わなかった区域 [区域]", "sum", "low", 0),
    ("env_final_dev", "最終報時刻の実測範囲からのずれ [秒]", "meanabs", "low", 1),
    ("env_n_dev", "報数の実測範囲からのずれ [報]", "meanabs", "low", 1),
    ("d_mag_first", "【気象庁との差】第1報のM", "meanabs", "low", 2),
    ("d_mag_final", "【気象庁との差】最終報のM", "meanabs", "low", 2),
    ("d_n_reports", "【気象庁との差】報数", "meanabs", "low", 1),
    ("d_pos_first", "【気象庁との差】第1報の震央 [km]", "mean", "low", 1),
    ("d_maxcls_first", "【気象庁との差】第1報の予想最大震度[階級]", "meanabs", "low", 2),
    ("d_maxcls_final", "【気象庁との差】最終報の予想最大震度[階級]", "meanabs", "low", 2),
    ("d_final_after_first", "【気象庁との差】第1報→最終報の秒数", "meanabs", "low", 1),
    ("warn_match", "【気象庁との差】警報の有無が一致した割合", "mean", "high", 3),
]


def _vals(rows, k):
    return [r[k] for r in rows if k in r and isinstance(r[k], (int, float)) and not (isinstance(r[k], float) and math.isnan(r[k]))]


def summarize(rows):
    rows = [r for r in rows if "error" not in r]
    out = {}
    for k, _, agg, _, _ in SUMMARY:
        if agg == "derived":
            tp, fn, fp = sum(r["warn_tp"] for r in rows), sum(r["warn_fn"] for r in rows), sum(r["warn_fp"] for r in rows)
            if k == "warn_recall":
                out[k] = tp / (tp + fn) if tp + fn else NAN
            else:
                out[k] = tp / (tp + fp) if tp + fp else NAN
            continue
        v = _vals(rows, k)
        if not v:
            out[k] = NAN
        elif agg == "sum":
            out[k] = float(sum(v))
        elif agg == "meanabs":
            out[k] = float(np.mean(np.abs(v)))
        else:
            out[k] = float(np.mean(v))
        out[k + "__n"] = len(v)
    return out


def fmt(v, nd=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return ("%." + str(nd) + "f") % v


def case_table(rows):
    hdr = ["ID", "M", "深さ", "報数", "最終報[s]", "震央誤差(初/終)km", "M誤差(初/終)", "区域一致/±1", "警報 的中/見逃/空振", "先行[s]", "警報", "実測差"]
    lines = []
    for r in rows:
        if "error" in r:
            lines.append([r["id"], "ERROR", r["error"]] + [""] * 9)
            continue
        jd = []
        for k, lab in (("d_n_reports", "報数"), ("d_mag_final", "最終M"), ("d_maxcls_final", "最大震度")):
            if not (isinstance(r[k], float) and math.isnan(r[k])):
                jd.append("%s%+.1f" % (lab, r[k]))
        if not (isinstance(r["warn_match"], float) and math.isnan(r["warn_match"])):
            jd.append("警報" + ("○" if r["warn_match"] else "×"))
        env = []
        if not math.isnan(r["env_final_dev"]) and r["env_final_dev"] != 0:
            env.append("最終報%+.0fs" % r["env_final_dev"])
        if not math.isnan(r["env_n_dev"]) and r["env_n_dev"] != 0:
            env.append("報数%+.0f" % r["env_n_dev"])
        lines.append([
            r["id"], "%.1f" % r["mag"], "%d" % r["depth"], str(r["n_reports"]), fmt(r["t_final_rel"], 0),
            fmt(r["pos_err_first"], 0) + "/" + fmt(r["pos_err_final"], 0),
            fmt(r["mag_err_first"], 1) + "/" + fmt(r["mag_err_final"], 1),
            fmt(r["area_exact"], 2) + "/" + fmt(r["area_within1"], 2),
            "%d/%d/%d" % (r["warn_tp"], r["warn_fn"], r["warn_fp"]), fmt(r["lead_median"], 0),
            "あり" if r["warned"] else "なし", " ".join(jd + env) or "-"])
    w = [max(len(str(x[i])) for x in lines + [hdr]) for i in range(len(hdr))]
    out = ["  ".join(str(h).ljust(w[i]) for i, h in enumerate(hdr))]
    for ln in lines:
        out.append("  ".join(str(x).ljust(w[i]) for i, x in enumerate(ln)))
    return "\n".join(out)


def summary_text(s):
    out = []
    for k, lab, agg, good, nd in SUMMARY:
        n = s.get(k + "__n")
        out.append("%-52s %10s %s" % (lab, fmt(s.get(k), nd), ("(n=%d)" % n) if n is not None else ""))
    return "\n".join(out)


def compare_text(cur, base):
    out = []
    for k, lab, agg, good, nd in SUMMARY:
        a, b = cur.get(k, NAN), base.get(k, NAN)
        if any(isinstance(x, float) and math.isnan(x) for x in (a, b)) or a is None or b is None:
            out.append("%-52s %10s -> %10s" % (lab, fmt(b, nd), fmt(a, nd)))
            continue
        d = a - b
        if abs(d) < 10 ** -(nd + 1):
            mark = "  ＝"
        else:
            if good == "high":
                better = d > 0
            elif good == "low":
                better = d < 0
            else:
                better = abs(a) < abs(b)
            mark = "  ◎改善" if better else "  ▲悪化"
        out.append("%-52s %10s -> %10s%s" % (lab, fmt(b, nd), fmt(a, nd), mark))
    return "\n".join(out)


def jsonable(o):
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return jsonable(o.item())
    return o


# ---------------------------------------------------------------------------
def main():
    global _A
    ap = argparse.ArgumentParser(description="Quakecast EEW 精度評価")
    ap.add_argument("--app", help="app.py のパス（省略時は同じフォルダの app*.py）")
    ap.add_argument("--cases", help="基準データJSON（省略時は eew_reference.json）")
    ap.add_argument("--only", help="実行する地震ID（カンマ区切り）")
    ap.add_argument("--grid", action="store_true", help="M×場所の格子の地震も実行（報数・最終報時刻の実測範囲との比較）")
    ap.add_argument("--grid-only", action="store_true", help="格子の地震だけ実行")
    ap.add_argument("--repeat", type=int, default=1, help="震源を少しずつずらして何回ずつ実行するか（ばらつきの確認）")
    ap.add_argument("--jobs", type=int, default=1, help="並列プロセス数")
    ap.add_argument("--out", default=str(HERE / "eval_out"), help="結果の出力先")
    ap.add_argument("--save-baseline", metavar="NAME")
    ap.add_argument("--compare", metavar="NAME")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    ref = load_reference(a.cases)
    cases = [] if a.grid_only else list(ref["cases"])
    if a.grid or a.grid_only:
        cases += grid_cases()
    if a.only:
        want = set(x.strip() for x in a.only.split(","))
        cases = [c for c in cases if c["id"] in want]
    if a.list:
        for c in cases:
            j = c.get("eew") or {}
            filled = [k for k in ("warning_issued", "n_reports", "final_after_first_s") if j.get(k) is not None]
            filled += [t + "." + k for t in ("first", "final") for k, v in (j.get(t) or {}).items() if v is not None]
            print("%-16s M%.1f 深さ%3dkm 実測値: %s" % (c["id"], c["mag"], c["depth"], ",".join(filled) or "なし"))
        return
    if not cases:
        sys.exit("実行する地震がありません")

    _A = load_app(a.app, a.verbose)
    jobs = [(c, k, ref) for c in cases for k in range(max(a.repeat, 1))]
    t0 = time.time()
    rows = []
    if a.jobs > 1 and "fork" in multiprocessing.get_all_start_methods():
        with ProcessPoolExecutor(max_workers=a.jobs, mp_context=multiprocessing.get_context("fork")) as ex:
            for i, r in enumerate(ex.map(_safe_run, jobs), 1):
                rows.append(r)
                print("  [%d/%d] %s" % (i, len(jobs), r["id"]), file=sys.stderr)
    else:
        for i, jb in enumerate(jobs, 1):
            r = _safe_run(jb)
            rows.append(r)
            print("  [%d/%d] %s (%.0fs)" % (i, len(jobs), r["id"], r.get("elapsed_s", 0)), file=sys.stderr)
    for r, (c, k, _) in zip(rows, jobs):
        if k and "error" not in r:
            r["id"] = "%s#%d" % (r["id"], k)
    summ = summarize(rows)

    real = [r for r in rows if not r.get("grid")]
    report = ["== 地震ごとの結果 ==", case_table(rows), "",
              "== 全体の指標（%d件 / %.0f秒）==" % (len(rows), time.time() - t0), summary_text(summ)]
    if any(r.get("grid") for r in rows) and real:
        report += ["", "== 実在地震のみの指標 ==", summary_text(summarize(real))]
    errs = [r for r in rows if "error" in r]
    for r in errs:
        report += ["", "!! %s でエラー: %s" % (r["id"], r["error"]), r.get("tb", "")]
    if a.compare:
        bp = Path(a.out) / ("baseline_%s.json" % a.compare)
        if bp.exists():
            base = json.load(open(bp, encoding="utf-8"))["summary"]
            base = {k: (NAN if v is None else v) for k, v in base.items()}
            report += ["", "== %s との比較（保存時 → 今回）==" % a.compare, compare_text(summ, base)]
        else:
            report += ["", "!! 比較用の %s がありません（先に --save-baseline %s で保存）" % (bp.name, a.compare)]
    text = "\n".join(report)
    print(text)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    payload = jsonable({"time": stamp, "summary": summ, "cases": rows})
    with open(out / ("result_%s.json" % stamp), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    with open(out / ("report_%s.txt" % stamp), "w", encoding="utf-8") as f:
        f.write(text)
    if a.save_baseline:
        with open(out / ("baseline_%s.json" % a.save_baseline), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print("\n基準として保存: %s" % (out / ("baseline_%s.json" % a.save_baseline)))
    print("\n出力先: %s" % out)


if __name__ == "__main__":
    main()
