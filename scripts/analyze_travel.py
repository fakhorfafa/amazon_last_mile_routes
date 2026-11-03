"""Optimizer-vs-driver gap and time-window risk, using the travel-time matrix.

travel_times.json is ~1.8 GB, so it is streamed one route at a time instead of loaded whole.
OR-Tools is not available here, so the optimizer is a numpy nearest-neighbour + 2-opt + Or-opt
local search on the (asymmetric) travel-time matrix. It is a strong heuristic, not a proven optimum.

Input  : data/raw/travel_times.json, data/processed/{stops,packages,route_features}.csv
Output : data/processed/{route_gap,zone_gap,package_tw}.csv, dashboard/travel.js
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, DASH = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "dashboard"
RISK_MIN = 15          # a package is "at risk" when it arrives within this many minutes of its window closing
MIN_ZONE_VISITS = 20   # zones need this many route visits to be ranked


# ---------- streaming reader ----------
def stream_routes(path, chunk=1 << 22):
    """Yield (route_id, matrix_dict) from a {route_id: {...}, ...} JSON file without loading it whole."""
    dec = json.JSONDecoder()
    with open(path, encoding="utf-8") as fh:
        buf, pos, eof = "", 0, False

        def fill():
            nonlocal buf, pos, eof
            more = fh.read(chunk)
            if not more:
                eof = True
            buf = buf[pos:] + more
            pos = 0

        def skip_ws():
            nonlocal pos
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,:":
                    pos += 1
                if pos < len(buf) or eof:
                    return
                fill()

        fill()
        skip_ws()
        assert buf[pos] == "{"
        pos += 1
        while True:
            skip_ws()
            if pos >= len(buf) or buf[pos] == "}":
                return
            while True:
                start = pos  # retry from the key if the value is cut off by the chunk boundary
                try:
                    key, end = dec.raw_decode(buf, pos)
                    pos = end
                    skip_ws()
                    val, end = dec.raw_decode(buf, pos)
                    pos = end
                    break
                except json.JSONDecodeError:
                    pos = start
                    if eof:
                        return  # truncated file: stop cleanly
                    fill()
            yield key, val


# ---------- tour search ----------
def tour_cost(M, t):
    return M[t[:-1], t[1:]].sum()


def nearest_neighbour(M):
    n = len(M)
    seen = np.zeros(n, bool)
    seen[0] = True
    t = [0]
    for _ in range(n - 1):
        d = M[t[-1]].copy()
        d[seen] = np.inf
        j = int(d.argmin())
        t.append(j)
        seen[j] = True
    return np.array(t)


def two_opt(M, t):
    """Open-path 2-opt on the asymmetric matrix; segment costs are re-evaluated in both directions."""
    t = t.copy()
    n = len(t)
    improved = True
    while improved:
        improved = False
        fwd = M[t[:-1], t[1:]]
        bwd = M[t[1:], t[:-1]]
        cf, cb = np.concatenate([[0], np.cumsum(fwd)]), np.concatenate([[0], np.cumsum(bwd)])
        for i in range(0, n - 2):
            js = np.arange(i + 2, n)
            # reverse t[i+1..j]: replace edge (i,i+1) and (j,j+1); inner edges flip direction
            inner_old = cf[js] - cf[i + 1]
            inner_new = cb[js] - cb[i + 1]
            a, b = t[i], t[i + 1]
            old = M[a, b] + inner_old
            new = M[a, t[js]] + inner_new
            nxt = js + 1
            has = nxt < n
            old = old + np.where(has, M[t[js], t[np.minimum(nxt, n - 1)]], 0)
            new = new + np.where(has, M[b, t[np.minimum(nxt, n - 1)]], 0)
            delta = new - old
            k = int(delta.argmin())
            if delta[k] < -1e-6:
                j = js[k]
                t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                improved = True
                break
    return t


def or_opt(M, t, max_len=3):
    """Move short segments (1..max_len stops) to a better position, keeping their direction."""
    t = list(t)
    improved = True
    while improved:
        improved = False
        n = len(t)
        for L in range(1, max_len + 1):
            for i in range(1, n - L + 1):
                seg = t[i:i + L]
                prev, nxt = t[i - 1], (t[i + L] if i + L < n else None)
                remove_gain = M[prev, seg[0]] + (M[seg[-1], nxt] if nxt is not None else 0) - (M[prev, nxt] if nxt is not None else 0)
                rest = t[:i] + t[i + L:]
                r = np.array(rest)
                a, b = r, np.append(r[1:], -1)
                ins = M[a, seg[0]] + np.where(b >= 0, M[seg[-1], b] - M[a, np.maximum(b, 0)], 0)
                k = int(ins.argmin())
                if ins[k] < remove_gain - 1e-6:
                    t = rest[:k + 1] + seg + rest[k + 1:]
                    improved = True
                    break
            if improved:
                break
    return np.array(t)


def improve(M, t):
    for _ in range(3):
        before = tour_cost(M, t)
        t = or_opt(M, two_opt(M, t))
        if tour_cost(M, t) > before - 1e-6:
            break
    return t


def solve(M, seed=None):
    """Best of local search from nearest-neighbour and, when given, from the driver's own order."""
    best = improve(M, nearest_neighbour(M))
    if seed is not None:
        alt = improve(M, seed)
        if tour_cost(M, alt) < tour_cost(M, best):
            best = alt
    return best


# ---------- per-route work ----------
def work(args):
    rid, mat, stops, svc, depart = args
    ids = stops.stop_id.tolist()
    idx = {s: i for i, s in enumerate(ids)}
    M = np.array([[mat[a].get(b, 0.0) for b in ids] for a in ids], float)
    actual = np.array([idx[s] for s in stops.sort_values("sequence").stop_id])
    opt = solve(M, actual)

    act_transit, opt_transit = tour_cost(M, actual), tour_cost(M, opt)
    service = svc.reindex(ids).fillna(0).to_numpy()

    def arrivals(t):
        legs = np.concatenate([[0], M[t[:-1], t[1:]]])
        start = np.cumsum(legs + np.concatenate([[0], service[t[:-1]]]))
        out = np.empty(len(t))
        out[t] = start
        return out  # seconds after departure, indexed by stop

    arr_act, arr_opt = arrivals(actual), arrivals(opt)

    # zone gap: intra-zone legs as driven vs. an optimized open path over the same stops from the same entry stop
    zones = []
    zid = stops.set_index("stop_id").zone_id.reindex(ids).to_numpy()
    for z in pd.unique(zid[actual][1:]):
        if not isinstance(z, str):
            continue
        order = [s for s in actual if zid[s] == z]
        if len(order) < 3:
            continue
        sub = M[np.ix_(order, order)]
        drv = sub[np.arange(len(order) - 1), np.arange(1, len(order))].sum()
        best = tour_cost(sub, solve(sub, np.arange(len(order))))
        zones.append((z, len(order), drv, best))

    return {"route_id": rid, "n": len(ids), "actual_transit_s": act_transit, "opt_transit_s": opt_transit,
            "service_s": service.sum(), "depart": depart,
            "arr_act": dict(zip(ids, arr_act)), "arr_opt": dict(zip(ids, arr_opt)), "zones": zones}


def main(limit=None):
    # Stop ids are two letters, so "NA" must not be read as a missing value.
    stops = pd.read_csv(PROC / "stops.csv", keep_default_na=False, na_values={"zone_id": [""]})
    pk = pd.read_csv(PROC / "packages.csv", keep_default_na=False, na_values={"tw_start_utc": [""], "tw_end_utc": [""]})
    rf = pd.read_csv(PROC / "route_features.csv")
    svc = pk.groupby(["route_id", "stop_id"]).planned_service_time_s.sum()
    st_by = dict(tuple(stops.groupby("route_id")))
    depart = (pd.to_datetime(rf.date + " " + rf.departure_time_utc)).set_axis(rf.route_id).to_dict()

    def jobs():
        for k, (rid, mat) in enumerate(stream_routes(RAW / "travel_times.json")):
            if limit and k >= limit:
                return
            if rid in st_by:
                yield rid, mat, st_by[rid], svc.loc[rid], depart[rid]

    rows, zrows, arr = [], [], []
    with Pool(max(1, min(14, __import__("os").cpu_count() - 2))) as pool:
        for k, r in enumerate(pool.imap_unordered(work, jobs(), chunksize=4)):
            rows.append({k2: r[k2] for k2 in ("route_id", "n", "actual_transit_s", "opt_transit_s", "service_s")})
            zrows += [{"route_id": r["route_id"], "zone_id": z, "stops": n, "driver_s": d, "opt_s": b} for z, n, d, b in r["zones"]]
            arr.append(pd.DataFrame({"route_id": r["route_id"], "stop_id": list(r["arr_act"]),
                                     "arr_act_s": list(r["arr_act"].values()), "arr_opt_s": [r["arr_opt"][s] for s in r["arr_act"]],
                                     "depart": r["depart"]}))
            if k % 250 == 0:
                print(f"{k} routes", flush=True)

    g = pd.DataFrame(rows).merge(rf[["route_id", "station_code", "city", "route_score", "date"]], on="route_id")
    g["gap_s"] = g.actual_transit_s - g.opt_transit_s
    g["gap_pct"] = g.gap_s / g.opt_transit_s
    g.to_csv(PROC / "route_gap.csv", index=False)
    z = pd.DataFrame(zrows).merge(rf[["route_id", "station_code", "city"]], on="route_id")
    z.to_csv(PROC / "zone_gap.csv", index=False)

    a = pd.concat(arr)
    p = pk[pk.tw_end_utc.notna()].merge(a, on=["route_id", "stop_id"])
    p["arr_act"] = pd.to_datetime(p.depart) + pd.to_timedelta(p.arr_act_s, unit="s")
    p["arr_opt"] = pd.to_datetime(p.depart) + pd.to_timedelta(p.arr_opt_s, unit="s")
    p["tw_start"], p["tw_end"] = pd.to_datetime(p.tw_start_utc), pd.to_datetime(p.tw_end_utc)
    for k in ("act", "opt"):
        p[f"late_{k}_min"] = (p[f"arr_{k}"] - p.tw_end).dt.total_seconds() / 60
        p[f"early_{k}_min"] = (p.tw_start - p[f"arr_{k}"]).dt.total_seconds() / 60
    p = p.merge(rf[["route_id", "station_code", "city", "route_score"]], on="route_id")
    p[["route_id", "stop_id", "package_id", "station_code", "city", "route_score", "tw_start", "tw_end", "arr_act", "arr_opt",
       "late_act_min", "early_act_min", "late_opt_min", "early_opt_min", "scan_status"]].to_csv(PROC / "package_tw.csv", index=False)
    export(g, z, p)


def export(g, z, p):
    def hist(v, lo, hi, step):
        e = np.arange(lo, hi + step / 2, step)
        c, _ = np.histogram(np.clip(v, lo, hi - 1e-9), e)
        return {"edges": e.round(3).tolist(), "counts": c.tolist()}

    zz = z.assign(gap_s=z.driver_s - z.opt_s)
    # Sub-zones (e.g. A-2.2A) rarely repeat across routes, so rank at the planning-zone level (A-2).
    zg = (zz.assign(zone=zz.zone_id.str.split(".").str[0])
            .groupby(["route_id", "station_code", "zone"])[["stops", "driver_s", "opt_s", "gap_s"]].sum().reset_index())
    zk = (zg.groupby(["station_code", "zone"])
            .agg(visits=("route_id", "size"), stops=("stops", "mean"), gap_s=("gap_s", "mean"), driver_s=("driver_s", "mean"), opt_s=("opt_s", "mean"))
            .reset_index().rename(columns={"zone": "zone_id"}))
    zk = zk[zk.visits >= MIN_ZONE_VISITS]
    zk["gap_pct"] = zk.gap_s / zk.opt_s
    intra = zz.groupby("route_id").gap_s.sum()
    gi = g.set_index("route_id")
    split = {"intra_s": float(intra.reindex(gi.index).fillna(0).mean()), "total_s": float(gi.gap_s.mean())}

    p["late"] = p.late_act_min > 0
    p["risk"] = (p.late_act_min <= 0) & (p.late_act_min > -RISK_MIN)
    p["early"] = p.early_act_min > 0
    p["late_o"] = p.late_opt_min > 0
    p["hour"] = p.tw_end.dt.hour

    out = {
        "gap": {
            "routes": int(len(g)), "mean_pct": round(g.gap_pct.mean(), 4), "median_pct": round(g.gap_pct.median(), 4),
            "mean_min": round(g.gap_s.mean() / 60, 1), "actual_h": round(g.actual_transit_s.mean() / 3600, 2), "opt_h": round(g.opt_transit_s.mean() / 3600, 2),
            "service_h": round(g.service_s.mean() / 3600, 2), "better_pct": round((g.gap_s < 0).mean(), 4),
            "hist": hist(g.gap_pct.to_numpy(), -0.1, 0.6, 0.025),
            "by_score": g.groupby("route_score").agg(gap_pct=("gap_pct", "median"), gap_min=("gap_s", lambda s: s.median() / 60), routes=("route_id", "size")).round(4).reset_index().to_dict("records"),
            "by_station": g.groupby(["station_code", "city"]).agg(gap_pct=("gap_pct", "median"), gap_min=("gap_s", lambda s: s.median() / 60),
                                                                  actual_h=("actual_transit_s", lambda s: s.mean() / 3600), routes=("route_id", "size"))
                           .round(4).reset_index().sort_values("gap_pct", ascending=False).to_dict("records"),
            "zones_top": zk.sort_values("gap_s", ascending=False).head(15).round(3).to_dict("records"),
            "zone_count": int(len(zk)), "split": {k: round(v, 1) for k, v in split.items()},
        },
        "tw": {
            "packages": int(len(p)), "routes": int(p.route_id.nunique()), "risk_min": RISK_MIN,
            "late_pct": round(p.late.mean(), 4), "risk_pct": round(p.risk.mean(), 4), "early_pct": round(p.early.mean(), 4),
            "late_opt_pct": round(p.late_o.mean(), 4),
            "late_median_min": round(p.loc[p.late, "late_act_min"].median(), 1) if p.late.any() else 0,
            "slack_hist": hist((-p.late_act_min).to_numpy(), -120, 480, 30),
            "by_station": p.groupby(["station_code", "city"]).agg(packages=("package_id", "size"), late_pct=("late", "mean"), risk_pct=("risk", "mean"),
                                                                  late_opt_pct=("late_o", "mean")).round(4).reset_index().sort_values("late_pct", ascending=False).to_dict("records"),
            "by_score": p.groupby("route_score").agg(late_pct=("late", "mean"), risk_pct=("risk", "mean"), packages=("package_id", "size")).round(4).reset_index().to_dict("records"),
            "by_hour": p.groupby("hour").agg(packages=("package_id", "size"), late_pct=("late", "mean")).round(4).reset_index().to_dict("records"),
        },
    }
    (DASH / "travel.js").write_text("window.ALMRRC_T = " + json.dumps(out, default=float) + ";\n", encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))} for k, v in out.items()}, indent=1))


def reexport():
    """Rebuild dashboard/travel.js from the saved CSVs without re-solving every route."""
    g = pd.read_csv(PROC / "route_gap.csv")
    z = pd.read_csv(PROC / "zone_gap.csv", keep_default_na=False)
    p = pd.read_csv(PROC / "package_tw.csv", keep_default_na=False, parse_dates=["tw_start", "tw_end", "arr_act", "arr_opt"])
    export(g, z, p)


if __name__ == "__main__":
    if sys.argv[1:] == ["--export"]:
        reexport()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
