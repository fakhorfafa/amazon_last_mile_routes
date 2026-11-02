"""Extra analyses for the dashboard: High vs Low, fleet capacity, service-time model, station benchmark.

Input  : data/processed/{route_features,packages}.csv
Output : dashboard/extra.js
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DASH = ROOT / "dashboard"
SEED = 42
LOW_UTIL, HIGH_UTIL = 0.5, 0.95

COMPARE = ["n_stops", "n_packages", "path_km", "path_vs_nn", "first_leg_km", "n_zones", "zone_switches_per_zone",
           "zones_revisited_pct", "tw_pct", "failed_pct", "service_time_h", "capacity_util", "departure_hour_utc"]


def high_vs_low(f):
    hl = f[f.route_score.isin(["High", "Low"])]
    rows = []
    for c in COMPARE:
        h, l = hl.loc[hl.route_score == "High", c], hl.loc[hl.route_score == "Low", c]
        pooled = np.sqrt((h.var() + l.var()) / 2)
        rows.append({"feature": c, "high": round(h.mean(), 4), "low": round(l.mean(), 4),
                     "effect": round((h.mean() - l.mean()) / pooled, 3) if pooled else 0})
    rows.sort(key=lambda r: -abs(r["effect"]))

    X, y = hl[COMPARE].fillna(0), (hl.route_score == "High").astype(int)
    model = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
    proba = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]

    hour = (f.groupby("departure_hour_utc")
             .agg(routes=("route_id", "size"), high_pct=("route_score", lambda s: (s == "High").mean()))
             .reset_index())
    hour = hour[hour.routes >= 30]
    return {"compare": rows, "n_high": int(y.sum()), "n_low": int((1 - y).sum()),
            "auc": round(roc_auc_score(y, proba), 3),
            "by_hour": hour.round(4).to_dict("records")}


def capacity(f):
    u = f.capacity_util
    edges = np.arange(0, 1.45, 0.05)
    counts, _ = np.histogram(u.clip(0, edges[-1] - 1e-9), edges)
    st = (f.groupby(["station_code", "city"])
           .agg(routes=("route_id", "size"), median_util=("capacity_util", "median"),
                under_pct=("capacity_util", lambda s: (s < LOW_UTIL).mean()),
                over_pct=("capacity_util", lambda s: (s > HIGH_UTIL).mean()))
           .reset_index().sort_values("median_util"))
    return {"median": round(u.median(), 4), "under_pct": round((u < LOW_UTIL).mean(), 4), "over_pct": round((u > HIGH_UTIL).mean(), 4),
            "low": LOW_UTIL, "high": HIGH_UTIL,
            "hist": {"edges": edges.round(2).tolist(), "counts": counts.tolist()},
            "by_score": f.groupby("route_score").capacity_util.median().round(4).to_dict(),
            "by_station": st.round(4).to_dict("records")}


def service_model(pk, f):
    pk = pk.assign(volume_l=pk.depth_cm * pk.height_cm * pk.width_cm / 1000,
                   max_dim_cm=pk[["depth_cm", "height_cm", "width_cm"]].max(axis=1),
                   has_tw=pk.tw_start_utc.notna())
    stops = (pk.groupby(["route_id", "stop_id"])
               .agg(service_s=("planned_service_time_s", "sum"), n_pkgs=("package_id", "size"),
                    volume_l=("volume_l", "sum"), max_dim_cm=("max_dim_cm", "max"), has_tw=("has_tw", "max"))
               .reset_index()
               .merge(f[["route_id", "city"]], on="route_id"))
    stops = stops[stops.service_s < stops.service_s.quantile(0.995)]  # drop extreme outliers
    for c in f.city.unique():
        stops[f"city_{c}"] = (stops.city == c).astype(int)
    feats = ["n_pkgs", "volume_l", "max_dim_cm", "has_tw"] + [c for c in stops.columns if c.startswith("city_")]
    X, y = stops[feats].astype(float), stops.service_s
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)

    lin = LinearRegression().fit(Xtr[["n_pkgs", "volume_l"]], ytr)
    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=SEED).fit(Xtr, ytr)
    pred_lin, pred_gbm = lin.predict(Xte[["n_pkgs", "volume_l"]]), gbm.predict(Xte)

    by_n = (stops.assign(n=stops.n_pkgs.clip(upper=10))
                 .groupby("n").service_s.agg(["median", "count", lambda s: s.quantile(.25), lambda s: s.quantile(.75)])
                 .reset_index())
    by_n.columns = ["n_pkgs", "median_s", "stops", "p25_s", "p75_s"]
    city = stops.groupby("city").agg(median_s=("service_s", "median"), per_pkg_s=("service_s", lambda s: s.sum()),
                                     pkgs=("n_pkgs", "sum")).reset_index()
    city["per_pkg_s"] = city.per_pkg_s / city.pkgs
    return {"stops": int(len(stops)),
            "intercept_s": round(lin.intercept_, 1), "per_pkg_s": round(lin.coef_[0], 1), "per_litre_s": round(lin.coef_[1], 2),
            "r2_linear": round(r2_score(yte, pred_lin), 3), "r2_gbm": round(r2_score(yte, pred_gbm), 3),
            "mae_gbm_s": round(mean_absolute_error(yte, pred_gbm), 1),
            "by_n": by_n.round(1).to_dict("records"),
            "by_city": city[["city", "median_s", "per_pkg_s"]].round(1).to_dict("records")}


def stations(f):
    st = (f.groupby(["station_code", "city"])
           .agg(routes=("route_id", "size"), stops=("n_stops", "sum"), packages=("n_packages", "sum"),
                service_h=("service_time_h", "sum"), path_km=("path_km", "sum"),
                high_pct=("route_score", lambda s: (s == "High").mean()), util=("capacity_util", "median"))
           .reset_index())
    st["stops_per_service_h"] = st.stops / st.service_h
    st["km_per_pkg"] = st.path_km / st.packages
    st["pkgs_per_stop"] = st.packages / st.stops
    cols = ["station_code", "city", "routes", "stops_per_service_h", "km_per_pkg", "pkgs_per_stop", "util", "high_pct"]
    return st[cols].sort_values("station_code").round(4).to_dict("records")


if __name__ == "__main__":
    f = pd.read_csv(PROC / "route_features.csv")
    pk = pd.read_csv(PROC / "packages.csv")
    out = {"hl": high_vs_low(f), "cap": capacity(f), "svc": service_model(pk, f), "st": stations(f)}
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} if isinstance(v, dict) else len(v) for k, v in out.items()}, indent=1, default=float))
    print("top High vs Low:", out["hl"]["compare"][:5])
    (DASH / "extra.js").write_text("window.ALMRRC_X = " + json.dumps(out, default=float) + ";\n", encoding="utf-8")
    print("wrote dashboard/extra.js")
