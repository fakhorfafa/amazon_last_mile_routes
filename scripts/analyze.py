"""Route-level feature engineering, route-quality model, and dashboard data export.

Input  : data/processed/{routes,stops,packages}.csv
Output : data/processed/route_features.csv, dashboard/data.js
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DASH = ROOT / "dashboard"
EARTH_KM = 6371.0


def haversine(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def nearest_neighbor_km(lat, lng):
    """Length of a greedy nearest-neighbour tour starting and ending at index 0 (the station)."""
    n = len(lat)
    visited = np.zeros(n, bool)
    visited[0] = True
    cur, total = 0, 0.0
    for _ in range(n - 1):
        d = haversine(lat[cur], lng[cur], lat, lng)
        d[visited] = np.inf
        nxt = int(np.argmin(d))
        total += d[nxt]
        visited[nxt] = True
        cur = nxt
    return total + haversine(lat[cur], lng[cur], lat[0], lng[0])


def sequence_features(stops):
    rows = []
    for rid, s in stops.sort_values(["route_id", "sequence"]).groupby("route_id", sort=False):
        lat, lng = s["lat"].to_numpy(), s["lng"].to_numpy()
        legs = haversine(lat[:-1], lng[:-1], lat[1:], lng[1:])
        path_km = legs.sum() + haversine(lat[-1], lng[-1], lat[0], lng[0])
        nn_km = nearest_neighbor_km(lat, lng)

        zones = s["zone_id"].dropna().tolist()
        switches = sum(a != b for a, b in zip(zones, zones[1:]))
        n_zones = len(set(zones))
        # A zone is "revisited" when the driver leaves it and comes back later.
        entries = pd.Series([z for i, z in enumerate(zones) if i == 0 or z != zones[i - 1]]).value_counts()
        rows.append({
            "route_id": rid,
            "path_km": path_km,
            "nn_km": nn_km,
            "path_vs_nn": path_km / nn_km if nn_km else np.nan,
            "first_leg_km": legs[0],
            "n_zones": n_zones,
            "zone_switches_per_zone": switches / n_zones if n_zones else np.nan,
            "zones_revisited_pct": (entries > 1).mean() if n_zones else np.nan,
        })
    return pd.DataFrame(rows)


def package_features(packages):
    p = packages.assign(
        volume_cm3=packages.depth_cm * packages.height_cm * packages.width_cm,
        has_tw=packages.tw_start_utc.notna(),
        failed=packages.scan_status != "DELIVERED",
    )
    return p.groupby("route_id").agg(
        n_packages=("package_id", "size"),
        pkg_volume_cm3=("volume_cm3", "sum"),
        tw_pct=("has_tw", "mean"),
        failed_pct=("failed", "mean"),
        service_time_h=("planned_service_time_s", lambda x: x.sum() / 3600),
    ).reset_index()


def build_features():
    routes = pd.read_csv(PROC / "routes.csv")
    stops = pd.read_csv(PROC / "stops.csv")
    packages = pd.read_csv(PROC / "packages.csv")

    n_stops = stops[stops.type == "Dropoff"].groupby("route_id").size().rename("n_stops")
    f = (routes
         .merge(n_stops, on="route_id")
         .merge(package_features(packages), on="route_id")
         .merge(sequence_features(stops), on="route_id"))
    f["capacity_util"] = f.pkg_volume_cm3 / f.executor_capacity_cm3
    f["departure_hour_utc"] = pd.to_datetime(f.departure_time_utc, format="%H:%M:%S").dt.hour
    f["weekday"] = pd.to_datetime(f.date).dt.day_name()
    f.to_csv(PROC / "route_features.csv", index=False)
    return f, stops


MODEL_FEATURES = ["n_stops", "n_packages", "tw_pct", "failed_pct", "service_time_h", "capacity_util",
                  "path_km", "path_vs_nn", "first_leg_km", "n_zones", "zone_switches_per_zone",
                  "zones_revisited_pct", "departure_hour_utc"]


def quality_model(f):
    X = f[MODEL_FEATURES].fillna(0)
    y = (f.route_score == "High").astype(int)
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    model.fit(X, y)
    imp = sorted(zip(MODEL_FEATURES, model.feature_importances_), key=lambda t: -t[1])
    return {"auc": round(roc_auc_score(y, proba), 3), "base_rate": round(y.mean(), 3),
            "importance": [{"feature": k, "value": round(v, 4)} for k, v in imp]}


def histogram(f, col, bins, clip=None):
    out = {}
    edges = np.linspace(*(clip or (f[col].min(), f[col].max())), bins + 1)
    for score, g in f.groupby("route_score"):
        counts, _ = np.histogram(g[col].clip(edges[0], edges[-1]), edges)
        out[score] = (counts / counts.sum()).round(4).tolist()
    return {"edges": edges.round(3).tolist(), "series": out}


def sample_routes(f, stops, city="Boston", per_score=2):
    picks = []
    for score in ("High", "Medium", "Low"):
        g = f[(f.city == city) & (f.route_score == score)].sort_values("n_stops")
        picks += g.iloc[[len(g) // 3, 2 * len(g) // 3]][:per_score].route_id.tolist()
    out = []
    for rid in picks:
        s = stops[stops.route_id == rid].sort_values("sequence")
        r = f[f.route_id == rid].iloc[0]
        out.append({"route_id": rid, "score": r.route_score, "n_stops": int(r.n_stops),
                    "path_km": round(r.path_km, 1), "zones_revisited_pct": round(r.zones_revisited_pct, 3),
                    "points": s[["lat", "lng"]].round(5).values.tolist(),
                    "zones": s.zone_id.fillna("").tolist()})
    return out


def export(f, stops, model):
    by_city = (f.groupby("city").agg(
        routes=("route_id", "size"), stations=("station_code", "nunique"),
        stops=("n_stops", "mean"), packages=("n_packages", "mean"),
        path_km=("path_km", "mean"), service_h=("service_time_h", "mean"),
        high_pct=("route_score", lambda s: (s == "High").mean()),
    ).round(3).reset_index())
    score_city = pd.crosstab(f.city, f.route_score, normalize="index").round(4)
    by_score = f.groupby("route_score")[MODEL_FEATURES].mean().round(3)

    data = {
        "kpi": {
            "routes": int(len(f)), "stops": int(f.n_stops.sum()), "packages": int(f.n_packages.sum()),
            "stations": int(f.station_code.nunique()), "cities": int(f.city.nunique()),
            "date_min": f.date.min(), "date_max": f.date.max(),
            "high_pct": round((f.route_score == "High").mean(), 4),
            "failed_pct": round((f.failed_pct * f.n_packages).sum() / f.n_packages.sum(), 4),
        },
        "score_counts": f.route_score.value_counts().to_dict(),
        "by_city": by_city.to_dict("records"),
        "score_by_city": {c: score_city.loc[c].to_dict() for c in score_city.index},
        "by_score": by_score.to_dict("index"),
        "routes_by_date": f.groupby("date").size().to_dict(),
        "departure_hour": f.groupby("departure_hour_utc").size().to_dict(),
        "hist": {
            "zones_revisited_pct": histogram(f, "zones_revisited_pct", 20, (0, 0.5)),
            "path_vs_nn": histogram(f, "path_vs_nn", 20, (0.6, 1.6)),
            "n_stops": histogram(f, "n_stops", 20, (40, 260)),
        },
        "model": model,
        "sample_routes": sample_routes(f, stops),
    }
    DASH.mkdir(exist_ok=True)
    (DASH / "data.js").write_text("window.ALMRRC = " + json.dumps(data, default=float) + ";\n", encoding="utf-8")


if __name__ == "__main__":
    f, stops = build_features()
    print(f"features: {f.shape}")
    model = quality_model(f)
    print(f"model AUC (5-fold): {model['auc']}  top: {model['importance'][:3]}")
    export(f, stops, model)
    print("wrote dashboard/data.js")
