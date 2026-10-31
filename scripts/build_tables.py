"""Flatten the nested ALMRRC 2021 JSON files into tidy CSV tables.

Input  : data/raw/{route_data,package_data,actual_sequences,invalid_sequence_scores}.json
Output : data/processed/{routes,stops,packages}.csv
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

CITY = {"LA": "Los Angeles", "SE": "Seattle", "AU": "Austin", "CH": "Chicago", "BO": "Boston"}


def load(name):
    with open(RAW / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def build_routes_and_stops():
    route_data = load("route_data")
    sequences = load("actual_sequences")
    invalid = load("invalid_sequence_scores")

    routes, stops = [], []
    for rid, r in route_data.items():
        order = sequences.get(rid, {}).get("actual", {})
        routes.append({
            "route_id": rid,
            "station_code": r["station_code"],
            "city": CITY.get(r["station_code"][1:3], r["station_code"]),
            "date": r["date_YYYY_MM_DD"],
            "departure_time_utc": r["departure_time_utc"],
            "executor_capacity_cm3": r["executor_capacity_cm3"],
            "route_score": r["route_score"],
            "invalid_sequence_score": invalid.get(rid),
        })
        for sid, s in r["stops"].items():
            zone = s.get("zone_id")
            stops.append({
                "route_id": rid,
                "stop_id": sid,
                "lat": s["lat"],
                "lng": s["lng"],
                "type": s["type"],
                "zone_id": zone if isinstance(zone, str) else None,
                "sequence": order.get(sid),
            })
    return pd.DataFrame(routes), pd.DataFrame(stops)


def build_packages():
    package_data = load("package_data")
    rows = []
    for rid, route in package_data.items():
        for sid, pkgs in route.items():
            for pid, p in pkgs.items():
                tw, dim = p["time_window"], p["dimensions"]
                rows.append((
                    rid, sid, pid, p["scan_status"],
                    tw.get("start_time_utc"), tw.get("end_time_utc"),
                    p["planned_service_time_seconds"],
                    dim["depth_cm"], dim["height_cm"], dim["width_cm"],
                ))
    cols = ["route_id", "stop_id", "package_id", "scan_status", "tw_start_utc", "tw_end_utc",
            "planned_service_time_s", "depth_cm", "height_cm", "width_cm"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ("tw_start_utc", "tw_end_utc"):
        df[c] = df[c].where(df[c].astype(str) != "nan")
    return df


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    routes, stops = build_routes_and_stops()
    routes.to_csv(OUT / "routes.csv", index=False)
    stops.to_csv(OUT / "stops.csv", index=False)
    print(f"routes: {len(routes):,}  stops: {len(stops):,}")
    packages = build_packages()
    packages.to_csv(OUT / "packages.csv", index=False)
    print(f"packages: {len(packages):,}")
