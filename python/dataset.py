import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from common import ROOT, load_config, sha256_file, write_json


def iter_daily_rows(member_path, stop_date=None, portfolio_count=None, missing_codes=None, return_scale=None):
    config = load_config()
    expected_count = int(portfolio_count if portfolio_count is not None else config["portfolio_count"])
    missing = config["missing_codes"] if missing_codes is None else missing_codes
    scale = config["return_scale"] if return_scale is None else float(return_scale)
    with Path(member_path).open("r", encoding="latin-1", newline="") as stream:
        lines = csv.reader(stream)
        in_section = False
        labels = None
        for row in lines:
            if not row:
                continue
            first = row[0].strip()
            if first == "Average Value Weighted Returns -- Daily":
                in_section = True
                continue
            if not in_section:
                continue
            if labels is None:
                if len(row) == expected_count + 1 and row[0].strip() == "":
                    labels = [item.strip() for item in row[1:]]
                    if len(set(labels)) != expected_count:
                        raise RuntimeError("portfolio labels are not unique")
                continue
            if not first.isdigit() or len(first) != 8:
                if first.startswith("Average Equal Weighted"):
                    break
                continue
            if stop_date is not None and first > stop_date.strftime("%Y%m%d"):
                break
            date = datetime.strptime(first, "%Y%m%d").date()
            if len(row) != expected_count + 1:
                continue
            values = np.asarray([float(item) for item in row[1:]], dtype=np.float64)
            if np.any(np.isin(values, missing)) or not np.all(np.isfinite(values)):
                continue
            simple = values * scale
            if np.any(simple <= -1.0):
                raise RuntimeError("a daily simple return is at or below -100 percent")
            yield date.isoformat(), labels, np.log1p(simple), simple.mean()
        if labels is None:
            raise RuntimeError("value-weighted daily header was not found")


def load_panel(stop_date=None, member_path=None, portfolio_count=None):
    path = Path(member_path) if member_path is not None else ROOT / "data" / "10_Industry_Portfolios_Daily.csv"
    expected_count = int(portfolio_count if portfolio_count is not None else load_config()["portfolio_count"])
    dates = []
    log_returns = []
    equal_weight = []
    labels = None
    for date, current_labels, log_values, equal_value in iter_daily_rows(path, stop_date, expected_count):
        if labels is None:
            labels = current_labels
        dates.append(date)
        log_returns.append(log_values)
        equal_weight.append(equal_value)
    if len(labels or []) != expected_count:
        raise RuntimeError(f"daily header did not contain {expected_count} portfolios")
    return np.asarray(dates, dtype="datetime64[D]"), np.asarray(log_returns, dtype=np.float64), np.asarray(equal_weight, dtype=np.float64), labels


def create_derived_file():
    config = load_config()
    folder = ROOT / "data"
    member = folder / "10_Industry_Portfolios_Daily.csv"
    derived = folder / "returns.csv"
    labels = None
    with derived.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        for date, current_labels, log_values, equal_value in iter_daily_rows(member):
            if labels is None:
                labels = current_labels
                writer.writerow(["date", *[f"log_{item}" for item in labels], "equal_weight_simple"])
            writer.writerow([date, *[format(float(value), ".17g") for value in log_values], format(float(equal_value), ".17g")])
    download = json.loads((ROOT / "results" / "download.json").read_text(encoding="utf-8"))
    metadata = {
        "tier": "frozen",
        "url": config["data_url"],
        "retrieval_utc": download["retrieval_utc"],
        "archive_sha256": sha256_file(folder / "source.zip"),
        "member_sha256": sha256_file(member),
        "derived_returns_sha256": sha256_file(derived),
        "member": member.name,
        "derived_returns": derived.name,
        "labels": labels
    }
    write_json(ROOT / "results" / "data_snapshot.json", metadata)
    return derived


def load_derived(stop_date=None):
    path = ROOT / "data" / "returns.csv"
    dates = []
    logs = []
    equal_weight = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) != 12:
            raise RuntimeError("derived return file must contain date, ten assets, and equal-weight return")
        for row in reader:
            if stop_date is not None and row[0] > stop_date:
                break
            dates.append(row[0])
            logs.append([float(value) for value in row[1:11]])
            equal_weight.append(float(row[11]))
    return np.asarray(dates, dtype="datetime64[D]"), np.asarray(logs), np.asarray(equal_weight)
