import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from common import ROOT, sha256_file, write_json
from dataset import iter_daily_rows


URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/17_Industry_Portfolios_daily_CSV.zip"
COUNT = 17
DESTINATION = ROOT / "data" / "exploratory" / "17_industry"


def download_secondary():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    archive = DESTINATION / "source.zip"
    member = DESTINATION / "17_Industry_Portfolios_Daily.csv"
    derived = DESTINATION / "returns.csv"
    retrieved = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with urlopen(URL, timeout=60) as response, archive.open("wb") as target:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            target.write(block)
    with ZipFile(archive) as source:
        members = [name for name in source.namelist() if name.endswith("17_Industry_Portfolios_Daily.csv")]
        if len(members) != 1:
            raise RuntimeError("the 17-industry archive did not contain one expected daily CSV")
        member.write_bytes(source.read(members[0]))
    labels = None
    row_count = 0
    with derived.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        for date, current_labels, log_values, equal_value in iter_daily_rows(member, portfolio_count=COUNT):
            if labels is None:
                labels = current_labels
                writer.writerow(["date", *[f"log_{item}" for item in labels], "equal_weight_simple"])
            writer.writerow([date, *[format(float(value), ".17g") for value in log_values], format(float(equal_value), ".17g")])
            row_count += 1
    snapshot = {
        "tier": "post_freeze_exploratory",
        "source": "Kenneth R. French Data Library",
        "url": URL,
        "retrieval_utc": retrieved,
        "archive_sha256": sha256_file(archive),
        "member": member.name,
        "member_sha256": sha256_file(member),
        "derived_returns": derived.name,
        "derived_returns_sha256": sha256_file(derived),
        "labels": labels,
        "row_count": row_count,
        "redistribution": "The upstream archive is not included; its source terms remain applicable."
    }
    write_json(ROOT / "results" / "exploratory" / "secondary_data_snapshot.json", snapshot)
    return snapshot


if __name__ == "__main__":
    print(json.dumps(download_secondary(), indent=2))
