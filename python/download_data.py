from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from common import ROOT, load_config, sha256_file, write_json


def download():
    config = load_config()
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    archive = folder / "source.zip"
    member = folder / "10_Industry_Portfolios_Daily.csv"
    retrieval_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with urlopen(config["data_url"], timeout=60) as response, archive.open("wb") as target:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            target.write(block)
    with ZipFile(archive) as source:
        names = [name for name in source.namelist() if name.endswith("10_Industry_Portfolios_Daily.csv")]
        if len(names) != 1:
            raise RuntimeError("archive did not contain the expected single daily industry file")
        member.write_bytes(source.read(names[0]))
    return write_json(ROOT / "results" / "download.json", {
        "tier": "diagnostic",
        "url": config["data_url"],
        "retrieval_utc": retrieval_time,
        "archive_sha256": sha256_file(archive),
        "member_name": member.name,
        "member_sha256": sha256_file(member),
        "archive_bytes": archive.stat().st_size,
        "member_bytes": member.stat().st_size
    })


if __name__ == "__main__":
    print(download())
