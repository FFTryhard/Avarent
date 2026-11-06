import json
import platform
import subprocess
import sys
import sysconfig
import hashlib
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import market_core

from common import ROOT, hash_verified, load_config, sha256_file, write_json


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def run_build():
    subprocess.run(["cmake", "--build", str(ROOT / "build"), "--config", "Release"], cwd=ROOT, check=True)


def fingerprint(target):
    subprocess.run([sys.executable, str(ROOT / "python" / "core_fingerprint.py"), str(target)], cwd=ROOT, check=True)


def compiler_metadata():
    cache = ROOT / "build" / "CMakeCache.txt"
    entries = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" in line and not line.startswith("//") and not line.startswith("#"):
                key, value = line.split("=", 1)
                entries[key.split(":", 1)[0]] = value
    compiler = entries.get("CMAKE_CXX_COMPILER", "unknown")
    compiler_id = entries.get("CMAKE_CXX_COMPILER_ID", "unknown")
    compiler_version = entries.get("CMAKE_CXX_COMPILER_VERSION", "unknown")
    return {"compiler": f"{compiler_id} {compiler_version}", "compiler_path": compiler, "cpp_standard": "C++17"}


def verify_rebuild():
    with tempfile.TemporaryDirectory(prefix="rhfs-build-check-") as temporary:
        first = Path(temporary) / "first.bin"
        second = Path(temporary) / "second.bin"
        run_build()
        fingerprint(first)
        run_build()
        fingerprint(second)
        first_bytes = first.read_bytes()
        second_bytes = second.read_bytes()
        if first_bytes != second_bytes:
            raise RuntimeError("C++ output changed across identical rebuilds")
        return hashlib.sha256(first_bytes).hexdigest()


def collect_reproducibility(fingerprint_sha256):
    cmake = subprocess.check_output(["cmake", "--version"], text=True).splitlines()[0]
    compiler = compiler_metadata()
    return {
        "tier": "frozen",
        **compiler,
        "cmake": cmake,
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "python_abi": sysconfig.get_config_var("SOABI"),
        "platform": platform.platform(),
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
        "pandas": package_version("pandas"),
        "matplotlib": package_version("matplotlib"),
        "scikit_learn": package_version("scikit-learn"),
        "pybind11": package_version("pybind11"),
        "eigen": "not used",
        "executable_path": str(Path(market_core.__file__).resolve().relative_to(ROOT)),
        "executable_sha256": sha256_file(market_core.__file__),
        "build_command": "cmake --build build --config Release",
        "byte_for_byte_rebuild_verified": True,
        "rebuild_fingerprint_sha256": fingerprint_sha256
    }


def freeze_protocol(validation, fingerprint_sha256):
    if (ROOT / "results" / "protocol.json").exists():
        raise RuntimeError("an existing frozen protocol cannot be replaced")
    plan_path = ROOT / "results" / "analysis_plan.json"
    plan_hash = (ROOT / "results" / "analysis_plan.sha256").read_text(encoding="ascii").strip()
    hash_verified(plan_path, plan_hash)
    config = load_config()
    config_hash = sha256_file(ROOT / "config.json")
    download = json.loads((ROOT / "results" / "download.json").read_text(encoding="utf-8"))
    archive_path = ROOT / "data" / "source.zip"
    member_path = ROOT / "data" / "10_Industry_Portfolios_Daily.csv"
    if sha256_file(archive_path) != download["archive_sha256"] or sha256_file(member_path) != download["member_sha256"]:
        raise RuntimeError("downloaded data hashes do not match the recorded archive")
    if download["url"] != config["data_url"]:
        raise RuntimeError("download source changed after retrieval")
    reproducibility = collect_reproducibility(fingerprint_sha256)
    write_json(ROOT / "results" / "reproducibility.json", reproducibility)
    references = json.loads((ROOT / "results" / "references.json").read_text(encoding="utf-8"))
    dev_log = json.loads((ROOT / "results" / "dev_log.json").read_text(encoding="utf-8"))
    attempts = int(dev_log["full_gate_attempts"])
    if attempts > json.loads(plan_path.read_text(encoding="utf-8"))["rules"]["maximum_attempts"]:
        raise RuntimeError("development attempt limit was exceeded")
    protocol = {
        "tier": "frozen",
        "protocol_version": 1,
        "config": config,
        "config_sha256": config_hash,
        "analysis_plan": json.loads(plan_path.read_text(encoding="utf-8")),
        "analysis_plan_sha256": plan_hash,
        "download": {key: download[key] for key in ("url", "retrieval_utc", "archive_sha256", "member_sha256")},
        "validation_sha256": sha256_file(ROOT / "results" / "validation.json"),
        "reproducibility_sha256": sha256_file(ROOT / "results" / "reproducibility.json"),
        "selected_lambda": float(validation["G1"]["selected_lambda"]),
        "development_attempts": attempts,
        "verdict_rules": {
            "H1": {
                "supported": "all four conditions pass",
                "unsupported": "all adjusted block-shuffle probabilities are at least the pre-onset placebo threshold",
                "otherwise": "inconclusive"
            },
            "H2": {
                "support_point_threshold": config["h2_support_point_threshold"],
                "confidence_level": config["h2_confidence_level"],
                "supported": "adjusted lower interval is above zero and point increment reaches threshold",
                "unsupported": "adjusted upper interval is below threshold",
                "otherwise": "inconclusive"
            }
        },
        "target_definition": "Equal-weight daily simple-return index drawdown crossing the configured threshold; positive label marks an onset in the next configured trading-day horizon; event-cluster risk dates are removed.",
        "feature_sets": {
            "M0": ["annualized rolling volatility", "drawdown", "mean absolute rolling correlation deviation"],
            "M0_F": ["M0", "largest singular-value amplification over the primary horizon"],
            "M1": ["M0", "past-only AR(1) coefficient", "past-only absorption ratio"],
            "M1_F": ["M1", "largest singular-value amplification over the primary horizon"]
        },
        "fairness": "All models use a shared forecast-anchor mask, expanding training windows, training-only feature scaling, and a fixed logistic penalty. The richer comparator adds established early-warning quantities; F is the sole added feature.",
        "references": references,
        "development_deviations": [entry for entry in dev_log["entries"] if entry.get("stage") == "chronology deviation"],
        "gate_summary": {key: validation[key] for key in ("U", "E1", "G1", "G2", "G3", "G4")}
    }
    write_json(ROOT / "results" / "protocol.json", protocol)
    protocol_hash = sha256_file(ROOT / "results" / "protocol.json")
    (ROOT / "results" / "protocol.sha256").write_text(protocol_hash + "\n", encoding="ascii", newline="\n")
    write_json(ROOT / "results" / "protocol_hash.json", {"tier": "frozen", "protocol_sha256": protocol_hash})
    hash_verified(ROOT / "results" / "protocol.json", protocol_hash)
    return protocol
