import json
import os
import shutil
import subprocess
import sys

from common import ROOT, hash_verified, load_config, sha256_file, write_json
from gates import run_gates
from protocol import freeze_protocol, verify_rebuild
from real_analysis import run_real_analysis, verify_protocol


def validate_existing_protocol(validation):
    protocol = verify_protocol()
    if protocol["download"]["archive_sha256"] != sha256_file(ROOT / "data" / "source.zip"):
        raise RuntimeError("current source archive does not match the frozen protocol")
    if protocol["download"]["member_sha256"] != sha256_file(ROOT / "data" / "10_Industry_Portfolios_Daily.csv"):
        raise RuntimeError("current data member does not match the frozen protocol")
    if float(validation["G1"]["selected_lambda"]) != float(protocol["selected_lambda"]):
        raise RuntimeError("re-run gates selected a different forgetting factor than the frozen protocol")
    if not all(validation[key]["pass"] for key in ("E1", "G1", "G2", "G3", "G4")):
        raise RuntimeError("a re-run pre-test gate failed")
    return protocol


def compile_documents():
    tectonic = os.environ.get("CODEX_TECTONIC_PATH") or shutil.which("tectonic")
    latex = shutil.which("pdflatex")
    if not latex and os.name == "nt":
        candidates = (
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "MiKTeX", "miktex", "bin", "x64", "pdflatex.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "MiKTeX", "miktex", "bin", "x64", "pdflatex.exe"),
            os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Local", "Programs", "MiKTeX", "miktex", "bin", "x64", "pdflatex.exe"),
        )
        latex = next((candidate for candidate in candidates if candidate and os.path.isfile(candidate)), None)
    paper = ROOT / "paper"
    if tectonic:
        try:
            for source in ("main.tex", "supplement.tex"):
                subprocess.run([tectonic, "-o", str(paper), source], cwd=paper, check=True)
            return
        except subprocess.CalledProcessError:
            if not latex:
                raise
    if not latex:
        raise RuntimeError("Install Tectonic or a LaTeX distribution that provides pdflatex before building the paper PDFs")
    for source in ("main.tex", "supplement.tex"):
        for _ in range(2):
            subprocess.run(
                [latex, "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(paper), source],
                cwd=paper,
                check=True,
            )


def run_pipeline():
    plan = ROOT / "results" / "analysis_plan.json"
    plan_hash = (ROOT / "results" / "analysis_plan.sha256").read_text(encoding="ascii").strip()
    hash_verified(plan, plan_hash)
    config = load_config()
    if sha256_file(ROOT / "config.json") != protocol_config_hash_if_frozen():
        raise RuntimeError("config changed after protocol freeze")
    fresh_validation = run_gates(config)
    existing = (ROOT / "results" / "protocol.json").exists()
    if existing:
        validate_existing_protocol(fresh_validation)
        validation = json.loads((ROOT / "results" / "validation.json").read_text(encoding="utf-8"))
    else:
        write_json(ROOT / "results" / "validation.json", fresh_validation)
        fingerprint_sha256 = verify_rebuild()
        freeze_protocol(fresh_validation, fingerprint_sha256)
        validation = fresh_validation
    protocol = verify_protocol()
    if sha256_file(ROOT / "results" / "validation.json") != protocol["validation_sha256"]:
        raise RuntimeError("validation record does not match the frozen protocol")
    if sha256_file(ROOT / "results" / "reproducibility.json") != protocol["reproducibility_sha256"]:
        raise RuntimeError("reproducibility record does not match the frozen protocol")
    # Preserve the submitted frozen study file byte-for-byte.  The full runner
    # records a non-deterministic runtime field, so retain the original record
    # after recomputing and use that frozen record for figures and documents.
    study_path = ROOT / "results" / "study.json"
    frozen_study_bytes = study_path.read_bytes() if study_path.exists() else None
    frozen_study_sha = sha256_file(study_path) if study_path.exists() else None
    try:
        run_real_analysis()
    finally:
        if frozen_study_bytes is not None:
            study_path.write_bytes(frozen_study_bytes)
    if frozen_study_sha is not None and sha256_file(study_path) != frozen_study_sha:
        raise RuntimeError("the frozen study output was not restored byte-for-byte")
    study = json.loads(study_path.read_text(encoding="utf-8"))
    from plots import create_figures
    from documents import generate_documents
    create_figures()
    generate_documents()
    compile_documents()
    return {"tier": "frozen", "H1": study["H1"]["verdict"], "H2": study["H2"]["verdict"]}


def protocol_config_hash_if_frozen():
    path = ROOT / "results" / "protocol.json"
    if not path.exists():
        return sha256_file(ROOT / "config.json")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    return protocol["config_sha256"]


if __name__ == "__main__":
    try:
        outcome = run_pipeline()
        (ROOT / "results" / "gate_failure.json").unlink(missing_ok=True)
        print(json.dumps(outcome, indent=2))
    except Exception as error:
        write_json(ROOT / "results" / "gate_failure.json", {"tier": "diagnostic", "error": f"{type(error).__name__}: {error}"})
        raise
