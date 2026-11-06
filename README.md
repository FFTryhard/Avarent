# Returns-only market fragility audit

I test whether a returns-only market-state summary contains surrogate-resistant structure and whether a local susceptibility feature improves forecasts of severe drawdowns. The frozen result is deliberately narrow: fragility is the only signal to pass its surrogate comparison, while adding it does not meet the practical H2 forecast threshold. I make no trading or investment-performance claim.

## Reproduce from the repository root

These commands assume a fresh clone and that every command is run from the repository root. The two French Library archives are fetched by script and remain outside Git.

1. Install Python requirements and build the C++ extension:

   ```powershell
   python -m pip install -r requirements.txt
   $pybindDir = python -m pybind11 --cmakedir
   cmake -S . -B build -Dpybind11_DIR="$pybindDir"
   cmake --build build --config Release
   ctest --test-dir build -C Release --output-on-failure
   ```

   On systems using another C++17 compiler, omit the `pybind11_DIR` option and use the platform's normal CMake generator.

2. Fetch the ten- and 17-industry data from the repository root:

   ```powershell
   python python/download_data.py
   python python/download_secondary.py
   ```

   The first script fetches the frozen ten-industry archive and writes its retrieval record. The second fetches the 17-industry archive used only in the exploratory aggregation check. Neither script adds raw French Library data to the repository.

3. Re-run the separately logged exploratory addendum:

   ```powershell
   python python/exploratory.py --stage all
python python/f_surrogate_precision.py
   ```

The exploratory runner verifies frozen input hashes before and after its runs and writes new summaries only under `results/exploratory/`. The F precision runner then uses its separately hashed post-freeze plan to run 4,999 F shuffles per comparison and updates only the exploratory F results; it leaves the frozen protocol, study JSON, thresholds, and verdicts unchanged.

4. Run the frozen pipeline from the root:

   ```powershell
   python python/pipeline.py
   ```

   The runner checks the predeclared plan, protocol, source archive, and data-member hashes before analysis. It restores `results/study.json` byte-for-byte after recomputing the frozen analysis, then regenerates figures and LaTeX sources and compiles both PDFs. The pipeline prefers Tectonic and falls back to `pdflatex`; install either Tectonic or a LaTeX distribution with `pdflatex` on `PATH`.

To regenerate only the paper from saved results, run `python python/plots.py` and `python python/documents.py`, then from the root compile with `tectonic -o paper paper/main.tex` and `tectonic -o paper paper/supplement.tex`. With `pdflatex`, use `Push-Location paper`, run `pdflatex -interaction=nonstopmode -halt-on-error main.tex` twice and then do the same twice for `supplement.tex`, then run `Pop-Location`.

The French Library may revise historical series. The frozen pipeline checks the archive and extracted-member SHA-256 values recorded in `results/protocol.json`; if the source has changed, it stops and reports the mismatch instead of silently changing the frozen analysis. For exact reproduction, use the archived source snapshot matching those recorded hashes. The downloader records new hashes and retrieval dates for each fetch.

## Post-freeze exploratory work

The exploratory addendum has a hashed analysis plan, two hashed extensions, and matching amendment entries. It covers the two zero-return treatments, a factor-aligned direction null, autocorrelation-aware inference for the frozen Haar percentile, the H2 threshold/horizon/penalty grid, EWMA and fixed-alert metrics, refit-aware uncertainty, a block-length comparison, subperiod fragility surrogates, forgetting-factor sensitivity, and the 17-industry aggregation check. None of these analyses changes the frozen protocol, `results/study.json`, or the primary verdicts. The 17-industry portfolio is not an independent replication because it aggregates largely the same U.S. stock universe.

## Repository contents and licenses

- `core/` and `python/` contain the implementation and reproducibility scripts.
- `paper/` contains the manuscript sources, figures, and compiled PDFs.
- `results/` contains the frozen derived study record, validation and protocol records, hashes, and post-freeze summaries. `results/study.json` contains derived model outputs and summaries, not raw French Library return rows.
- `data/` is excluded from Git and is populated by the download scripts.

I license source code under the MIT License in `LICENSE`. I license my paper, figures, and derived results under CC BY 4.0 as described in `LICENSE-PAPER.md`. The Kenneth R. French Data Library series remains subject to its own terms; I do not sublicense or redistribute those source data. See `DATA_NOTICE.md` for attribution and retrieval details.

## Source data

I use the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). The frozen ten-industry archive and derived-panel hashes appear in the paper's reproducibility section; the 17-industry archive and derived-panel hashes are in `results/exploratory/secondary_data_snapshot.json`.
