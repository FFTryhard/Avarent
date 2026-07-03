"""
regimes.py
Collapse the engine's 16-regime taxonomy into 5 actionable labels
and format the clean API payload.
"""

def format_snapshot(timestamp: str, out) -> dict:
    # --- Drivers: top-5 PC1 contributors by absolute weight ---
    variables = list(out.structure.input_variables)
    weights   = list(out.structure.pc1_weights_smoothed)

    if variables and len(variables) == len(weights):
        pairs = sorted(zip(variables, weights), key=lambda x: abs(x[1]), reverse=True)[:5]
        drivers = {name: round(w, 4) for name, w in pairs}
    else:
        drivers = {}

    # --- Confidence / stability: -1 means engine not yet warm enough ---
    confidence = out.regime.confidence
    stability  = out.regime.stability
    confidence = round(confidence, 4) if confidence >= 0 else None
    stability  = round(stability,  4) if stability  >= 0 else None

    return {
        "timestamp":        timestamp,
        "regime":           out.regime.current,
        "confidence":       confidence,
        "geometry":         round(out.stress.geometry,   4),
        "ergodicity":       round(out.stress.ergodicity, 4),
        "complexity":       round(out.stress.complexity, 4),
        "drivers":          drivers,
        "regime_age":       out.regime.age,
        "regime_stability": stability,
        "rationale":        out.decision.rationale,
    }
