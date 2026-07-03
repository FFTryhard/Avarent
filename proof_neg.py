import argparse
import sys
import time
import requests
import pandas as pd
import numpy as np

# ==========================
# CONFIG
# ==========================
CSV_PATH = "C:/Users/H/Documents/projects/v_output/etf_daily_log_returns.csv"
API_URL = "http://api.avarent.co/v1/client/latest"
API_KEY = "av-live-owner-key"

INITIAL_CAPITAL = 1_000_000
TRANSACTION_COST_BPS = 10
REBALANCE_DAYS = 5

# ==========================
# HELPER: SNAPSHOT PRINTER
# ==========================
def print_snapshot(data: dict):
    sep = "-" * 60
    source   = data.get("source",                "?")
    ts       = data.get("timestamp",             "?")
    regime   = data.get("regime",                "?")
    conf     = data.get("confidence",            0.0)
    stab     = data.get("regime_stability",      0.0)
    age      = data.get("regime_age",            0)
    geo      = data.get("geometry",              0.0)
    ergo     = data.get("ergodicity",            0.0)
    comp     = data.get("complexity",            0.0)
    rationale= data.get("rationale",             "?")
    drivers  = data.get("drivers",               {})

    print(sep)
    print(f"  [{source.upper()}]  {ts}")
    print(sep)
    print(f"  Regime    {regime:<22} Confidence {conf:.0%}")
    print(f"  Stability {stab:.3f}               Age {age} bars")
    print(f"  Geometry  {geo:.3f}               Ergodicity {ergo:.3f}")
    print(f"  Complexity{comp:.3f}")
    print(f"  Rationale {rationale[:60]}")
    if drivers:
        top = "  ".join(f"{k}({v:+.3f})" for k, v in list(drivers.items())[:3])
        print(f"  Drivers   {top}")
    print(sep)

# ==========================
# MAIN EXECUTION
# ==========================
def run_backtest(mode: str):
    print(f"Starting Avarent Engine Backtest in mode: {mode.upper()}")
    
    # Load Data
    print(f"Loading data from {CSV_PATH}...")
    returns = pd.read_csv(CSV_PATH, parse_dates=["Date"], index_col="Date").sort_index()
    prices = (1 + returns).cumprod() * 100
    returns = returns.fillna(0) # Prevent NaN poisoning
    
    # Baseline Strategy: Cross-sectional Momentum
    momentum = prices.pct_change(20)
    volatility = returns.rolling(20).std()
    raw_signal = (momentum / volatility).shift(1).fillna(0)
    
    # Initialize Engines/Variables
    engine = None
    engine_initialized = False
    headers = list(returns.columns)
    
    if mode == "native":
        try:
            # pyrefly: ignore [missing-import]
            import mga_module
            config = mga_module.profile_defaults(mga_module.EngineProfile.GLOBAL_MACRO)
            engine = mga_module.Engine(config)
        except ImportError:
            print("ERROR: mga_module not found. Compile the C++ extension first.")
            sys.exit(1)
            
    # Portfolio States (Tracking both Naive Baseline and Avarent Overlay)
    cash_base = INITIAL_CAPITAL
    cash_overlay = INITIAL_CAPITAL
    pos_base = pd.Series(0.0, index=headers)
    pos_overlay = pd.Series(0.0, index=headers)
    
    hist_base = []
    hist_overlay = []
    
    # Persist the snapshot across daily iterations
    snapshot = None
    
    # Pre-allocate for speed
    len_prices = len(prices)
    
    for i in range(len_prices):
        date = prices.index[i]
        ts_str = date.strftime("%Y-%m-%d")
        current_prices = prices.iloc[i]
        current_returns = returns.iloc[i].tolist()
        
        # ---------------------------------------------------------
        # 1. ENGINE INGESTION
        # ---------------------------------------------------------
        
        if mode == "native":
            if not engine_initialized:
                engine.setHeaders(headers)
                engine_initialized = True
            engine.ingest(current_returns)
            if engine.ready():
                raw_out = engine.get_snapshot()
                # Manually map C++ output to Python dict format for consistency 
                # Collapse the regime directly
                from avarent.regimes import collapse
                snapshot = {
                    "source": "native",
                    "timestamp": ts_str,
                    "regime": collapse(raw_out.regime.current),
                    "confidence": raw_out.regime.confidence,
                    "regime_stability": raw_out.regime.stability,
                    "regime_age": raw_out.regime.age,
                    "geometry": raw_out.stress.geometry,
                    "ergodicity": raw_out.stress.ergodicity,
                    "complexity": raw_out.stress.complexity,
                    "rationale": raw_out.decision.rationale,
                    "drivers": dict(zip(raw_out.structure.axis1_contributors, raw_out.structure.axis1_weights))
                }
                
        elif mode == "api":
            payload = {
                "timestamp": ts_str,
                "headers": headers,
                "values": current_returns
            }
            try:
                resp = requests.post(API_URL, json=payload, headers={"Authorization": f"Bearer {API_KEY}"})
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") != "warming_up":
                    snapshot = data
            except Exception as e:
                print(f"API Error on {ts_str}: {e}")
                sys.exit(1)

        # ---------------------------------------------------------
        # 2. PORTFOLIO UPDATES
        # ---------------------------------------------------------
        val_base = cash_base + (pos_base * current_prices).sum()
        val_overlay = cash_overlay + (pos_overlay * current_prices).sum()
        
        if snapshot is not None:
            # Print snapshot every 50 days to show progress
            if i % 50 == 0:
                print_snapshot(snapshot)
                
            base_sig = raw_signal.iloc[i]
            
            # --- NAIVE MOMENTUM STRATEGY ---
            abs_sum_base = np.abs(base_sig).sum()
            if abs_sum_base > 0:
                target_w_base = base_sig / abs_sum_base
            else:
                target_w_base = base_sig * 0.0
                
            # --- AVARENT OVERLAY STRATEGY ---
            stability = snapshot.get("regime_stability", 0.0)
            regime = snapshot.get("regime", "Compressed")
            
            # Kelly Scaling via Stability
            overlay_sig = base_sig * stability
            
            # Taxonomy Triggers
            if regime == "Stressed":
                # Liquidation
                overlay_sig = overlay_sig * 0.0
            elif regime == "Transitioning":
                # Half-exposure (high uncertainty)
                overlay_sig = overlay_sig * 0.5
            elif regime == "Compressed":
                # Potential breakout forming, hold cash
                overlay_sig = overlay_sig * 0.0
                
            abs_sum_overlay = np.abs(overlay_sig).sum()
            if abs_sum_overlay > 0:
                target_w_overlay = overlay_sig / abs_sum_overlay
            else:
                target_w_overlay = overlay_sig * 0.0

            # --- EXECUTE TRADES (Only on Rebalance Days or Emergency Cuts) ---
            is_rebalance_day = (i % REBALANCE_DAYS == 0)
            is_emergency = (regime == "Stressed") or (stability < 0.2)
            
            # Execute Base
            if is_rebalance_day:
                target_dollars_base = target_w_base * val_base
                current_dollars_base = pos_base * current_prices
                trade_base = target_dollars_base - current_dollars_base
                cost_base = np.abs(trade_base).sum() * (TRANSACTION_COST_BPS / 10000.0)
                cash_base -= cost_base
                pos_base += (trade_base / current_prices)
                cash_base -= trade_base.sum()
                
            # Execute Overlay
            if is_rebalance_day or is_emergency:
                target_dollars_overlay = target_w_overlay * val_overlay
                current_dollars_overlay = pos_overlay * current_prices
                trade_overlay = target_dollars_overlay - current_dollars_overlay
                cost_overlay = np.abs(trade_overlay).sum() * (TRANSACTION_COST_BPS / 10000.0)
                cash_overlay -= cost_overlay
                pos_overlay += (trade_overlay / current_prices)
                cash_overlay -= trade_overlay.sum()
                
        # Re-calc final daily value
        val_base = cash_base + (pos_base * current_prices).sum()
        val_overlay = cash_overlay + (pos_overlay * current_prices).sum()
        
        hist_base.append({"date": date, "portfolio": val_base})
        hist_overlay.append({"date": date, "portfolio": val_overlay})

    # ==========================
    # RESULTS
    # ==========================
    res_base = pd.DataFrame(hist_base).set_index("date")
    res_base["returns"] = res_base["portfolio"].pct_change()
    
    res_overlay = pd.DataFrame(hist_overlay).set_index("date")
    res_overlay["returns"] = res_overlay["portfolio"].pct_change()
    
    def calc_metrics(df):
        std = df["returns"].std()
        sharpe = np.sqrt(252) * (df["returns"].mean() / std) if std > 1e-8 else 0.0
        rolling_max = df["portfolio"].cummax()
        dd = (df["portfolio"] - rolling_max) / rolling_max
        return df["portfolio"].iloc[-1], sharpe, dd.min()

    final_b, sharpe_b, dd_b = calc_metrics(res_base)
    final_o, sharpe_o, dd_o = calc_metrics(res_overlay)
    
    print("\n" + "="*40)
    print("BACKTEST RESULTS".center(40))
    print("="*40)
    print(f"{'Metric':<15} | {'Naive Base':<10} | {'Avarent Overlay':<10}")
    print("-" * 40)
    print(f"{'Final Value':<15} | ${final_b:,.0f} | ${final_o:,.0f}")
    print(f"{'Sharpe Ratio':<15} | {sharpe_b:<10.2f} | {sharpe_o:<10.2f}")
    print(f"{'Max Drawdown':<15} | {dd_b:<10.2%} | {dd_o:<10.2%}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avarent Engine Capacity Proof")
    parser.add_argument("--mode", choices=["native", "api"], default="native", help="Execution mode (native C++ or via FastAPI)")
    args = parser.parse_args()
    
    run_backtest(args.mode)