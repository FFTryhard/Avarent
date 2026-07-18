import sys
import os
import csv
import time

# Add the build directory to sys.path so we can import the compiled C++ extension
build_paths = [
    os.path.abspath("cpp/build/Release"),
    os.path.abspath("cpp/build/Debug"),
    os.path.abspath("cpp/build")
]
for p in build_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

try:
    # pyrefly: ignore [missing-import]
    import mga_module
except ImportError as e:
    print(f"Error: Could not import mga_module. Did you run 'cmake --build build --config Release'?\nDetails: {e}")
    sys.exit(1)

def run():
    print("--- Starting Avarent C++ Engine via Python ---")
    
    # 1. Initialize Engine Configuration
    config = mga_module.profile_defaults(mga_module.EngineProfile.LONG_SHORT_EQUITY)
    engine = mga_module.Engine(config)

    # 2. Open CSV file
    csv_path = "data/etf_daily_log_returns_new_time_period2.csv"
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    print(f"Loading data from {csv_path}...")
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        
        # Read headers
        headers_row = next(reader)
        headers = headers_row[1:] # Skip the Date column
        
        engine.setHeaders(headers)
        
        # Ingest rows
        rows_processed = 0
        start_time = time.time()
        
        for row in reader:
            if not row:
                continue
                
            date_str = row[0]
            # Convert string returns to floats
            try:
                values = [float(x) for x in row[1:] if x.strip()]
            except ValueError:
                continue # Skip bad rows
                
            if not values:
                continue
                
            # Feed to C++ Engine
            engine.ingest(values)
            rows_processed += 1
            
            # Print output if the engine is fully warmed up and produced a result
            if engine.ready():
                snapshot = engine.get_snapshot()
                print("-" * 60)
                print(f"[{date_str}] Regime: {snapshot.regime.current} (Age: {snapshot.regime.age})")
                print(f"Rationale: {snapshot.decision.rationale}")
                
        elapsed = time.time() - start_time
        print("-" * 60)
        print(f"Finished processing {rows_processed} ticks in {elapsed:.3f} seconds.")

if __name__ == "__main__":
    run()
