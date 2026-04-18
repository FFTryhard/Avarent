import threading
import time
import sys
import pandas as pd
sys.path.append(r"C:/Users/H/Documents/cpp/MGA/cpp/build")  #for the .pyd
import mga_module
import os
from dotenv import load_dotenv

load_dotenv()

HEADERS = ["GLD","HYG","IWM","QQQ","SPY","TLT","XLE","XLF","XLK","XLV"]
CSV_PATH = os.getenv("CSV_PATH")

class EngineService:
    def __init__(self):
        self.lock = threading.Lock()

        self.engine = mga_module.Engine()
        self.engine.setHeaders(HEADERS)

        self.latest_output = None
        self.running = False

    def ingest_row(self, timestamp, row):
        with self.lock:
            self.engine.ingest(row)

            if self.engine.ready():
                out = self.engine.getOutput()

                self.latest_output = {
            "timestamp": str(timestamp),
            "regime": out.regime,
            "description": out.description,
            "action": out.action,
            "G": out.G,
            "E": out.E,
            "C": out.C,
            "variable_k1_weights": out.variable_k1_weights,
            "variables": out.variables,
            "variable_weights": out.variable_weights,
            "dominant_eig_projection_coordinates_3dim": out.dominant_eig_projection_coordinates_3dim,
            "eigen_axis1_contributors": out.eigen_axis1_contributors,
            "eigen_axis2_contributors": out.eigen_axis2_contributors,
            "eigen_axis3_contributors": out.eigen_axis3_contributors,
            "eigen_axis1_weights": out.eigen_axis1_weights,
            "eigen_axis2_weights": out.eigen_axis2_weights,
            "eigen_axis3_weights": out.eigen_axis3_weights,
            "structural_stability": out.structural_stability,
                }

    def get_latest(self):
        with self.lock:
            return self.latest_output


service = EngineService()

def start_csv_loop():
    df = pd.read_csv(CSV_PATH)

    def run():
        service.running = True

        for i in range(len(df)):
            timestamp = df.iloc[i, 0]
            row = df.iloc[i, 1:].values.tolist()

            service.ingest_row(timestamp, row)

            time.sleep(0.01)   # simulate live updates

    thread = threading.Thread(target=run, daemon=True)
    thread.start()