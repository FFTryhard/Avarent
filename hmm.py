from hmmlearn import hmm
import numpy as np

with open("data.bin", "rb") as f:
    rows = np.fromfile(f, dtype=np.uint64, count=1)[0] 
    cols = np.fromfile(f, dtype=np.uint64, count=1)[0]
    data = np.fromfile(f, dtype=np.float64).reshape(rows, cols)

# gec shape: (N, 3) — columns are G, E, C after warmup
gec = np.array(data)

model = hmm.GaussianHMM(
    n_components=5,
    covariance_type="full",
    n_iter=100,
    tol=1e-3
    )

model.fit(gec)

centroids = model.means_

states = model.predict(gec)

for i in range(model.n_components):
    count = np.sum(states == i)
    print(f"State {i}: {count} observations")

print(centroids)

print("Transition matrix:")
print(model.transmat_)