import sys

import numpy as np

import market_core


def main():
    seed = np.random.default_rng(20261106)
    returns = np.ascontiguousarray(seed.normal(0.0, 0.01, size=(4096, 10)), dtype=np.float64)
    mu = np.full(10, -10.0)
    phi = np.linspace(0.82, 0.97, 10)
    q = np.linspace(0.05, 0.35, 10)
    output = market_core.run_core(returns, mu, phi, q, 0.998, 1e-8, 4.934802200544679, -1.2704, 8.0, 10.0, 0.05, 6.0, 252, False, 10.0)
    with open(sys.argv[1], "wb") as stream:
        for key in sorted(output):
            value = np.ascontiguousarray(output[key])
            stream.write(key.encode("ascii"))
            stream.write(value.dtype.str.encode("ascii"))
            stream.write(np.asarray(value.shape, dtype=np.int64).tobytes())
            stream.write(value.tobytes())


if __name__ == "__main__":
    main()
