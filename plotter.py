import sys
import matplotlib.pyplot as plt

plt.ion()

fig, ax = plt.subplots()

xs1, ys1 = [], []
xs2, ys2 = [], []

for line in sys.stdin:
    parts = line.split()

    # skip malformed/empty lines
    if len(parts) != 4:
        continue

    x1, y1, x2, y2 = map(float, parts)

    xs1.append(x1)
    ys1.append(y1)

    xs2.append(x2)
    ys2.append(y2)

    ax.clear()

    ax.plot(xs1, ys1, label='NAV Base')
    ax.plot(xs2, ys2, label='NAV Avarent Overlay')

    ax.legend()

    plt.pause(0.01)

plt.ioff()

print("Input stream ended.")
plt.show(block=True)