import sys
import matplotlib.pyplot as plt

plt.ion()

fig, ax = plt.subplots()

xs_g, ys_g = [], []
xs_e, ys_e = [], []
xs_c, ys_c = [], []

for line in sys.stdin:
    parts = line.split()

    # Expect 6 numbers per line:
    # x_g y_g x_e y_e x_c y_c
    if len(parts) != 6:
        continue

    x_g, y_g, x_e, y_e, x_c, y_c = map(float, parts)

    xs_g.append(x_g)
    ys_g.append(y_g)

    xs_e.append(x_e)
    ys_e.append(y_e)

    xs_c.append(x_c)
    ys_c.append(y_c)

    ax.clear()

    ax.plot(xs_g, ys_g, label='G')  #plot is line, scatter is raw
    ax.plot(xs_e, ys_e, label='E')
    ax.plot(xs_c, ys_c, label='C')

    ax.legend()
    ax.grid(True)

    plt.pause(0.01)

plt.ioff()

print("Input stream ended.")
plt.show(block=True)