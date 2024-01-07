from pathlib import Path

import numpy as np


def generate_data(data_path: Path, num_steps: int, interval: float = 0.1) -> None:
    x = np.linspace(0, num_steps * interval, num_steps)
    y1 = np.sin(x)
    y2 = np.sin(x * 0.1)

    noise = np.random.normal(0, 0.1, y1.shape)

    y3 = (y1 + noise) * y2

    np.savez(data_path, y1=y1, y2=y2, y3=y3)


generate_data("data.npz", 1000000)
