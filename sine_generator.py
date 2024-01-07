from pathlib import Path

import numpy as np


def generate_data(data_path: Path, num_steps: int, interval: float = 0.1) -> None:
    x = np.linspace(0, num_steps * interval, num_steps)
    y = np.sin(x) + np.random.normal(0, 0.1, x.shape)

    np.savez(data_path, y=y)


generate_data("data.npz", 1000000)
