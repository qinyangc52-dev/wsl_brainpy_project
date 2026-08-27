from __future__ import annotations

import math


class LegacyRNG:
    """Bit-compatible implementation of random.c for single-process runs."""

    M0 = 455329277
    M1 = 282074
    PW2A = 1073741823
    PW2B = 1152921504606846975
    RPW2B = 1.11022302462515654e-16
    RPW2C = 2.77555756156289135e-17
    BMAX = 32
    SKIP = 10

    def __init__(self, seed: int, seed2: int = 0):
        self.state = 1060485742695258666
        self._flip = False
        self._rho = 0.0
        self._phi = 0.0
        self.init_random(seed, seed2)

    def init_random(self, seed: int, seed2: int = 0) -> None:
        if seed >> self.BMAX == 0:
            if seed == 0:
                raise ValueError("A deterministic LegacyRNG requires a non-zero seed")
            b0 = (2 + (seed << 2)) & self.PW2A
            b1 = ((seed >> 28) ^ seed2) & self.PW2A
            self.state = b0 + (b1 << 30)
            for _ in range(self.SKIP):
                self.uniform()
        else:
            if seed2 != 0:
                raise ValueError("seed2 must be zero when seed >= 2**32")
            self.state = seed

    def random_seed(self) -> int:
        return self.state

    def push_random(self, state: int) -> None:
        self.state = int(state)

    def uniform(self) -> float:
        b0 = self.state & self.PW2A
        b1 = (self.state >> 30) & self.PW2A
        self.state = (b0 * self.M0 + ((b0 * self.M1 + b1 * self.M0) << 30)) & self.PW2B
        return self.RPW2C + self.RPW2B * (self.state >> 7)

    def normal(self) -> float:
        if not self._flip:
            self._flip = True
            self._rho = math.sqrt(-2.0 * math.log(self.uniform()))
            self._phi = 2.0 * math.pi * self.uniform()
            return self._rho * math.cos(self._phi)
        self._flip = False
        return self._rho * math.sin(self._phi)

    def select_weighted(self, weights) -> int:
        total = 0.0
        for value in weights:
            if value < 0:
                raise ValueError("Selection weights must be non-negative")
            total += float(value)
        target = total * self.uniform()
        index = 0
        while index < len(weights) - 1 and target > weights[index]:
            target -= float(weights[index])
            index += 1
        return index

    def partial_permutation(self, values, count: int) -> None:
        size = len(values)
        for i in range(count):
            j = int(math.floor((size - i) * self.uniform()))
            values[i], values[i + j] = values[i + j], values[i]

