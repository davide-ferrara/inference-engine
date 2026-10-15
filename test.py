import math
import unittest

import numpy as np


class Test(unittest.TestCase):
    def test_bf16_to_f32(self):
        n = np.array([np.uint16(0x3F80)])
        n = np.astype(n, np.uint32)
        n = np.left_shift(n, 16)
        n = n.view(np.float32)
        self.assertEqual(n, 1.0)

    def test_rope(self):
        # Test with even len(a)
        theta = math.radians(90)
        a = np.array([1, 2, 3, 4])
        split = int(len(a) / 2)
        x = a[:split]
        y = a[split:]

        x_prime = np.zeros(split)
        y_prime = np.zeros(split)
        for i in range(len(x_prime)):
            x_prime[i] = (x[i] * math.cos(theta)) - (y[i] * math.sin(theta))
        for i in range(len(y_prime)):
            y_prime[i] = (x[i] * math.sin(theta)) + (y[i] * math.cos(theta))

        print(np.concatenate((x_prime, y_prime)))


if __name__ == "__main__":
    unittest.main()
