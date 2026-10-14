import unittest
import numpy as np

from main import cosine_sim, mean_square_root

class Test(unittest.TestCase):
    def test_bf16_to_f32(self):
        n = np.array([np.uint16(0x3F80)])
        n = np.astype(n, np.uint32)
        n = np.left_shift(n, 16)
        n = n.view(np.float32)
        self.assertEqual(n, 1.0)

    def test_cosine_sim(self):
        a = np.array([1.0, 1.0, 1.0])
        self.assertEqual(cosine_sim(a, a), 1.0)
        self.assertEqual(cosine_sim(a, -a), -1.0)

    def test_msr(self):
        exp = np.array([0.46291005, 0.9258201, 1.38873015])
        res = mean_square_root(np.array([2, 4, 6]))
        self.assertEqual(res.all(), exp.all())


if __name__ == "__main__":
    unittest.main()
