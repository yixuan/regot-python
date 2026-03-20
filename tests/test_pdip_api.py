import unittest
import numpy as np
import regot


class TestPdipApi(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        n, m = 24, 24
        self.M = rng.random((n, m), dtype=np.float64)
        a = rng.random(n, dtype=np.float64)
        b = rng.random(m, dtype=np.float64)
        self.a = a / a.sum()
        self.b = b / b.sum()

    def _check_result_schema(self, res):
        self.assertTrue(hasattr(res, "niter"))
        self.assertTrue(hasattr(res, "plan"))
        self.assertTrue(hasattr(res, "obj_vals"))
        self.assertTrue(hasattr(res, "mar_errs"))
        self.assertTrue(hasattr(res, "run_times"))
        self.assertGreater(len(res.obj_vals), 0)
        self.assertEqual(len(res.obj_vals), len(res.mar_errs))
        self.assertEqual(len(res.obj_vals), len(res.run_times))
        self.assertEqual(res.plan.shape, self.M.shape)

    def test_pdip_cg_schema_and_accuracy(self):
        res = regot.pdip_cg(self.M, self.a, self.b, 0.1, tol=1e-6, max_iter=200)
        self._check_result_schema(res)
        self.assertLess(float(res.mar_errs[-1]), 1e-5)

    def test_pdip_fp_schema_and_accuracy(self):
        res = regot.pdip_fp(self.M, self.a, self.b, 0.1, tol=1e-6, max_iter=200)
        self._check_result_schema(res)
        self.assertLess(float(res.mar_errs[-1]), 1e-5)


if __name__ == "__main__":
    unittest.main()
