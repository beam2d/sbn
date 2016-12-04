from unittest import TestCase

import numpy as np

from sbn.util import cached_property, KahanSum


class TestCachedProperty(TestCase):

    def test_cache(self):
        count = [0]

        class A(object):
            @cached_property
            def x(self):
                count[0] += 1
                return count[0]

        a = A()
        self.assertEqual(a.x, 1)
        self.assertEqual(a.x, 1)
        self.assertEqual(a._x, 1)

        b = A()
        self.assertEqual(b.x, 2)
        self.assertEqual(b.x, 2)
        self.assertEqual(b._x, 2)


class TestKahanSum(TestCase):

    def test_scalar_sum(self):
        arr = np.random.rand(1000)
        kahan = KahanSum()
        for x in arr:
            kahan.add(x)
        self.assertAlmostEqual(kahan.sum, arr.sum())
        self.assertAlmostEqual(kahan.mean, arr.mean())

    def test_array_sum(self):
        arr = np.random.rand(1000, 4, 5)
        kahan = KahanSum()
        for x in arr:
            kahan.add(x)
        np.testing.assert_almost_equal(kahan.sum, arr.sum(axis=0))
        np.testing.assert_almost_equal(kahan.mean, arr.mean(axis=0))
