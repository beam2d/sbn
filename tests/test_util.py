from unittest import TestCase

from sbn.util import cached_property


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
