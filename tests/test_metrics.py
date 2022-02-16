from unittest import TestCase
from keras_utils import metrics


class TestCoverage(TestCase):
    def testcoverage(self):
        y_true = [[0, 1, 1, 1], [0, 0, 1, 1], [1, 0, 0, 0]]
        y_pred = [[.2, .3, .4, .5], [0.2, 0.5, .4, .6], [.8, .3, .9, .8]]
        # Expected ranks: 3, 3, 3
        m = metrics.Coverage()
        m.update_state(y_true, y_pred)
        self.assertEqual(float(m.result()), 3.0)
        # make sure aggregation is correctly performed
        m.update_state(y_true, y_pred)
        self.assertEqual(3.0, float(m.result()))

        # Test behavior when there is no true label
        # Maybe I should expect nan instead
        y_true = [[0, 0, 0, 0]]
        y_pred = [[.2, .3, .4, .5]]
        m = metrics.Coverage()
        m.update_state(y_true, y_pred)
        self.assertEqual(0.0, float(m.result()))

        # Test behavior when there are some negative pred values
        # Should be used with from_logits = True only
        y_true = [[0, 1, 0, 1]]
        y_pred = [[-10, -2, .4, .5]]
        m = metrics.Coverage(from_logits=True)
        m.update_state(y_true, y_pred)
        self.assertEqual(3.0, float(m.result()))
