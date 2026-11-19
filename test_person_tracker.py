import unittest
from collections import deque
from person_tracker import PersonTracker, motion_features


class MotionTests(unittest.TestCase):
    def test_irregular_sampling_stationary_and_moving(self):
        still = deque((t, 50, 50, 100) for t in [0, .3, 1.1, 2.0])
        features = motion_features(still, 2, .3)
        self.assertAlmostEqual(features['stationary_fraction'], 1)
        self.assertAlmostEqual(features['score'], 1)
        moving = deque((t, 50 + 30 * t, 50, 100) for t in [0, .3, 1.1, 2.0])
        features = motion_features(moving, 2, .3)
        self.assertAlmostEqual(features['mean_speed'], .3)
        self.assertLess(features['score'], .75)

    def test_window_boundary_interpolation(self):
        # 旧时段的大幅位移不能影响当前窗口。
        samples = deque([(0, 0, 0, 100), (1, 100, 0, 100), (2, 100, 0, 100), (3, 100, 0, 100)])
        features = motion_features(samples, 2, .3)
        self.assertAlmostEqual(features['extent'], 0)
        self.assertAlmostEqual(features['score'], 1)

    def test_tracks_and_observation_gap(self):
        tracker = PersonTracker(window=2, max_missing=3, max_observation_gap=1.5, waterline=.2, enable_risk=True)
        a, b = [20, 100, 60, 200], [200, 100, 240, 200]
        def step(t, boxes):
            return {x.id: x for x in tracker.update([(box, .9) for box in boxes], t, 400)}
        step(0, [a, b])
        step(1, [b, a])
        tracks = step(2, [a, b])
        self.assertTrue(tracks[1].risk and tracks[2].risk)
        tracks = step(2.2, [b])
        self.assertFalse(tracks[1].visible)
        self.assertIsNone(tracks[1].features)
        tracks = step(4, [a, b])
        self.assertFalse(tracks[1].risk)
        self.assertIsNone(tracks[1].features)
        self.assertFalse(step(8, []))

    def test_disabled_risk_and_warmup(self):
        tracker = PersonTracker(window=1, max_observation_gap=2)
        box = [20, 100, 60, 200]
        first = tracker.update([(box, .9)], 0, 400)[0]
        self.assertIsNone(first.features)
        track = tracker.update([(box, .9)], 1, 400)[0]
        self.assertIsNotNone(track.features)
        self.assertFalse(track.risk)


if __name__ == '__main__':
    unittest.main()
