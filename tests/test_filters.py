from src.filters import HaltDetector, StateUniformDetector


def test_halt_detector_triggers_after_exact_window() -> None:
    detector = HaltDetector(window=3)
    snapshot = ((0, 0, 0, 1), (1, 1, 1, 2))

    assert detector.observe(snapshot) is False
    assert detector.observe(snapshot) is False
    assert detector.observe(snapshot) is False
    assert detector.observe(snapshot) is True


def test_state_uniform_detector_only_full_uniform() -> None:
    detector = StateUniformDetector()
    assert detector.observe([1, 1, 1, 1]) is True
    assert detector.observe([1, 1, 2, 1]) is False
