from src.metrics import compression_ratio, normalized_hamming_distance, state_entropy


def test_state_entropy_binary_balanced() -> None:
    assert state_entropy([0, 0, 1, 1]) == 1.0


def test_normalized_hamming_distance() -> None:
    assert normalized_hamming_distance([0, 1, 2], [0, 2, 2]) == 1 / 3


def test_compression_ratio_positive() -> None:
    payload = b"aaaaaaaaaabbbbbbbbbbcccccccccc"
    assert compression_ratio(payload) > 0
