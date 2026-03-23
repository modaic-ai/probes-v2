import torch

from probes2.inference.reflect import (
    CONFIDENCE_TEMPERATURE,
    PLATT_BIAS,
    calibrate_confidences,
    compute_confidence_margin,
)


def test_compute_confidence_margin_uses_winning_gap() -> None:
    logits = torch.tensor(
        [
            [-0.4, 0.2, -10.0],
            [0.2, -0.4, -10.0],
            [0.7, 0.1, -0.2],
        ],
        dtype=torch.float32,
    )

    margins = compute_confidence_margin(logits)

    expected = torch.tensor([0.6, 0.6, 0.6], dtype=torch.float32)
    assert torch.allclose(margins, expected)


def test_calibrate_confidences_is_class_agnostic_and_less_extreme() -> None:
    logits = torch.tensor(
        [
            [-4.0, 4.0],
            [4.0, -4.0],
        ],
        dtype=torch.float32,
    )

    calibrated = calibrate_confidences(logits)

    expected = torch.sigmoid(
        torch.tensor([8.0, 8.0], dtype=torch.float32) / CONFIDENCE_TEMPERATURE + PLATT_BIAS
    )
    raw_confidence = torch.softmax(logits, dim=-1).amax(dim=-1)

    assert torch.allclose(calibrated, expected)
    assert torch.allclose(calibrated[0], calibrated[1])
    assert calibrated[0] < raw_confidence[0]
