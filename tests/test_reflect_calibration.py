import pytest
import torch
from datasets import Dataset

from probes2.finetuning.linear_head import (
    LinearProbeHead,
    build_linear_head_from_checkpoint,
    save_linear_head_checkpoint,
)
from probes2.inference.reflect import (
    CONFIDENCE_TEMPERATURE,
    PLATT_BIAS,
    calibrate_confidences,
    compute_confidence_margin,
    extract_last_token_embeddings,
    run_inference,
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


def test_extract_last_token_embeddings_uses_attention_mask_positions() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [9.0, 9.0]],
            [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 1],
        ]
    )

    embeddings = extract_last_token_embeddings(hidden, attention_mask)

    expected = torch.tensor([[2.0, 2.0], [5.0, 5.0]])
    assert torch.allclose(embeddings, expected)


def test_run_inference_uses_tuned_linear_head_checkpoint(tmp_path) -> None:
    class FakeTokenizer:
        pad_token_id = 0

    class FakeOutputs:
        def __init__(self, logits: torch.Tensor, hidden_states: list[torch.Tensor]) -> None:
            self.logits = logits
            self.hidden_states = hidden_states

    class FakeModel:
        def __init__(self) -> None:
            self._device = torch.device("cpu")

        def parameters(self):
            return iter([torch.zeros(1)])

        def __call__(self, input_ids, attention_mask, output_hidden_states=False):
            del input_ids, attention_mask
            logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
            hidden = torch.tensor(
                [
                    [[0.1, 0.1], [1.0, -1.0]],
                    [[0.2, 0.2], [-1.0, 2.0]],
                ],
                dtype=torch.float32,
            )
            return FakeOutputs(logits=logits, hidden_states=[hidden] if output_hidden_states else None)

    checkpoint_path = tmp_path / "head.pt"
    head = LinearProbeHead(input_dim=2)
    with torch.no_grad():
        head.linear.weight.copy_(torch.tensor([[2.0, -1.0]]))
        head.linear.bias.copy_(torch.tensor([0.5]))
    save_linear_head_checkpoint(checkpoint_path, head, base_model="modaic/test-probe")

    loaded_head, _ = build_linear_head_from_checkpoint(checkpoint_path, device="cpu")

    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2], [3, 4]],
            "attention_mask": [[1, 1], [1, 1]],
        }
    )

    confidences, embeddings = run_inference(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        dataset=dataset,
        batch_size=2,
        with_embeddings=True,
        linear_head=loaded_head,
    )

    expected_embeddings = [[1.0, -1.0], [-1.0, 2.0]]
    expected_confidences = torch.sigmoid(torch.tensor([3.5, -3.5])).tolist()

    assert embeddings == expected_embeddings
    assert confidences == pytest.approx(expected_confidences)
