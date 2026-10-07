"""Continuous batching on a real model (issue #149): VoxCPM2's batched loop as a SlotModel
(``voxcpm_slots``) under the generic ``serve``, on the tiny CPU VoxCPM2. Requests of
different lengths share two slots, each refilled as soon as its request stops, every slot
at its own KV position; each request must still be VoxCPM's own batch-1 ``generate`` of its
text and seed, whatever slot it ran in and whoever ran beside it."""

from typing import Any

import pytest
import torch
import voxcpm_tiny
from test_voxcpm_batch import _spec, _stock
from voxcpm_slots import VoxCPMSlots

from kernel_agent.workloads import create_workload
from kernel_agent.workloads.serving import per_request, serve
from kernel_agent.workloads.voxcpm_batch import VoxCPMBatchWorkload, request_texts

pytestmark = pytest.mark.skipif(not voxcpm_tiny.available(), reason="needs voxcpm")

LENGTHS = [6, 2, 3, 1, 4]  # patches per request


@pytest.fixture
def workload(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    wl = create_workload(_spec(batch_size=2, patches=max(LENGTHS)))
    assert isinstance(wl, VoxCPMBatchWorkload)
    wl.model = voxcpm_tiny.tiny_voxcpm2()
    wl.sampling_rate = wl.model.sample_rate
    return wl


def _served(wl: VoxCPMBatchWorkload, continuous: bool) -> tuple[Any, torch.Tensor, list[Any]]:
    texts = request_texts(str(wl.options["text"]), len(LENGTHS))
    slots = VoxCPMSlots(wl, texts, slots=2, patches=max(LENGTHS))
    records: list[torch.Tensor] = []

    def record(pred: torch.Tensor) -> torch.Tensor:
        records.append(pred.detach().clone())
        return pred

    with torch.inference_mode(), wl._decoder_hook(record):
        served = serve(slots, len(texts), continuous=continuous, max_steps=LENGTHS, stop=False)
    latents, steps = per_request(records, served.tables, len(texts))
    assert steps == LENGTHS
    return served, latents, texts


def test_every_request_of_a_continuous_batch_is_voxcpms_batch_1_run(workload):
    served, latents, texts = _served(workload, continuous=True)
    assert served.steps == LENGTHS and served.iterations == 10  # static batching: 13
    assert served.slots == [0, 1, 1, 1, 0] and served.first == [0, 0, 2, 5, 6]
    stock = _stock(workload)
    for m, text in enumerate(texts):
        with stock.with_options({"seed": m, "patches": LENGTHS[m]}):
            single = stock.run(text)  # VoxCPM's own generate(text_m, seed m)
        torch.testing.assert_close(latents[: LENGTHS[m], m : m + 1], single["latents"])
        audio = served.outputs[m]
        assert audio is not None
        torch.testing.assert_close(audio, single["audio"], atol=1e-5, rtol=1e-4)

    static, again, _ = _served(workload, continuous=False)
    assert static.iterations == 13 and static.first == [0, 0, 6, 6, 9]
    torch.testing.assert_close(again, latents)  # the schedule changes nothing per request
