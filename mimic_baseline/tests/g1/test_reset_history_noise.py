"""Collection and imagination must agree on reset-padding noise correlation."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from mimic_baseline.tasks.supermimic.core.mdp import MDP, OBS_NOISE
from mimic_baseline.tasks.supermimic.learning.collection import CollectionMixin
from mimic_baseline.tasks.supermimic.learning.policy_update import PolicyUpdateMixin


class HistoryCaptured(Exception):
    pass


class Collector(CollectionMixin):
    @property
    def pmg(self):
        # Stop after the production history update, before policy/simulator work.
        raise HistoryCaptured


def noise_model():
    return SimpleNamespace(add_noise=lambda x: MDP.add_noise(
        SimpleNamespace(obs_noise=torch.cat((OBS_NOISE[:64], torch.zeros(29)))), x))


def initialize(history):
    trainer = SimpleNamespace(_as_state=lambda x: x,
                              _obs=lambda s, a, xi: s, mdp=noise_model())
    return PolicyUpdateMixin._initial_observation_history(trainer, history)


@pytest.mark.parametrize('hist', [1, 2, 10])
def test_imagined_history_matches_actual_collection_with_same_frame_noise(hist):
    ages = torch.tensor([0, 1, 4, 12])
    back = torch.minimum(torch.arange(hist-1, -1, -1)[None], ages[:, None])
    steps = ages[:, None] - back
    raw = steps[..., None].float().expand(-1, -1, 93).clone()
    history = dict(s=raw, a_prev=None, xi=None, ep_step=steps)
    torch.manual_seed(71)
    draws = noise_model().add_noise(raw)
    torch.manual_seed(71)
    imagined = initialize(history)

    collector = Collector()
    collector.physics_health = None
    collector.cuda_cache = SimpleNamespace(before_collection=lambda _: None)
    collector.iteration = 0
    collector.profiler = SimpleNamespace(record=lambda _: nullcontext())
    collector.cfg = SimpleNamespace(hist=hist)
    collector._obs_history = None
    collector._as_state = lambda x: x
    collector.env = SimpleNamespace(state=lambda: None, xi=None)
    # Same noise realization for each real frame, including the seed itself.
    # Different episode ages exercise asynchronous reset of individual rows.
    for time in range(13):
        ep_step = (ages - (12-time)).clamp_min(0)
        slot = (hist-1 - (ages-ep_step)).clamp_min(0)
        collector.env.ep_step = ep_step
        collector.env.policy_observation = lambda: draws[torch.arange(4), slot]
        with pytest.raises(HistoryCaptured):
            collector.collect(1)
    torch.testing.assert_close(imagined, collector._obs_history, rtol=0, atol=0)


@pytest.mark.parametrize('steps', [[7, 7, 7, 8, 9], [0, 0, 0, 0, 0], [2, 3, 4, 5, 6]])
def test_padding_reuses_oldest_retained_frame_and_keeps_rng_budget(steps):
    # First case also covers a replay ring boundary inside an ongoing episode.
    ep = torch.tensor([steps])
    raw = ep[..., None].float().expand(-1, -1, 93).clone()
    history = dict(s=raw, a_prev=None, xi=None, ep_step=ep)
    torch.manual_seed(123)
    original = noise_model().add_noise(raw)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(123)
    actual = initialize(history)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for i, step in enumerate(steps):
        canonical = max(j for j, other in enumerate(steps) if other == step)
        torch.testing.assert_close(actual[:, i], original[:, canonical], rtol=0, atol=0)
    assert torch.equal(history['s'], raw)
    if len(set(steps)) == len(steps):
        assert torch.equal(actual, original)


@pytest.mark.parametrize('oldest_step', [0, 7])
def test_production_replay_metadata_handles_wrapped_ring_and_short_episode(oldest_step):
    from mimic_baseline.tasks.supermimic.learning.replay import ReplayBuffer

    buffer = ReplayBuffer(3, 1, 'cpu', action_dim=29, recovery_enabled=True)
    buffer.size, buffer.ptr = 3, 1  # Oldest frame lives at physical row 1.
    for tensor in buffer.data.values():
        tensor.zero_()
    rows = buffer._rows_for_ages(torch.arange(3))
    buffer.data['ep_step'][rows, 0] = torch.arange(oldest_step, oldest_step+3, dtype=torch.int32)
    buffer.data['s'][rows, 0, 0] = torch.arange(oldest_step, oldest_step+3).float()
    motion = SimpleNamespace(clip_id=torch.zeros(20, dtype=torch.long), clip_end=torch.tensor([20]))
    seed, history = buffer.sample_starts(64, 10, 1, motion, generator=torch.Generator().manual_seed(2))
    # Observation adapter for this replay fixture: encode the stored frame id.
    history['s'] = history['s'][..., :1].expand(-1, -1, 93).clone()
    assert set(seed['ep_step'].tolist()) == set(range(oldest_step, oldest_step+3))
    torch.manual_seed(81)
    draws = noise_model().add_noise(history['s'])
    torch.manual_seed(81)
    actual = initialize(history)
    for b, steps in enumerate(history['ep_step'].tolist()):
        for i, step in enumerate(steps):
            canonical = max(j for j, other in enumerate(steps) if other == step)
            torch.testing.assert_close(actual[b, i], draws[b, canonical], rtol=0, atol=0)
    torch.testing.assert_close(actual[:, -1], draws[:, -1], rtol=0, atol=0)
