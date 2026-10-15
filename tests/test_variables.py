import pytest
import torch

from pulse import variables as V
from synthetic import trajectories


def test_lists():
    assert len(V.CONTINUOUS) == 37 and len(V.BINARY) == 8 and len(V.STATIC) == 6
    assert set(V.RANGES) == set(V.CONTINUOUS)
    assert all(name in V.CONTINUOUS for name in V.ORDINAL)
    assert all(name in V.BINARY for name in V.ABSORBING)


def test_against_reference(expected):
    data = trajectories(seed=3)
    cont, keep, binary, means, event_now = (data[k] for k in ("continuous", "keep", "binary", "means", "event_now"))
    low, high = V.bounds()
    expected("bounds_low", low)
    expected("bounds_high", high)

    derived, plain = cont.clone(), cont.clone()
    baseline = derived[:, 0].clone()
    for t in range(cont.shape[1]):
        V.update_derived(derived, t, baseline)
        V.update_derived(plain, t)
    expected("derived", derived)
    expected("derived_without_baseline", plain)

    expected("fill_zero_invalid", V.forward_fill_zero_invalid(cont[:, 5].clone(), cont[:, 4], event_now))
    expected("fill_zero_invalid_no_event", V.forward_fill_zero_invalid(cont[:, 5].clone(), cont[:, 4]))

    death, release, event = V.event_times(binary)
    expected("death", torch.tensor(death))
    expected("release", torch.tensor(release))
    expected("event", event)

    expected("zero_invalid_mask", V.zero_invalid_mask(cont, keep.float(), event))
    expected("zero_invalid_mask_no_event", V.zero_invalid_mask(cont, keep.float()))
    filled = V.forward_fill(cont, keep, means)
    expected("forward_fill", filled)
    expected("fill_after_events", V.fill_after_events(filled, death, release, means))
    expected("fill_after_death_only", V.fill_after_events(filled, death))
    expected("fill_after_single_death", V.fill_after_events(filled, death[:1]))
    expected("fill_after_scalar_hours", V.fill_after_events(filled, 5, 9, means))

    sequence = cont.clone()
    for t in range(1, sequence.shape[1]):
        V.forward_fill_zero_invalid(sequence[:, t], sequence[:, t - 1])
    expected("fill_zero_invalid_all_hours", sequence)


def test_event_hours_need_one_per_row_or_one_for_all():
    cont = torch.zeros(3, 6, len(V.CONTINUOUS))
    with pytest.raises(ValueError):
        V.fill_after_events(cont, [2, 4])


def test_event_rule():
    binary = torch.zeros(3, 6, len(V.BINARY))
    binary[0, 2:, V.RELEASE] = 1
    binary[0, 4:, V.DEATH] = 1
    binary[1, 3:, V.RELEASE] = 1
    binary[1, 3:, V.DEATH] = 1
    death, release, event = V.event_times(binary)
    assert (death, release) == ([6, 3, 6], [2, 6, 6])
    assert event.sum(1).tolist() == [4, 3, 0]
