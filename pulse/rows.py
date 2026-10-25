"""The rows of the paper's component table (Table 2), each one setting away from the row above.

A row names the PULSE variant (`pulse.model.Architecture`) and the rollout
arm (`pulse.feedback.ARMS`), and reads its own checkpoint, `<row>.pt`. Every
row reads the treatment policy at the last row of the event history and,
with a point head, adds the static noise of the validation patients.

`reference` is the architecture key of the reference driver
(`density_feedback_full.ARCHITECTURES`), kept to match the published
checkpoints with the rows.
"""
from __future__ import annotations

from dataclasses import dataclass

from pulse.feedback import ARMS
from pulse.model import Architecture


@dataclass(frozen=True)
class Row:
    label: str
    architecture: Architecture
    arm: str = "sample"
    reference: str = ""

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}")


_point = dict(head="point", event_inputs=False)
ROWS = {
    "point": Row("Point, static noise", Architecture(**_point, time="discrete", timing_head=False),
                 reference="point_noinputs"),
    "continuous_time": Row("+ continuous time", Architecture(**_point, timing_head=False),
                           reference="direct_continuous_time_noinputs"),
    "timing_head": Row("+ timing head", Architecture(**_point), reference="direct_timing_noinputs"),
    "event_inputs": Row("+ event-time inputs", Architecture(head="point"), reference="probe_event_time_direct"),
    "gaussian": Row("+ learned Gaussian", Architecture(head="gaussian"), reference="gaussian_direct"),
    "mixture": Row("+ two-Gaussian mixture", Architecture(head="mog2"), reference="probe_mog2_direct"),
    "flow": Row("- mixture, + flow", Architecture(head="cnf"), reference="probe_cnf_direct"),
    # The anchor row also changes the flow to the residual parameterisation, whose location is the
    # anchor plus an unconstrained residual; the reference built gate layers it never used.
    "anchor": Row("+ anchor", Architecture(head="rescnf"), reference="pulse"),
    "feedback": Row("+ feedback", Architecture(head="rescnf"), arm="sampled_state", reference="pulse"),
}
