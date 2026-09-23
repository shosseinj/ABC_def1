import torch

from scripts.resume_shd_snn_350 import make_scheduler


def test_shd_resume350_scheduler_configuration_and_reduction():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=0.001)
    scheduler = make_scheduler(optimizer)
    scheduler.step(0.7)
    for _ in range(11):
        scheduler.step(0.6)
    assert optimizer.param_groups[0]["lr"] == 0.0005
    assert scheduler.mode == "max"
    assert scheduler.min_lrs == [1e-5]
