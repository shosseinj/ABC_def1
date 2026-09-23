import torch

from experiments.iris.training import set_seed
from models.qsnn import IrisQSNN


def test_training_updates_weights_and_checkpoint_round_trip(tmp_path):
    set_seed(42)
    model = IrisQSNN(4, 2, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    features = torch.rand(9, 4) * (torch.pi / 2)
    targets = torch.tensor([0, 1, 2] * 3)
    before = model.qlayer.weights.detach().clone()

    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(features), targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.qlayer.weights.grad is not None
    assert torch.isfinite(model.qlayer.weights.grad).all()
    optimizer.step()
    assert not torch.equal(before, model.qlayer.weights.detach())

    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = IrisQSNN(4, 2, 3)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert torch.allclose(model(features), restored(features))
