import json
from pathlib import Path
import numpy as np
import torch

from experiments.nmnist.snn_baseline import events_to_frames
from scripts.run_nmnist_common_attack_protocol_seed42 import OUT, load_models
from scripts.run_nmnist_attack_protocol_v2_canonical_seed42 import canonical_frames_torch


def test_all_common_samples_zero_timestamp_tensor_is_canonical():
    from tonic.datasets import NMNIST
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((OUT / "common_clean_correct_manifest.json").read_text())
    data=NMNIST(save_to=str(root / "data/nmnist"), train=True)
    for item in manifest["samples"]:
        events,_=data[int(item["sample_id"])]
        clean=torch.from_numpy(events_to_frames(events,10)).float()[None].cuda()
        actual=canonical_frames_torch(events,torch.as_tensor(np.asarray(events["t"],dtype=np.float64),device="cuda",dtype=torch.float32))
        assert torch.equal(clean,actual)


def test_zero_control_predictions_and_attack_success_are_clean():
    from tonic.datasets import NMNIST
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((OUT / "common_clean_correct_manifest.json").read_text())
    data=NMNIST(save_to=str(root / "data/nmnist"), train=True); models=load_models()
    for item in manifest["samples"]:
        events,label=data[int(item["sample_id"])]
        clean=torch.from_numpy(events_to_frames(events,10)).float()[None].cuda()
        actual=canonical_frames_torch(events,torch.as_tensor(np.asarray(events["t"],dtype=np.float64),device="cuda",dtype=torch.float32))
        assert torch.equal(clean,actual)
        for model in models.values():
            with torch.no_grad():
                clean_prediction=int(model(clean).argmax(1)); zero_prediction=int(model(actual).argmax(1))
            assert clean_prediction==zero_prediction==int(label)
