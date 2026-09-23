"""Finalize metrics for the saved validation-only checkpoint after timeout."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.nmnist_hybrid_qsnn import NMNISTHybridQSNNV2, NMNISTSpatialLIFExtractor
from experiments.nmnist.hybrid_data import CachedFrameDataset
from scripts.run_nmnist_frontend_v3_seed42 import evaluate
from scripts.run_nmnist_hybrid_qsnn_seed42 import CACHE

checkpoint = ROOT / "checkpoints/nmnist_hybrid_qsnn_seed42/v3_wide4_quantum_full_best.pt"
payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
model = NMNISTHybridQSNNV2(
    extractor=NMNISTSpatialLIFExtractor(channels=(16, 32), spatial_size=4, latent_dim=32),
    latent_dim=32, n_blocks=2, learned_projection=True, two_axis_encoding=True,
    learned_measurement=True,
)
model.load_state_dict(payload["model_state"])
manifest = json.loads((ROOT / "results/nmnist_snn_multiseed_split.json").read_text(encoding="utf-8"))
validation_set = CachedFrameDataset(CACHE, np.asarray(manifest["validation_indices"], dtype=np.int64))
validation_loader = DataLoader(validation_set, batch_size=256, shuffle=False, num_workers=0,
                                 pin_memory=True)
model = model.cuda()
validation = evaluate(model, validation_loader)
result = {
    "frontend": "wide4", "model_type": "quantum", "stage": "full",
    "validation": validation, "stored_validation": payload["validation"],
    "validation_matches_checkpoint": validation == payload["validation"],
    "best_epoch": payload["epoch"],
    "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "batch_size": 256, "training_seconds_observed_until_external_timeout": 1200.0,
    "peak_gpu_bytes_not_written_before_external_timeout": None,
    "checkpoint": str(checkpoint.relative_to(ROOT)),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "official_test_accessed": False,
    "note": "Training reached epoch 38; saved best checkpoint is epoch 37. Final run JSON was not written before timeout.",
}
(ROOT / "results/nmnist_hybrid_qsnn_seed42/v3_wide4_quantum_full_result.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
