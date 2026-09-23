"""Paired Seed-42 N-MNIST attack-protocol validation.

Both frozen models receive the same native event stream. PGD and TEMP-DRIFT
perturb only event timestamps before the shared 10-bin frame construction;
coordinates, polarity, event count, and labels are immutable.
"""
from pathlib import Path
import csv, hashlib, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.nmnist.snn_baseline import events_to_frames
from models.nmnist_snn import NMNISTConvSNN
from models.nmnist_hybrid_qsnn import NMNISTHybridQSNNV2, NMNISTSpatialLIFExtractor
from scripts.run_nmnist_hybrid_qsnn_seed42 import sha256
from scripts.run_nmnist_qsnn_v3_multiseed import load_fixed_frontend, QUANTUM_CONFIG, FRONTEND_CONFIG

SEED = 42
SPLIT = ROOT / "results/nmnist_snn_multiseed_split.json"
SNN_CKPT = ROOT / "checkpoints/nmnist_snn_clean_seed42_best.pt"
QSNN_CKPT = ROOT / "checkpoints/nmnist_hybrid_qsnn_seed42/nmnist_qsnn_v3_multiseed/seed42_best.pt"
OUT = ROOT / "results/nmnist_common_attack_protocol_seed42"
EPS = (0.02, 0.05, 0.10)
PGD_STEPS = 20
TEMP_INITIAL, TEMP_GENERATIONS, TEMP_PER_GENERATION = 600, 4, 250

def frames_torch(events, timestamps):
    """Differentiable timestamp-to-frame map with hard forward semantics."""
    if timestamps.ndim == 1: timestamps = timestamps[None, :]
    device, dtype = timestamps.device, timestamps.dtype
    t0, duration = float(events["t"][0]), max(float(events["t"][-1])-float(events["t"][0])+1, 1.0)
    u = (timestamps - t0) * 10.0 / duration
    hard_bin = u.floor().long().clamp(0, 9)
    centers = torch.arange(10, device=device, dtype=dtype) + 0.5
    soft_weight = (1 - (u[..., None] - centers).abs()).clamp_min(0)
    channel_np = 2 * (np.asarray(events["y"]) * 34 + np.asarray(events["x"])) + np.asarray(events["p"])
    channel = torch.as_tensor(channel_np, device=device, dtype=torch.long)[None, :].expand(len(timestamps), -1)
    hard = torch.zeros((len(timestamps), 10*2*34*34), device=device, dtype=dtype)
    hard.scatter_add_(1, hard_bin * (2*34*34) + channel, torch.ones_like(timestamps))
    soft = torch.zeros_like(hard)
    for b in range(10):
        soft.scatter_add_(1, torch.full_like(channel, b) * (2*34*34) + channel, soft_weight[:, :, b])
    counts = soft + (hard - soft).detach()
    return counts.reshape(-1, 10, 2, 34, 34).clamp_max(255)

def project(candidate, clean, eps):
    out = np.clip(candidate, clean-eps, clean+eps)
    out = np.maximum.accumulate(out)
    return np.clip(out, clean[0], clean[-1])

def logits(model, events, timestamps): return model(frames_torch(events, timestamps))
def margin(output, label):
    other = output.clone(); other[:, label] = -torch.inf
    return float((output[:, label] - other.max(1).values).item())

def pgd(model, events, label, eps):
    clean = np.asarray(events["t"], dtype=np.float64)
    x = torch.tensor(clean, device="cuda", dtype=torch.float32)
    best, best_loss = clean.copy(), -float("inf")
    for _ in range(PGD_STEPS + 1):
        with torch.no_grad(): loss = float(F.cross_entropy(logits(model, events, x), torch.tensor([label], device="cuda")))
        if loss > best_loss: best_loss, best = loss, x.detach().cpu().numpy().copy()
        if _ == PGD_STEPS: break
        x = x.detach().requires_grad_(True)
        loss = F.cross_entropy(logits(model, events, x), torch.tensor([label], device="cuda"))
        grad, = torch.autograd.grad(loss, x)
        x = torch.tensor(project((x.detach()+eps/5*grad.sign()).cpu().numpy(), clean, eps), device="cuda", dtype=torch.float32)
    return best

def temp_drift(model, events, label, eps, rng):
    clean = np.asarray(events["t"], dtype=np.float64); n = len(clean)
    pool = np.asarray([project(clean + rng.uniform(-eps, eps, n), clean, eps) for _ in range(TEMP_INITIAL)])
    def score(rows):
        with torch.no_grad():
            out = logits(model, events, torch.tensor(rows, device="cuda", dtype=torch.float32))
        return -out[:, label].detach().cpu().numpy(), rows
    values, pool = score(pool)
    for scale in (0.50, 0.30, 0.18, 0.10):
        elite = pool[np.argsort(values)[:12]]
        children = []
        for i in range(TEMP_PER_GENERATION):
            parent = elite[i % len(elite)]
            proposal = parent + 0.5*(elite[rng.integers(12)] - elite[rng.integers(12)])
            proposal += rng.normal(0, eps*scale, n)
            children.append(project(proposal, clean, eps))
        child = np.asarray(children); child_values, _ = score(child)
        pool = np.vstack((elite, child)); values = np.concatenate((values[np.argsort(values)[:12]], child_values))
    return pool[int(np.argmin(values))]

def load_models():
    snn = NMNISTConvSNN(0.5).cuda().eval(); snn.load_state_dict(torch.load(SNN_CKPT, map_location="cuda", weights_only=True)["model_state"])
    extractor = NMNISTSpatialLIFExtractor(**FRONTEND_CONFIG)
    qsnn = NMNISTHybridQSNNV2(extractor=extractor, latent_dim=32, **QUANTUM_CONFIG).cuda().eval()
    q = torch.load(QSNN_CKPT, map_location="cuda", weights_only=True); load_fixed_frontend(qsnn); qsnn.load_state_dict(q["model_state"])
    return {"SNN": snn, "QSNN": qsnn}

def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    from tonic.datasets import NMNIST
    data = NMNIST(save_to=str(ROOT/"data/nmnist"), train=True)
    split = json.loads(SPLIT.read_text()); ids = np.asarray(split["validation_indices"], dtype=np.int64)
    models = load_models(); clean = {"SNN": [], "QSNN": []}; class_counts = np.zeros(10, dtype=int)
    for sid in ids:
        events, label = data[int(sid)]; frame = torch.from_numpy(events_to_frames(events, 10)).float()[None].cuda()
        row = {"sample_id": int(sid), "label": int(label)}
        with torch.no_grad():
            for name, model in models.items(): row[name] = int(model(frame).argmax(1).item())
        if (row["SNN"] == row["label"] and row["QSNN"] == row["label"]
                and class_counts[row["label"]] < 10):
            clean["SNN"].append((int(sid), int(label))); class_counts[row["label"]] += 1
        if len(clean["SNN"]) == 100: break
    if len(clean["SNN"]) != 100 or not np.all(class_counts == 10): raise RuntimeError(f"Fewer than 10 common clean-correct samples in a class: {class_counts.tolist()}")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "selection": "first 10 per class among common clean-correct validation samples", "samples": [{"sample_id": i, "label": y} for i,y in clean["SNN"]], "split_sha256": sha256(SPLIT), "snn_checkpoint_sha256": sha256(SNN_CKPT), "qsnn_checkpoint_sha256": sha256(QSNN_CKPT)}
    (OUT/"common_clean_correct_manifest.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    for pos, (sid, label) in enumerate(clean["SNN"]):
        events, _ = data[sid]; timestamps = np.asarray(events["t"], dtype=np.float64)
        for name, model in models.items():
            for ef in EPS:
                eps = ef * (timestamps[-1]-timestamps[0]+1)
                for attack in ("PGD", "TEMP-DRIFT"):
                    start=time.perf_counter(); rng=np.random.default_rng(SEED+pos*1009+int(ef*100))
                    attacked = pgd(model, events, label, eps) if attack=="PGD" else temp_drift(model, events, label, eps, rng)
                    attacked=project(attacked,timestamps,eps)
                    with torch.no_grad(): out=logits(model,events,torch.tensor(attacked,device="cuda",dtype=torch.float32)); pred=int(out.argmax(1).item()); m=margin(out,label)
                    rows.append({"sample_id":sid,"label":label,"model":name,"attack":attack,"epsilon_fraction":ef,"event_count":len(events),"clean_prediction":label,"attacked_prediction":pred,"attack_success":pred!=label,"attacked_margin":m,"mean_abs_timestamp_drift":float(np.mean(np.abs(attacked-timestamps))),"feasible":bool(np.all(np.diff(attacked)>=0) and np.all(np.abs(attacked-timestamps)<=eps+1e-5)),"runtime_seconds":time.perf_counter()-start})
                    print(f"{pos+1}/100 {name} {attack} {ef:.2f} success={pred!=label}",flush=True)
    with (OUT/"per_sample_results.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    (OUT/"protocol.json").write_text(json.dumps({"pgd_space":"native event timestamps before shared 10-bin frame construction; true-label CE; 20 sign steps; L-infinity timestamp bound epsilon","temp_drift_space":"native event timestamps before shared 10-bin frame construction; derivative-free timestamp search; same bound","representation":"events_to_frames: 10 ordered bins, 2 polarity channels, 34x34 sensor; identical for both models","coordinates_polarity_event_count_label":"preserved","epsilons":EPS,"models":["frozen Seed-42 SNN","frozen Seed-42 QSNN-v3 wide4 + project_measure2"],"five_seed_campaign_started":False},indent=2))

if __name__ == "__main__": main()
