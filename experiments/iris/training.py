import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from encoding.quantum import angle_encode
from encoding.ttfs import ttfs_encode
from experiments.iris.data import load_iris_splits, load_iris_train_validation
from models.qsnn import IrisQSNN
from attacks.training_timing import training_timing_pgd
from defenses.quantum_temp import (
    adversarial_timing_loss,
    jensen_shannon,
    measured_product_fidelity,
    perturb_spike_times,
    quantum_temp_loss,
    timing_to_angle_torch,
    true_class_margin,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_theta(features, time_window=100.0):
    spike_times = ttfs_encode(features, time_window)
    return torch.tensor(angle_encode(spike_times, time_window), dtype=torch.float32)


def classification_metrics(targets, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def train_iris_model(config, checkpoint_path=None, evaluate_test=True, epoch_observer=None):
    seed = int(config["seed"])
    split_seed = int(config.get("split_seed", seed))
    set_seed(seed)
    if evaluate_test:
        Xtr, Xv, Xte, ytr, yv, yte, scaler = load_iris_splits(
            seed=split_seed, test_size=float(config["test_size"]),
            val_size=float(config["val_size"]),
        )
    else:
        Xtr, Xv, ytr, yv, scaler, _, _ = load_iris_train_validation(
            seed=split_seed, test_size=float(config["test_size"]),
            val_size=float(config["val_size"]),
        )
        Xte = yte = None
    time_window = float(config["time_window"])
    trX, vaX = (to_theta(X, time_window) for X in (Xtr, Xv))
    teX = to_theta(Xte, time_window) if evaluate_test else None
    train_times = torch.tensor(ttfs_encode(Xtr, time_window), dtype=torch.float32)
    train_y = torch.tensor(ytr, dtype=torch.long)
    val_y = torch.tensor(yv, dtype=torch.long)

    model = IrisQSNN(
        n_qubits=int(config["n_qubits"]),
        n_layers=int(config["n_layers"]),
        n_classes=int(config["n_classes"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    loss_fn = nn.CrossEntropyLoss()
    best_val = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    defense_history = []
    quantum_temp_enabled = bool(config.get("quantum_temp_enabled", False))
    epsilon_fraction = float(config.get("quantum_temp_epsilon", 0.02))
    lambda_q = float(config.get("lambda_q", 0.1))
    consistency_enabled = bool(config.get("consistency_enabled", False))
    lambda_pred = float(config.get("lambda_pred", 0.0)) if consistency_enabled else 0.0
    lambda_js = float(config.get("lambda_js", 0.0))
    lambda_margin = float(config.get("lambda_margin", 0.0))
    margin_target = float(config.get("margin_target", 0.1))
    perturbation_generator = torch.Generator().manual_seed(seed + 1500)
    normalize_quantum_loss = bool(config.get("normalize_quantum_loss", False))
    quantum_loss_ema = None
    ema_decay = float(config.get("quantum_loss_ema_decay", 0.9))
    adversarial_training_enabled = bool(config.get("adversarial_training_enabled", False))
    adversarial_training_epsilon = float(config.get("adversarial_training_epsilon", 0.02))
    adversarial_training_steps = int(config.get("adversarial_training_steps", 1))
    adversarial_training_step_size = config.get("adversarial_training_step_size")
    adversarial_training_step_size = (
        None if adversarial_training_step_size is None else float(adversarial_training_step_size)
    )
    adversarial_training_random_start = bool(config.get("adversarial_training_random_start", False))
    clean_ce_weight = float(config.get("clean_ce_weight", 1.0))
    lambda_adv = float(config.get("lambda_adv", 1.0))
    adversarial_generator = torch.Generator().manual_seed(seed + 1700)
    started = time.perf_counter()

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad()
        if adversarial_training_enabled:
            adversarial_times = training_timing_pgd(
                model,
                train_times,
                train_y,
                adversarial_training_epsilon * time_window,
                T=time_window,
                steps=adversarial_training_steps,
                step_size=adversarial_training_step_size,
                random_start=adversarial_training_random_start,
                generator=adversarial_generator,
            )
            clean_logits = model(trX)
            adversarial_logits = model(timing_to_angle_torch(adversarial_times, time_window))
            core_loss, core = adversarial_timing_loss(
                clean_logits, adversarial_logits, train_y,
                lambda_adv=lambda_adv,
                lambda_margin=lambda_margin,
                margin_target=margin_target,
            )
            clean_ce = core["clean"]
            adversarial_ce = core["adversarial"]
            js_loss = jensen_shannon(clean_logits, adversarial_logits)
            margin_loss = F.relu(margin_target - true_class_margin(adversarial_logits, train_y)).mean()
            weighted_clean = clean_ce_weight * clean_ce
            weighted_adversarial = core["weighted_adversarial"]
            weighted_js = lambda_js * js_loss
            weighted_margin = core["weighted_margin"]
            loss = core_loss + (clean_ce_weight - 1.0) * clean_ce + weighted_js
            defense_history.append({
                "epoch": epoch,
                "classification_loss": float(clean_ce.detach()),
                "adversarial_loss": float(adversarial_ce.detach()),
                "quantum_loss": 0.0,
                "consistency_loss": 0.0,
                "weighted_clean_loss": float(weighted_clean.detach()),
                "weighted_adversarial_loss": float(weighted_adversarial.detach()),
                "weighted_quantum_loss": 0.0,
                "weighted_consistency_loss": 0.0,
                "r_adv": float((weighted_adversarial / clean_ce).detach()),
                "r_q": 0.0,
                "r_pred": 0.0,
                "r_aux": float(((weighted_adversarial + weighted_js + weighted_margin) / clean_ce).detach()),
                "js_loss": float(js_loss.detach()),
                "margin_loss": float(margin_loss.detach()),
                "weighted_js_loss": float(weighted_js.detach()),
                "weighted_margin_loss": float(weighted_margin.detach()),
                "r_js": float((weighted_js / clean_ce).detach()),
                "r_margin": float((weighted_margin / clean_ce).detach()),
                "r_decision_aux": float(((weighted_adversarial + weighted_js + weighted_margin) / clean_ce).detach()),
                "total_loss": float(loss.detach()),
            })
        elif quantum_temp_enabled:
            perturbed_times = perturb_spike_times(
                train_times, epsilon_fraction, time_window, perturbation_generator
            )
            perturbed_angles = timing_to_angle_torch(perturbed_times, time_window)
            if normalize_quantum_loss:
                with torch.no_grad():
                    clean_features = model.quantum_features(trX)
                    perturbed_features = model.quantum_features(perturbed_angles)
                    raw_quantum = float((1.0 - measured_product_fidelity(clean_features, perturbed_features)).mean())
                    quantum_loss_ema = raw_quantum if quantum_loss_ema is None else (
                        ema_decay * quantum_loss_ema + (1.0 - ema_decay) * raw_quantum
                    )
                quantum_scale = max(quantum_loss_ema, 1e-8)
            else:
                quantum_scale = 1.0
            loss, components = quantum_temp_loss(
                model, trX, perturbed_angles, train_y, lambda_q, lambda_pred,
                quantum_scale=quantum_scale,
                lambda_js=lambda_js,
                lambda_margin=lambda_margin,
                margin_target=margin_target,
            )
            defense_history.append({
                "epoch": epoch,
                "classification_loss": float(components["classification"].detach()),
                "adversarial_loss": 0.0,
                "quantum_loss": float(components["quantum"].detach()),
                "consistency_loss": float(components["consistency"].detach()),
                "weighted_quantum_loss": float(components["weighted_quantum"].detach()),
                "weighted_clean_loss": float(components["classification"].detach()),
                "weighted_adversarial_loss": 0.0,
                "weighted_consistency_loss": float(components["weighted_consistency"].detach()),
                "r_q": float((components["weighted_quantum"] / components["classification"]).detach()),
                "r_adv": 0.0,
                "r_pred": float((components["weighted_consistency"] / components["classification"]).detach()),
                "r_aux": float(((components["weighted_quantum"] + components["weighted_consistency"] + components["weighted_js"] + components["weighted_margin"]) / components["classification"]).detach()),
                "js_loss": float(components["js"].detach()),
                "margin_loss": float(components["margin"].detach()),
                "weighted_js_loss": float(components["weighted_js"].detach()),
                "weighted_margin_loss": float(components["weighted_margin"].detach()),
                "r_js": float((components["weighted_js"] / components["classification"]).detach()),
                "r_margin": float((components["weighted_margin"] / components["classification"]).detach()),
                "r_decision_aux": float(((components["weighted_js"] + components["weighted_margin"] + components["weighted_quantum"]) / components["classification"]).detach()),
                "total_loss": float(loss.detach()),
            })
        else:
            logits = model(trX)
            loss = loss_fn(logits, train_y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(vaX)
            val_loss = float(loss_fn(val_logits, val_y).item())
            val_predictions = val_logits.argmax(1)
            val_accuracy = (val_predictions == val_y).float().mean().item()
        history.append((epoch, float(loss.item()), val_loss, val_accuracy))
        if epoch_observer is not None:
            epoch_observer(model, epoch, val_loss, val_accuracy, vaX, val_y)
        if val_accuracy > best_val or (
            val_accuracy == best_val and val_loss < best_val_loss
        ):
            best_val = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(exist_ok=True)
        torch.save(best_state, checkpoint_path)

    model.eval()
    with torch.no_grad():
        validation_predictions = model(vaX).argmax(1).cpu().numpy()
    validation_metrics = classification_metrics(yv, validation_predictions)
    predictions = None
    metrics = {}
    if evaluate_test:
        with torch.no_grad():
            predictions = model(teX).argmax(1).cpu().numpy()
        metrics = classification_metrics(yte, predictions)
    metrics.update({
        "best_val_accuracy": float(best_val),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "trainable_parameters": int(model.trainable_parameter_count()),
        "quantum_parameters": int(config["n_layers"]) * int(config["n_qubits"]) * 2,
        "n_layers": int(config["n_layers"]),
        "n_qubits": int(config["n_qubits"]),
        "seed": seed,
        "split_seed": split_seed,
        "epochs": int(config["epochs"]),
        "learning_rate": float(config["learning_rate"]),
        "quantum_temp_enabled": quantum_temp_enabled,
        "quantum_temp_epsilon": epsilon_fraction,
        "lambda_q": lambda_q,
        "lambda_pred": lambda_pred,
        "consistency_enabled": consistency_enabled,
        "normalize_quantum_loss": normalize_quantum_loss,
        "lambda_js": lambda_js,
        "lambda_margin": lambda_margin,
        "margin_target": margin_target,
        "adversarial_training_enabled": adversarial_training_enabled,
        "adversarial_training_epsilon": adversarial_training_epsilon,
        "adversarial_training_steps": adversarial_training_steps,
        "adversarial_training_step_size": adversarial_training_step_size,
        "adversarial_training_random_start": adversarial_training_random_start,
        "clean_ce_weight": clean_ce_weight,
        "lambda_adv": lambda_adv,
        "elapsed_seconds": float(time.perf_counter() - started),
    })
    if defense_history:
        loss_summary = {
            key: float(np.mean([row[key] for row in defense_history]))
            for key in (
                "classification_loss", "adversarial_loss", "quantum_loss", "consistency_loss",
                "weighted_clean_loss", "weighted_adversarial_loss",
                "weighted_quantum_loss", "weighted_consistency_loss",
                "r_adv", "r_q", "r_pred", "r_aux",
                "js_loss", "margin_loss", "weighted_js_loss",
                "weighted_margin_loss", "r_js", "r_margin", "r_decision_aux", "total_loss",
            )
        }
    else:
        loss_summary = {
            "classification_loss": float(np.mean([row[1] for row in history])),
            "adversarial_loss": 0.0,
            "quantum_loss": 0.0,
            "consistency_loss": 0.0,
            "weighted_quantum_loss": 0.0,
            "weighted_clean_loss": float(np.mean([row[1] for row in history])),
            "weighted_adversarial_loss": 0.0,
            "weighted_consistency_loss": 0.0,
            "r_q": 0.0,
            "r_adv": 0.0,
            "r_pred": 0.0,
            "r_aux": 0.0,
            "js_loss": 0.0,
            "margin_loss": 0.0,
            "weighted_js_loss": 0.0,
            "weighted_margin_loss": 0.0,
            "r_js": 0.0,
            "r_margin": 0.0,
            "r_decision_aux": 0.0,
            "total_loss": float(np.mean([row[1] for row in history])),
        }
    return {
        "model": model,
        "metrics": metrics,
        "history": history,
        "predictions": predictions,
        "targets": np.asarray(yte) if evaluate_test else None,
        "scaler": scaler,
        "defense_history": defense_history,
        "loss_summary": loss_summary,
        "validation_metrics": validation_metrics,
        "test_evaluated": bool(evaluate_test),
    }

def train_iris_model_from_arrays(config,X_train,X_validation,y_train,y_validation,checkpoint_path=None):
    """Established CE baseline protocol using authorized arrays; invokes no loader."""
    set_seed(int(config["seed"]));trX=to_theta(X_train,float(config["time_window"]));vaX=to_theta(X_validation,float(config["time_window"]));ty=torch.tensor(y_train,dtype=torch.long);vy=torch.tensor(y_validation,dtype=torch.long)
    model=IrisQSNN(int(config["n_qubits"]),int(config["n_layers"]),int(config["n_classes"]));opt=torch.optim.Adam(model.parameters(),lr=float(config["learning_rate"]));key=None;best=None;history=[]
    for epoch in range(1,int(config["epochs"])+1):
        model.train();opt.zero_grad();loss=F.cross_entropy(model(trX),ty);loss.backward();opt.step();model.eval()
        with torch.no_grad():z=model(vaX);vce=float(F.cross_entropy(z,vy));vacc=float((z.argmax(1)==vy).float().mean())
        history.append((epoch,float(loss.detach()),vce,vacc));k=(-vacc,vce,epoch)
        if key is None or k<key:key=k;best={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    model.load_state_dict(best)
    if checkpoint_path is not None:checkpoint_path.parent.mkdir(exist_ok=True);torch.save(best,checkpoint_path,_use_new_zipfile_serialization=False)
    return {"model":model,"history":history,"metrics":{"best_val_accuracy":-key[0],"best_val_loss":key[1],"best_epoch":key[2]},"test_evaluated":False,"array_training":True}
