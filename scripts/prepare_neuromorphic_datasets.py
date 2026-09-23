"""Download and minimally verify native neuromorphic datasets without preprocessing."""
from pathlib import Path
import json
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tonic.datasets import CIFAR10DVS, DVSGesture, SHD


def sample_description(events):
    names = list(events.dtype.names or [])
    return {
        "container": "structured NumPy event/spike array",
        "shape": list(events.shape),
        "fields": names,
        "timestamps_available": "t" in names,
        "polarity_available": "p" in names,
        "channel_information": (
            "binary polarity p" if "p" in names else
            "700 cochlear input channels via x" if "x" in names else "not detected"
        ),
        "timestamp_min": int(events["t"].min()) if len(events) and "t" in names else None,
        "timestamp_max": int(events["t"].max()) if len(events) and "t" in names else None,
        "event_count": int(len(events)),
    }


def partition_summary(dataset):
    sample_index = 0
    events, label = dataset[sample_index]
    while len(events) == 0 and sample_index + 1 < len(dataset):
        sample_index += 1
        events, label = dataset[sample_index]
    if len(events) == 0:
        raise RuntimeError("No non-empty sample was found.")
    targets = np.asarray(dataset.targets, dtype=int)
    if targets.size == 0 and hasattr(dataset, "data_filename"):
        native_path = Path(dataset.location_on_system) / dataset.data_filename
        with h5py.File(native_path, "r") as handle:
            targets = np.asarray(handle["labels"], dtype=int)
    return {
        "samples": int(len(dataset)),
        "label_min": int(targets.min()),
        "label_max": int(targets.max()),
        "labels_present": sorted(map(int, np.unique(targets))),
        "sample_index": sample_index,
        "sample_label": int(label),
        "sample": sample_description(events),
    }


def verify_partitioned(name, loader, path, sensor_size, expected_classes):
    try:
        train = loader(save_to=str(path), train=True)
        test = loader(save_to=str(path), train=False)
        train_summary = partition_summary(train)
        test_summary = partition_summary(test)
        if name == "SHD":
            for summary in (train_summary, test_summary):
                summary["sample"]["polarity_available"] = False
                summary["sample"]["channel_information"] = (
                    "700 cochlear input channels via x; Tonic's p field is a constant placeholder"
                )
        labels = sorted(set(train_summary["labels_present"]) | set(test_summary["labels_present"]))
        if len(labels) != expected_classes:
            raise RuntimeError(f"Expected {expected_classes} classes, found {len(labels)}.")
        return {
            "status": "READY", "storage_path": str(path.relative_to(ROOT)),
            "native_partitions": ["train", "test"], "classes": len(labels),
            "sensor_size": list(sensor_size), "train": train_summary, "test": test_summary,
            "static_conversion_performed": False,
        }
    except Exception as error:
        return {
            "status": "FAILED", "storage_path": str(path.relative_to(ROOT)),
            "error_type": type(error).__name__, "error": str(error),
            "static_conversion_performed": False,
        }


def verify_cifar10_dvs(path):
    try:
        dataset = CIFAR10DVS(save_to=str(path))
        summary = partition_summary(dataset)
        if len(summary["labels_present"]) != 10:
            raise RuntimeError(f"Expected 10 classes, found {len(summary['labels_present'])}.")
        return {
            "status": "READY", "storage_path": str(path.relative_to(ROOT)),
            "native_partitions": ["all"],
            "partition_note": "Tonic/CIFAR10-DVS provides no official train/test partition; none was created.",
            "classes": 10, "sensor_size": list(CIFAR10DVS.sensor_size),
            "all": summary, "static_conversion_performed": False,
        }
    except Exception as error:
        return {
            "status": "FAILED", "storage_path": str(path.relative_to(ROOT)),
            "error_type": type(error).__name__, "error": str(error),
            "static_conversion_performed": False,
        }


def print_report(name, result):
    print(f"DATASET: {name}")
    print(f"status: {result['status']}")
    if result["status"] == "READY":
        for partition in result["native_partitions"]:
            print(f"{partition} samples: {result[partition]['samples']}")
        print(f"classes: {result['classes']}")
        first_partition = result["native_partitions"][0]
        sample = result[first_partition]["sample"]
        print(f"input/event structure: {sample['container']} {sample['shape']} fields={sample['fields']}")
        print(f"timestamps: {sample['timestamps_available']}")
        print(f"polarity/channel: {sample['channel_information']}")
        print(f"sensor/input dimensions: {result['sensor_size']}")
        print(
            f"label range: {result[first_partition]['label_min']}.."
            f"{result[first_partition]['label_max']}"
        )
        print(f"sample label: {result[first_partition]['sample_label']}")
    else:
        print(f"error: {result.get('error_type')}: {result.get('error')}")
    print(f"storage path: {result['storage_path']}")
    print()


def markdown(inventory):
    lines = [
        "# Neuromorphic Dataset Inventory", "",
        "Data preparation only. Native events/spikes were loaded without static conversion, "
        "preprocessing, or experimental train/validation split creation.", "",
        "| Dataset | Status | Native partitions | Samples | Classes | Sensor/input | Fields |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for name, result in inventory["datasets"].items():
        if result["status"] == "READY":
            partitions = ", ".join(result["native_partitions"])
            samples = sum(result[p]["samples"] for p in result["native_partitions"])
            first = result[result["native_partitions"][0]]["sample"]
            fields = ", ".join(first["fields"])
            lines.append(
                f"| {name} | READY | {partitions} | {samples} | {result['classes']} | "
                f"{result['sensor_size']} | {fields} |"
            )
        else:
            lines.append(f"| {name} | FAILED | -- | -- | -- | -- | -- |")
    lines.extend(["", "## Details", ""])
    for name, result in inventory["datasets"].items():
        lines.extend([f"### {name}", "", f"- Status: `{result['status']}`", f"- Storage: `{result['storage_path']}`"])
        if result["status"] == "READY":
            lines.append(f"- Native partitions: {', '.join(result['native_partitions'])}")
            if "partition_note" in result:
                lines.append(f"- Partition note: {result['partition_note']}")
            for partition in result["native_partitions"]:
                part = result[partition]
                lines.append(
                    f"- {partition}: {part['samples']} samples; labels {part['label_min']}.."
                    f"{part['label_max']}; sample 0 label {part['sample_label']}; "
                    f"sample events/spikes {part['sample']['event_count']}"
                )
            sample = result[result["native_partitions"][0]]["sample"]
            lines.append(f"- Native fields: {', '.join(sample['fields'])}")
            lines.append(f"- Timestamps available: {sample['timestamps_available']}")
            lines.append(f"- Polarity/channel information: {sample['channel_information']}")
            lines.append("- Static conversion performed: false")
        else:
            lines.append(f"- Error: `{result.get('error_type')}: {result.get('error')}`")
        lines.append("")
    return "\n".join(lines)


def main():
    data_root = ROOT / "data"
    paths = {
        "SHD": data_root / "shd",
        "DVS Gesture": data_root / "dvs_gesture",
        "CIFAR10-DVS": data_root / "cifar10_dvs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    # Figshare's browser endpoints return empty HTTP-202 responses in this environment.
    # Official API endpoints resolve to the same archives; Tonic still verifies its MD5s.
    DVSGesture.train_url = "https://api.figshare.com/v2/file/download/38022171"
    DVSGesture.test_url = "https://api.figshare.com/v2/file/download/38020584"
    CIFAR10DVS.url = "https://api.figshare.com/v2/file/download/38023437"
    inventory = {
        "phase": "data preparation only", "training_performed": False,
        "validation_splits_created": False, "static_conversion_performed": False,
        "loader": "tonic 1.6.0", "datasets": {},
    }
    inventory["datasets"]["SHD"] = verify_partitioned(
        "SHD", SHD, paths["SHD"], SHD.sensor_size, 20
    )
    print_report("SHD", inventory["datasets"]["SHD"])
    inventory["datasets"]["DVS Gesture"] = verify_partitioned(
        "DVS Gesture", DVSGesture, paths["DVS Gesture"], DVSGesture.sensor_size, 11
    )
    print_report("DVS Gesture", inventory["datasets"]["DVS Gesture"])
    inventory["datasets"]["CIFAR10-DVS"] = verify_cifar10_dvs(paths["CIFAR10-DVS"])
    print_report("CIFAR10-DVS", inventory["datasets"]["CIFAR10-DVS"])
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "neuromorphic_dataset_inventory.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8"
    )
    (results / "neuromorphic_dataset_inventory.md").write_text(
        markdown(inventory), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
