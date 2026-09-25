# Neuromorphic Dataset Inventory

Data preparation only. Native events/spikes were loaded without static conversion, preprocessing, or experimental train/validation split creation.

| Dataset | Status | Native partitions | Samples | Classes | Sensor/input | Fields |
|---|---|---|---:|---:|---|---|
| SHD | READY | train, test | 10420 | 20 | [700, 1, 1] | t, x, p |
| DVS Gesture | READY | train, test | 1341 | 11 | [128, 128, 2] | x, y, p, t |
| CIFAR10-DVS | READY | all | 10000 | 10 | [128, 128, 2] | t, x, y, p |

## Details

### SHD

- Status: `READY`
- Storage: `data\shd`
- Native partitions: train, test
- train: 8156 samples; labels 0..19; sample 0 label 11; sample events/spikes 4278
- test: 2264 samples; labels 0..19; sample 0 label 10; sample events/spikes 11273
- Native fields: t, x, p
- Timestamps available: True
- Polarity/channel information: 700 cochlear input channels via x; Tonic's p field is a constant placeholder
- Static conversion performed: false

### DVS Gesture

- Status: `READY`
- Storage: `data\dvs_gesture`
- Native partitions: train, test
- train: 1077 samples; labels 0..10; sample 0 label 0; sample events/spikes 213025
- test: 264 samples; labels 0..10; sample 0 label 0; sample events/spikes 87787
- Native fields: x, y, p, t
- Timestamps available: True
- Polarity/channel information: binary polarity p
- Static conversion performed: false

### CIFAR10-DVS

- Status: `READY`
- Storage: `data\cifar10_dvs`
- Native partitions: all
- Partition note: Tonic/CIFAR10-DVS provides no official train/test partition; none was created.
- all: 10000 samples; labels 0..9; sample 0 label 0; sample events/spikes 178090
- Native fields: t, x, y, p
- Timestamps available: True
- Polarity/channel information: binary polarity p
- Static conversion performed: false
