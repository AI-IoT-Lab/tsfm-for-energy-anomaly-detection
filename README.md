# TSFM Anomaly Detection Paper

This repository contains the code for the TSFM Anomaly Detection paper. It includes training and fine-tuning scripts for transformer-based anomaly detection models.

---

## Installation

### 0. Clone the Repository

```bash
git clone https://gitlab.com/basu1999/tsfm-anomaly-paper.git
cd tsfm-anomaly-paper
```

### 1. Install Docker

- **Windows & macOS**: Download and install from [docker.com/get-started](https://www.docker.com/get-started")  
- **Ubuntu**:
  ```bash
  sudo apt-get update
  sudo apt-get install -y docker.io
  sudo systemctl enable --now docker
  ```

> **GPUs:**  
> If you plan to leverage NVIDIA GPUs inside Docker, install the NVIDIA Container Toolkit:  
> ```bash
> distribution=$( . /etc/os-release; echo $ID$VERSION_ID )
> curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
> curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
>   sudo tee /etc/apt/sources.list.d/nvidia-docker.list
> sudo apt-get update
> sudo apt-get install -y nvidia-docker2
> sudo systemctl restart docker
> ```

### 2. Pull the Kaggle GPU Python Image

```bash
docker pull gcr.io/kaggle-gpu-images/python:latest
```

### 3. Run the Container

From your project root:

```bash
docker run --gpus all -it \
  --name tsfm-anomaly \
  -v "$(pwd)":/workspace \
  -w /workspace \
  gcr.io/kaggle-gpu-images/python:latest \
  /bin/bash
```

Inside the container, you will land in `/workspace` (the repo root).

---

## Environment Setup

Once inside the container, install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> If there is no `requirements.txt`, manually install essentials:
> ```bash
> pip install torch torchvision numpy pandas sklearn matplotlib
> ```

---
## Project Structure

```
.
├── lead-var-autoencoder.py
├── moment-lead-finetuning.py
├── lead-isolation-forest-cont(1).py
├── lead-Iof-count.py
├── lead-iqr.py
├── lead-mz-score.py
├── data/
├── configs/
├── results/
├── requirements.txt
└── README.md
```

---
## Usage

### 1. Executing `lead-var-autoencoder.py`

```bash
python lead-var-autoencoder.py \
  --data_path data/your_dataset.csv \
  --output_dir results/lead_vae \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --gpu
```
- `--data_path`: Path to CSV/NumPy dataset.
- `--output_dir`: Directory for checkpoints, logs, plots.
- `--epochs`: Number of training epochs.
- `--batch_size`: Samples per gradient update.
- `--learning_rate`: Optimizer step size.
- `--gpu`: Enable CUDA.

### 2. Executing `moment-lead-finetuning.py`

```bash
python moment-lead-finetuning.py \
  --pretrained_model results/lead_vae/best_model.pth \
  --data_path data/moment_lead.csv \
  --output_dir results/finetune \
  --epochs 20 \
  --batch_size 16 \
  --learning_rate 5e-5 \
  --gpu
```
- `--pretrained_model`: Path to `.pth` checkpoint.
- `--data_path`: Fine-tuning dataset path.
- `--output_dir`: Directory for fine-tune outputs.
- `--epochs`, `--batch_size`, `--learning_rate`, `--gpu`: As above.

### 3. Executing `lead-isolation-forest-cont(1).py`

```bash
python lead-isolation-forest-cont(1).py
```
Runs Isolation Forest on `train.csv` and reports F1, precision, recall.

### 4. Executing `lead-Iof-count.py`

```bash
python lead-Iof-count.py
```
Runs LOF experiment on `train.csv`.

### 5. Executing `lead-iqr.py`

```bash
python lead-iqr.py
```
Runs IQR-based anomaly detection on `train.csv`.

### 6. Executing `lead-mz-score.py`

```bash
python lead-mz-score.py
```
Runs modified Z-score detection on `train.csv`.

---

## Contributing

1. Fork the repo  
2. Create a branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes (`git commit -m 'Add YourFeature'`)  
4. Push (`git push origin feature/YourFeature`)  
5. Open a Merge Request  

---

## Contact

- **GitLab:** [@basu1999](https://gitlab.com/basu1999)
