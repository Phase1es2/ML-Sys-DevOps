# 🧠 Project Title: [AI-Driven Image Enhancement for Restoring Low-Quality Images to High Resolution]

## 👤 Unit 1: Any Person — Value Proposition

### 🎯 Specific Customer
Describe **one** specific customer/user persona your system is built for.

### 💡 Value Proposition
Explain how your ML system benefits this customer. What problem does it solve?

### 🔍 Customer-Driven Design Considerations
- Data requirements
- Deployment constraints
- Evaluation priorities

### 📊 Scale
- Dataset Size: XX GB
- Model Size: XX MB / Training Time: XX hours
- Deployment Load: XX inferences/hour, XX inferences/day

---

## ☁️ Unit 2/3: Cloud-Native Infrastructure

### 🗺️ Architecture Diagram
> *(Insert or link your updated system diagram here)*

### 🏗️ Infrastructure as Code (IaC)
Explain how you provision and configure each system component.
- Tools used: Terraform / Ansible / Pulumi / etc.
- Reproducibility scripts: [`/infra/`](./infra/)

---

## 📦 Unit 8: Data Management

### 📂 Persistent Storage
- Volumes/buckets used
- Example:
  - `bucket-ml-models/`: stores checkpoints (~2GB)
  - `volume-datasets/`: stores raw + preprocessed datasets (~20GB)

### 📉 Offline Datasets & Lineage
- Training dataset(s): e.g., DIV2K, CIFake
- Example sample(s): `data/sample_image.png`
- Label availability, feature timelines, etc.

### 🔄 Offline Data Pipeline
- Source → Storage: e.g., Kaggle API → MinIO
- Train/Val/Test splits: method and ratio
- Preprocessing: normalization, augmentation, etc.

### 📊 Optional: Data Dashboard
> *(Insert screenshot or link to your dashboard)*
Describe how this dashboard gives **data quality insights** for the customer.

---

## 🤖 Unit 4 & 5: Model Training

### 🧮 Modeling Setup
- Inputs: e.g., low-res image, metadata
- Outputs: high-res image, binary classification
- Model Used: e.g., ResNet-34 / ESRGAN
- Why this model fits your user & task

### 🏋️ Training Pipeline
- Training scripts: [`train.py`](./train.py)
- Re-training pipeline: [`pipeline/retrain.yaml`](./pipeline/retrain.yaml)

### 📈 Experiment Tracking
> *(Screenshot or link to MLFlow/W&B/DVC dashboard)*
Show key runs, comparisons, best run metrics.

### 📅 Scheduled Training
- How/when training jobs are triggered (cron/GitHub Actions/etc.)

### ⚡ Optional: Speedup Techniques
- E.g., “Training time reduced from 5h to 2h using mixed precision”

### 🧪 Optional: Ray Tune Integration
- Where and how Ray Tune was used for HPO

---

## 🌐 Unit 6 & 7: Serving and Evaluation

### 🔌 Serving API
- Input/Output format of API
- Serving framework: FastAPI / TorchServe / etc.

### 📋 Customer-Specific Requirements
- Latency, accuracy, etc.

### ⚙️ Model & System Optimizations
- e.g., quantization, ONNX conversion, autoscaling
- Relevant files: [`serve/`](./serve/), [`optimizations/`](./optimizations/)

### 🧪 Offline Evaluation
- Test suite location: [`tests/test_model.py`](./tests/test_model.py)
- Last model's metrics (accuracy, PSNR, etc.)

### 🚀 Load Test in Staging
> *(Paste output or link to results)*
- Tool used: Locust / JMeter / etc.

### 💼 Business-Specific Metric
- Hypothetical KPI, e.g., "revenue lift per improvement in PSNR"

### 🛠️ Optional: Multi-Option Serving
- Compare FastAPI vs. Triton, or GPU vs. CPU deployments
- Include cost/performance comparison

---

## 🔁 Unit 8: Online Data

### 📡 Online Data Flow
Explain how live inference inputs get sent to the endpoint during real-time use.

---

## 📈 Unit 6 & 7: Online Evaluation & Monitoring (1 minute)

### 🔍 Monitoring in Production
- Monitoring dashboards: Prometheus, Grafana, etc.
- Example feedback loop (e.g., user clicks → retraining signal)

### 📉 Optional: Data Drift Monitoring
> *(Link to dashboard/code path)*

### ⚠️ Optional: Model Degradation Alerts
> *(Link to dashboard/code path)*

---

## 🔁 Unit 2/3: CI/CD & Continuous Training (1 minute)

### 🔄 End-to-End Cycle
- Trigger: new data uploaded → retraining starts
- Show your CI/CD file: [`retrain.yml`](.github/workflows/retrain.yml)
- Path: data → training → serving → monitoring

---

## 📎 Appendix (optional)

- 🧪 Sample JSON requests/responses
- 🗂️ Folder structure
- 🔗 Links to demo/staging/prod
