# 🧠 Project Title: [AI-Driven Image Enhancement for Restoring Low-Quality Images to High Resolution]

## 👤 Unit 1: Any Person — Value Proposition

### 🎯 Specific Customer
One of our customer will be `Adobe`, A leader in creative and digital media softwar. Adobe's suit of tool-like Photoshop, Lightroom, and Premiere Pro-relies heavily on high-quality imgae processing. Our ML-based super-resolution system can be directly integrated into Adobe's editing pipeline or bacdkend services to enhance visual quality of preofessionals and everyday users alike.

### 💡 Value Proposition
Our depth-aware super-resolution system enabel Adobe to:
  - Improve image resolution while preserving fine details and edges by leveraging depth informaiton.
  - Enhane Legacy or low-resolution content into higher-quality outputs for creative workflows.
  - Enable smart image refinement features powered by AI, differentiating Adobe's products with cutting-edge super-resolution capabilites.

### 🔍 Customer-Driven Design Considerations
- Data requirements
  - High-quality paired datasets of `low-resolution` and `high-resolution`, ideally with aligned depth maps to train the DepthPro model effectively.
  - Real-world create assets to ensure generalization across diverse visual styles used by Adobe users.
- Deployment constraints
  - Low latency API access via `FastAPI` backend, containerized with Docker for coress-platform compatibility.
  - Ability to integrate into both cloud services (eg. Creative Cloud) and local installations (e.g., Photoshop plugins, Photoshop app).
- Evaluation priorities
  - maximize `PSNR` and `SSIM` on benchmark datasets (e.g., DIV2K, Urban100).
  - Human perceptual quality metrics are critical-outputs must look natural and sharp, especially for creative prefessional.
  - Model Size and inference speed tradeoffs matters.

### 📊 Scale
- Dataset Size: 4.2 GB
- Model Size: 600 MB  ~ 1.3 GB/ Training Time: XX hours
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

## Unit 8: Data Management

### Persistent Storage

To enable persistent storage access during training and inference, we completed the following steps:

1. **Provisioned Storage:**
   - Created one **40 GB block storage volume** and mounted it at `/mnt/block/` on `node1-cloud-project45`.
   - Created one **object storage bucket** (used for storing datasets like Urban100 and BSD100).

2. **Installed and Configured `rclone`:**
   - Installed `rclone` on the persistent compute node (`node-persist`).
   - Configured FUSE to allow user-mounted volumes to be accessed by other users and Docker containers.
   - Created a `rclone.conf` configuration file using the application credential ID, secret, and user ID.

3. **Mounted Object Storage:**
   - Created a mount point at `/mnt/object` and granted ownership to the `cc` user.
   - Used `rclone mount` to mount the object storage bucket (read-only) to `/mnt/object` with the `--allow-other` flag, enabling access from Docker containers and other services.

4. **Shell Script:**
   - Provided an automated setup script `setup_rclone_mnt.sh` under the `script/` directory to simplify the mounting process on new instances. This ensures consistent and reliable data access across the team.

---

### Offline Datasets & Lineage

- **Training dataset:** DIV2K  
- **Validation & Evaluation datasets:** BSD100, Urban100  
- **Example sample(s):**  
  ![Sample](img/0004.png)

The DIV2K dataset contains **1,000 high-resolution (2K) images**, split into:

- **Training Set:** 800 images  
- **Validation Set:** 100 images  
- **Test Set:** 100 images  

#### Data Lineage

1. **Raw Data Source:**
   - Downloaded from the official NTIRE 2017 Super-Resolution Challenge repository.
   - Files: `0001.png` to `1000.png`, each at 2K resolution.

2. **Preprocessing:**
   - Low-resolution (LR) images are generated using bicubic downsampling at 3 scales:
     - `YYYYx2.png` → ×2 scale
     - `YYYYx3.png` → ×3 scale
     - `YYYYx4.png` → ×4 scale
   - LR images are aligned with their high-resolution (HR) counterparts.
   - No additional image preprocessing is applied offline; cropping is done during training to maintain 1:1 aspect ratio.

---

### Offline Data Pipeline

We implemented a complete ETL pipeline using Docker Compose (`docker/docker-compose-etl.yaml`) to automate the download, organization, and upload of super-resolution datasets.

#### Data Extraction

Datasets are downloaded from Kaggle using three dedicated containers:

- `extract-data`: Downloads **DIV2K** (`div2k.zip`)
- `extract-test-data`: Downloads **Urban100** (`urban100.zip`)
- `extract-bsd100`: Downloads **BSD100** (`bsd100.zip`)

Each dataset is extracted into a shared volume (`sr:/data`) for downstream processing.

#### Data Transformation

Organized into structured folders:

- **DIV2K:**
  - `div2k/train`: 800 HR images (training)
  - `div2k/validation`: 100 HR images (validation)

- **Urban100:**
  - `test/urban100_x2/lr` and `hr`: ×2 scale testing
  - `test/urban100_x4/lr` and `hr`: ×4 scale testing

- **BSD100:**
  - `eval/bsd100_x2/lr` and `hr`: ×2 scale evaluation
  - `eval/bsd100_x4/lr` and `hr`: ×4 scale evaluation

#### Data Upload

The `load-data` container (based on the official `rclone` image) uploads the final dataset folder structure to the object store:

- **Destination:** `chi_tacc:$RCLONE_CONTAINER`
- Uses `rclone copy` with multi-threaded streaming (`--transfers`, `--checkers`, `--multi-thread-streams`)
- Cleans up prior files using `rclone delete` to avoid stale content
- Uploaded structure becomes accessible at `/mnt/object` during training/inference

---

### Data Splits & Leakage Avoidance

- **Training Set:** 800 HR–LR image pairs (DIV2K)
- **Validation Set:** 100 HR–LR image pairs (DIV2K)
- **Evaluation Sets:**
  - Urban100 (×2 and ×4 scales)
  - BSD100 (×2 and ×4 scales)

All datasets are separated strictly by source and use-case (train/validation/evaluation), and no images are shared between them. This ensures there is **no data leakage**, and evaluation remains unbiased.


### Block Storage Volumes for Persistent Services

To ensure data and service state are persistent across instance restarts, we mount our block storage volume (e.g., `/mnt/block`) into several key services in the Docker Compose environment:

| Service     | Mounted Volume                     | Purpose                                     |
|-------------|------------------------------------|---------------------------------------------|
| MinIO       | `/mnt/block/minio_data:/data`      | Stores MLflow artifact files (via S3 API)   |
| PostgreSQL  | `/mnt/block/postgres_data:/var/lib/postgresql/data` | Stores MLflow tracking metadata             |
| Prometheus  | `/mnt/block/prometheus:/prometheus`| Stores monitoring time-series data          |
| Grafana     | `/mnt/block/grafana:/var/lib/grafana` | Stores dashboards and user configuration  |

All these directories reside under a single mounted block volume, making them:

- **Persistent** across node restarts
- **Portable** across compute instances (if remounted)
- **Centralized** for easy backup or replication

This volume is manually mounted on the host system (e.g., `/dev/vdc1` → `/mnt/block`) before Docker Compose is run.

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

In production, new data is sent to the inference endpoint using a Python script located in the `script/` directory: `send_to_inference_endpoint.py`.

- **Input:** The script reads low-resolution images from the directory `/mnt/object/bsd100`.
- **Process:** Each image is sent via a POST request to the FastAPI inference server running at `http://<host>:5000`.
- **Output:** The server returns the super-resolved image, which the script then saves to a local directory.

This setup enables automated, batch-style inference suitable for production pipelines.

---

## 📈 Unit 6 & 7: Online Evaluation & Monitoring (1 minute)

### 🔍 Monitoring in Production
- Monitoring dashboards: Prometheus, Grafana, etc.
- Example feedback loop (e.g., user clicks → retraining signal)

---


### 🔄 End-to-End Cycle
- Trigger: new data uploaded → retraining starts
- Show your CI/CD file: [`retrain.yml`](.github/workflows/retrain.yml)
- Path: data → training → serving → monitoring

---

## 📎 Appendix (optional)

- 🧪 Sample JSON requests/responses
- 🗂️ Folder structure
- 🔗 Links to demo/staging/prod
