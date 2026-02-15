# 🔬 MedSigLIP Pathology Workload Visualizer

Interactive workload visualization for deploying [MedSigLIP](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/medsiglip-model-card) — Google's medical foundation model for pathology — on Google Cloud.

The visualizer provisions GCP infrastructure (APIs, IAM, VPC, Cloud NAT, Workbench), then monitors a researcher's notebook as it loads the model, downloads datasets, fine-tunes on NCT-CRC-HE-100K, and saves results to GCS.

## Prerequisites

- **Google Cloud project** with billing enabled
- **gcloud CLI** installed and authenticated
- **Node.js** ≥ 18 and **Python** ≥ 3.10

## Setup & Run

### 1. Export environment variables

```bash
export GCP_PROJECT_ID="your-project-id"

# Optional — override defaults
export GCP_REGION="asia-northeast3"
export SERVICE_ACCOUNT_NAME="medsiglip-pipeline-sa"
export WORKBENCH_INSTANCE_NAME="medsiglip-researcher-workbench"
```

The bucket name is automatically derived as `${GCP_PROJECT_ID}-bucket`.

### 2. Authenticate to GCP

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project $GCP_PROJECT_ID
```

### 3. Start the backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Backend runs on `http://localhost:5000`.

### 4. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`. Open it in your browser.

### 5. Run the workflow

Click **Run Workflow** in the UI. The visualizer will:

1. **IT Infra** (automated): Enable APIs → Create SA → IAM Roles → Org Policies → Create VPC → Cloud NAT
2. **Provision Workbench**: Creates a Vertex AI Workbench instance with A100 GPU (tries multiple zones on stockout)
3. **Enter Monitoring Mode**: The UI starts polling GCS for researcher progress

### 6. Work in the notebook

After the workbench is provisioned (~10 min), wait ~30 minutes for the startup script to finish and the notebook to hydrate. Then:

1. Open the Workbench in the GCP Console (link shown in the UI)
2. **Install NVIDIA driver**: Open a terminal in the notebook and say 'y' to installing nvidia driver
3. Open `medsiglip-workspace/MedSigLIP_Pathology_Pipeline.ipynb`
4. Run the cells in order:
   - **Step 1**: Install dependencies (pip install), then **restart the kernel**
   - **Step 2**: Create GCS bucket
   - **Step 3**: Load MedSigLIP (writes `status/load-model.json` marker → UI updates)
   - **Step 4**: Download dataset (writes `status/download-dataset.json` marker → UI updates)
   - **Step 5–6**: Prepare data & zero-shot baseline
   - **Step 7**: Fine-tune (writes `status/fine-tune.json` marker → UI updates)
   - **Step 8**: Evaluate fine-tuned model
   - **Step 9**: Save results to GCS (UI detects `results/` + `medsiglip-finetune/` artifacts)

The frontend monitors GCS markers and artifact prefixes, updating the flow visualization in real-time as each notebook cell completes.

## Pipeline Flow

```
IT Infra (6 steps)
    ↓
Provision Workbench → Storage Bucket → ┬─ Load MedSigLIP ──┬→ Fine-Tune → Save Results to Bucket
                                       └─ Download Dataset ─┘
```
