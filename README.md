# 🔬 MedSigLIP Pathology Workload Visualizer

Interactive workload visualization for deploying [MedSigLIP](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/medsiglip-model-card) — Google's medical foundation model for pathology — on Google Cloud. Provisions infrastructure, deploys the model to Vertex AI, generates image/text embeddings, runs zero-shot tissue classification, fine-tunes on NCT-CRC-HE-100K, and evaluates on CRC-VAL-HE-7K.

## Quick Start

### 1. Prerequisites

- **Google Cloud project** with billing enabled
- **gcloud CLI** authenticated with Application Default Credentials
- **Node.js** ≥ 18 and **Python** ≥ 3.10

### 2. Configure Your Project

All project-specific values are driven by environment variables. **Export these before starting the backend:**

```bash
# Required — your GCP project ID
export GCP_PROJECT_ID="your-project-id"

# Optional — override defaults if needed
export GCP_REGION="us-central1"                          # default: us-central1
export SERVICE_ACCOUNT_NAME="medsiglip-pipeline-sa"      # default: medsiglip-pipeline-sa
export WORKBENCH_INSTANCE_NAME="medsiglip-researcher-workbench"  # default: medsiglip-researcher-workbench
```

The bucket name is automatically derived as `${GCP_PROJECT_ID}-bucket`.

| Variable | Default | Description |
|----------|---------|-------------|
| `GCP_PROJECT_ID` | `wz-workload-viz-medsiglip` | Your GCP project ID |
| `GCP_REGION` | `us-central1` | Region for all resources |
| `SERVICE_ACCOUNT_NAME` | `medsiglip-pipeline-sa` | SA name for pipeline workloads |
| `WORKBENCH_INSTANCE_NAME` | `medsiglip-researcher-workbench` | Vertex AI Workbench instance name |

### 3. Authenticate to GCP

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project $GCP_PROJECT_ID
```



## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Frontend (React + ReactFlow)                                   │
│  Fetches /api/config → all URLs and project IDs are dynamic    │
├─────────────────────────────────────────────────────────────────┤
│  Backend (Flask)                                                │
│  Reads env vars → provisions GCP infra → polls Vertex AI       │
├─────────────────────────────────────────────────────────────────┤
│  Google Cloud                                                   │
│  Vertex AI • Cloud Storage • Compute • IAM • Healthcare API    │
└─────────────────────────────────────────────────────────────────┘
```

## Pipeline Steps

| # | Step | Actor | Description |
|---|------|-------|-------------|
| 1 | Enable APIs | IT | Vertex AI, Compute, Healthcare, IAM, Org Policy |
| 2 | Create Service Account | IT | `medsiglip-pipeline-sa` with 5 IAM roles |
| 3 | IAM Roles | IT | `aiplatform.user`, `storage.admin`, `batch.jobsEditor`, etc. |
| 4 | Org Policies | IT | Shielded VM + trusted image project exceptions |
| 5 | VPC Network | IT | Default VPC + firewall + Private Google Access |
| 6 | Cloud NAT | IT | Outbound internet for private VMs |
| 7 | Provision Workbench | IT | Vertex AI Workbench with GPU + startup script |
| 8 | Storage Bucket | Researcher | `gs://${PROJECT_ID}-bucket` for images, embeddings, checkpoints |
| 9 | Deploy MedSigLIP | Researcher | Model Garden → Vertex AI Endpoint (T4 GPU) |
| 10 | Generate Embeddings | Researcher | 768-dim image + text vectors via predict API |
| 11 | Zero-Shot Classification | Researcher | Text-prompt tissue classification (9 classes) |
| 12 | Fine-Tune Model | Researcher | HuggingFace Trainer on NCT-CRC-HE-100K |
| 13 | Evaluate Model | Researcher | Accuracy + F1 on CRC-VAL-HE-7K |

To use this template with your own project:

```bash
# 1. Clone the repo
git clone <this-repo-url>
cd workload-viz-medsiglip-pathology

# 2. Export YOUR project ID
export GCP_PROJECT_ID="my-team-project-id"

# 3. Authenticate
gcloud auth application-default login

# 4. Start backend + frontend
cd backend && source venv/bin/activate && python main.py &
cd ../frontend && npm run dev
```

That's it. The frontend automatically fetches your project config from the backend's `/api/config` endpoint. No code changes needed.
