"""
@file main.py
@brief Flask backend for MedSigLIP Pathology Workload Visualizer using Python GCP libraries

@details This backend provides real GCP infrastructure provisioning including:
- VPC network and firewall configuration
- Vertex AI Workbench provisioning for researcher environments
- MedSigLIP model deployment to Vertex AI Endpoints
- Embedding generation and zero-shot classification
- GCS bucket management for DICOM images, embeddings, and checkpoints

@author Willis Zhang
@date 2026-02-11
"""

import json
import os
import time
from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS

# GCP Libraries
from google.cloud import storage
from google.cloud import resourcemanager_v3
from googleapiclient import discovery
from google.auth import default
from google.api_core import exceptions as gcp_exceptions

app = Flask(__name__)
CORS(app)

# Configuration
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "wz-workload-viz-medsiglip")
BUCKET_NAME = f"{PROJECT_ID}-bucket"
SERVICE_ACCOUNT_NAME = os.environ.get("SERVICE_ACCOUNT_NAME", "medsiglip-pipeline-sa")
# Zone fallback list for A100 Workbench provisioning.
# Ordered by A100 availability from GCE Supply dashboard (2026-02-12).
# Notebooks API v2 only supports zones a/b/c per region.
# On STOCKOUT, execute_provision_workbench() tries the next zone and updates
# REGION/ZONE globals so all downstream endpoints (config, polling, bucket) match.
PREFERRED_ZONES = [
    # Asia — best A100 availability (14.58% sold, 123 unsold, 11 net empty hosts)
    'asia-northeast3-b', 'asia-northeast3-a', 'asia-northeast3-c',
    # Europe — good availability (73.91% sold, 96 unsold, 7 net empty)
    'europe-west4-a', 'europe-west4-b', 'europe-west4-c',
    # US — fallback (92.66% sold, 76 unsold, 12 net empty)
    'us-central1-b', 'us-central1-a', 'us-central1-c',
]
REGION = os.environ.get("GCP_REGION", "asia-northeast3")
ZONE = f"{REGION}-b"
WORKBENCH_INSTANCE_NAME = os.environ.get("WORKBENCH_INSTANCE_NAME", "medsiglip-researcher-workbench")


def stream_sse(data: dict) -> str:
    """Format data as Server-Sent Event"""
    return f"data: {json.dumps(data)}\n\n"


def log_msg(msg: str, msg_type: str = "info"):
    """Create a log SSE message"""
    print(f"[LOG] {msg}")
    return stream_sse({"log": msg, "type": msg_type})


def step_complete():
    """Mark step as complete"""
    return stream_sse({"log": "✓ Done", "type": "success", "status": "complete"})


def step_error(msg: str):
    """Mark step as error"""
    return stream_sse({"log": f"✗ {msg}", "type": "error", "status": "error"})


def execute_enable_apis():
    """Enable required GCP APIs using Service Usage API"""
    yield log_msg("Enabling Vertex AI, Compute, Healthcare, and IAM APIs...")
    
    try:
        credentials, project = default()
        service = discovery.build('serviceusage', 'v1', credentials=credentials)
        
        apis = [
            'aiplatform.googleapis.com',
            'compute.googleapis.com',
            'healthcare.googleapis.com',
            'iam.googleapis.com',
            'cloudresourcemanager.googleapis.com',
            'orgpolicy.googleapis.com',
            'notebooks.googleapis.com',
            'storage.googleapis.com',
        ]
        
        for api in apis:
            yield log_msg(f"  Enabling {api}...")
            try:
                request_body = {'consumerId': f'project:{PROJECT_ID}'}
                service.services().enable(
                    name=f'projects/{PROJECT_ID}/services/{api}'
                ).execute()
                yield log_msg(f"  ✓ {api} enabled", "success")
            except Exception as e:
                if "already enabled" in str(e).lower():
                    yield log_msg(f"  ✓ {api} already enabled", "info")
                else:
                    yield log_msg(f"  ⚠ {api}: {str(e)[:100]}", "info")
        
        yield step_complete()
    except Exception as e:
        yield step_error(str(e))


def execute_create_service_account():
    """Create service account using IAM API"""
    yield log_msg(f"Creating service account: {SERVICE_ACCOUNT_NAME}...")
    
    try:
        credentials, project = default()
        service = discovery.build('iam', 'v1', credentials=credentials)
        
        sa_email = f"{SERVICE_ACCOUNT_NAME}@{PROJECT_ID}.iam.gserviceaccount.com"
        
        try:
            # Check if SA exists
            service.projects().serviceAccounts().get(
                name=f"projects/{PROJECT_ID}/serviceAccounts/{sa_email}"
            ).execute()
            yield log_msg(f"  Service account already exists: {sa_email}", "info")
        except:
            # Create SA
            service.projects().serviceAccounts().create(
                name=f"projects/{PROJECT_ID}",
                body={
                    'accountId': SERVICE_ACCOUNT_NAME,
                    'serviceAccount': {
                        'displayName': 'MedSigLIP Pipeline Service Account'
                    }
                }
            ).execute()
            yield log_msg(f"  Created: {sa_email}", "success")
        
        yield step_complete()
    except Exception as e:
        yield step_error(str(e))


def execute_iam_roles():
    """Add IAM roles to service account"""
    yield log_msg("Adding IAM roles to service account...")
    
    try:
        credentials, project = default()
        service = discovery.build('cloudresourcemanager', 'v1', credentials=credentials)
        
        sa_email = f"{SERVICE_ACCOUNT_NAME}@{PROJECT_ID}.iam.gserviceaccount.com"
        member = f"serviceAccount:{sa_email}"
        
        roles = [
            'roles/iam.serviceAccountUser',
            'roles/aiplatform.user',
            'roles/notebooks.admin',
            'roles/logging.viewer',
            'roles/storage.admin',
        ]
        
        # Get current policy
        policy = service.projects().getIamPolicy(
            resource=PROJECT_ID,
            body={}
        ).execute()
        
        # Add roles
        for role in roles:
            yield log_msg(f"  Adding {role}...")
            
            # Check if binding exists
            binding_exists = False
            for binding in policy.get('bindings', []):
                if binding['role'] == role:
                    if member not in binding['members']:
                        binding['members'].append(member)
                    binding_exists = True
                    break
            
            if not binding_exists:
                policy.setdefault('bindings', []).append({
                    'role': role,
                    'members': [member]
                })
        
        # Set updated policy
        service.projects().setIamPolicy(
            resource=PROJECT_ID,
            body={'policy': policy}
        ).execute()
        
        for role in roles:
            yield log_msg(f"  ✓ {role} granted", "success")
        
        yield step_complete()
    except Exception as e:
        yield step_error(str(e))


def execute_configure_org_policies():
    """
    Override org policy constraints that block GPU workbench provisioning.
    
    compute.requireShieldedVm: When enforced (inherited from org), requires
    Secure Boot on all VMs. Secure Boot blocks NVIDIA proprietary GPU drivers
    because they are unsigned kernel modules. We override this at the project
    level to allow non-secure-boot VMs for A100 GPU workbenches.
    """
    yield log_msg("Configuring org policy overrides for GPU workbench...")
    
    try:
        credentials, project = default()
        
        # Use Org Policy API v2
        orgpolicy_service = discovery.build('orgpolicy', 'v2', credentials=credentials)
        
        # Override compute.requireShieldedVm → enforce: false
        # Required because NVIDIA GPU drivers are unsigned kernel modules
        # and Secure Boot blocks unsigned module loading
        policy_name = f"projects/{PROJECT_ID}/policies/compute.requireShieldedVm"
        
        yield log_msg("  Disabling compute.requireShieldedVm (required for NVIDIA GPU drivers)...")
        
        try:
            policy_body = {
                "name": policy_name,
                "spec": {
                    "rules": [{"enforce": False}]
                }
            }
            
            # Try create first (policy doesn't exist at project level yet, only inherited)
            try:
                orgpolicy_service.projects().policies().create(
                    parent=f"projects/{PROJECT_ID}",
                    body=policy_body
                ).execute()
                yield log_msg("  ✓ compute.requireShieldedVm overridden (enforce: false)", "success")
            except Exception as create_err:
                create_str = str(create_err)
                if '409' in create_str or 'already exists' in create_str.lower():
                    # Policy already exists at project level — update it
                    orgpolicy_service.projects().policies().patch(
                        name=policy_name,
                        body=policy_body
                    ).execute()
                    yield log_msg("  ✓ compute.requireShieldedVm updated (enforce: false)", "success")
                else:
                    raise create_err
            
            yield log_msg("  NVIDIA unsigned kernel modules can now load on VMs", "info")
        except Exception as e:
            err_str = str(e)
            if 'PERMISSION_DENIED' in err_str or '403' in err_str:
                yield log_msg(f"  ⚠ Permission denied — need orgpolicy.policyAdmin role", "error")
                yield log_msg(f"  Run: gcloud org-policies set-policy --project={PROJECT_ID} with enforce:false", "info")
                yield step_error("Missing orgpolicy.policyAdmin permission")
                return
            else:
                yield log_msg(f"  ⚠ Org policy override: {err_str[:120]}", "error")
                raise e
        
        yield step_complete()
    except Exception as e:
        yield step_error(str(e))


def execute_create_network():
    """Create VPC network and firewall rules for Vertex AI Workbench"""
    yield log_msg("Setting up VPC network for Vertex AI Workbench...")
    
    try:
        credentials, project = default()
        compute_service = discovery.build('compute', 'v1', credentials=credentials)
        
        # Check if default network exists
        try:
            compute_service.networks().get(
                project=PROJECT_ID,
                network='default'
            ).execute()
            yield log_msg("  ✓ Default VPC network already exists", "info")
        except Exception as e:
            if 'notFound' in str(e) or '404' in str(e):
                yield log_msg("  Creating default VPC network with auto-subnets...")
                
                network_body = {
                    'name': 'default',
                    'autoCreateSubnetworks': True,
                    'routingConfig': {
                        'routingMode': 'REGIONAL'
                    }
                }
                
                operation = compute_service.networks().insert(
                    project=PROJECT_ID,
                    body=network_body
                ).execute()
                
                # Wait for operation to complete
                yield log_msg("  Waiting for network creation...")
                while True:
                    result = compute_service.globalOperations().get(
                        project=PROJECT_ID,
                        operation=operation['name']
                    ).execute()
                    if result['status'] == 'DONE':
                        break
                
                yield log_msg("  ✓ Default VPC network created", "success")
            else:
                raise e
        
        # Check/create firewall rule for internal traffic
        firewall_name = 'default-allow-internal'
        try:
            compute_service.firewalls().get(
                project=PROJECT_ID,
                firewall=firewall_name
            ).execute()
            yield log_msg(f"  ✓ Firewall rule '{firewall_name}' already exists", "info")
        except Exception as e:
            if 'notFound' in str(e) or '404' in str(e):
                yield log_msg(f"  Creating firewall rule '{firewall_name}'...")
                
                firewall_body = {
                    'name': firewall_name,
                    'network': f'projects/{PROJECT_ID}/global/networks/default',
                    'direction': 'INGRESS',
                    'priority': 1000,
                    'allowed': [
                        {'IPProtocol': 'tcp'},
                        {'IPProtocol': 'udp'},
                        {'IPProtocol': 'icmp'}
                    ],
                    'sourceRanges': ['10.128.0.0/9']
                }
                
                operation = compute_service.firewalls().insert(
                    project=PROJECT_ID,
                    body=firewall_body
                ).execute()
                
                # Wait for operation to complete
                yield log_msg("  Waiting for firewall rule creation...")
                while True:
                    result = compute_service.globalOperations().get(
                        project=PROJECT_ID,
                        operation=operation['name']
                    ).execute()
                    if result['status'] == 'DONE':
                        break
                
                yield log_msg(f"  ✓ Firewall rule '{firewall_name}' created", "success")
            else:
                raise e
        
        # Enable Private Google Access on default subnet (required for internal-only VMs)
        yield log_msg("  Enabling Private Google Access on subnet...")
        try:
            subnet = compute_service.subnetworks().get(
                project=PROJECT_ID,
                region=REGION,
                subnetwork='default'
            ).execute()
            
            if subnet.get('privateIpGoogleAccess', False):
                yield log_msg("  ✓ Private Google Access already enabled", "info")
            else:
                compute_service.subnetworks().setPrivateIpGoogleAccess(
                    project=PROJECT_ID,
                    region=REGION,
                    subnetwork='default',
                    body={'privateIpGoogleAccess': True}
                ).execute()
                yield log_msg("  ✓ Private Google Access enabled", "success")
        except Exception as e:
            yield log_msg(f"  ⚠ Could not enable Private Google Access: {str(e)[:80]}", "info")
        
        yield log_msg("  Network: default (auto-subnets)", "info")
        yield log_msg("  Firewall: Internal traffic allowed (10.128.0.0/9)", "info")
        yield log_msg("  Private Google Access: Enabled (for internal-only VMs)", "info")
        yield step_complete()
    except Exception as e:
        yield step_error(str(e))


def execute_configure_cloud_nat():
    """
    Configure Cloud NAT to allow VMs without public IPs to access the internet.
    This is required for the workbench to download packages and pull model weights.
    """
    yield log_msg("Configuring Cloud NAT for outbound internet access...")
    
    try:
        credentials, project = default()
        compute_service = discovery.build('compute', 'v1', credentials=credentials)
        
        router_name = 'nat-router'
        nat_name = 'nat-config'
        
        # Check if router already exists
        try:
            compute_service.routers().get(
                project=PROJECT_ID,
                region=REGION,
                router=router_name
            ).execute()
            yield log_msg(f"  ✓ Cloud Router '{router_name}' already exists", "info")
        except Exception as e:
            if 'notFound' in str(e).lower() or '404' in str(e):
                yield log_msg(f"  Creating Cloud Router '{router_name}'...")
                
                router_body = {
                    'name': router_name,
                    'network': f'projects/{PROJECT_ID}/global/networks/default',
                    'region': REGION
                }
                
                operation = compute_service.routers().insert(
                    project=PROJECT_ID,
                    region=REGION,
                    body=router_body
                ).execute()
                
                # Wait for operation
                yield log_msg("  Waiting for router creation...")
                while True:
                    result = compute_service.regionOperations().get(
                        project=PROJECT_ID,
                        region=REGION,
                        operation=operation['name']
                    ).execute()
                    if result['status'] == 'DONE':
                        break
                    time.sleep(2)
                
                yield log_msg(f"  ✓ Cloud Router '{router_name}' created", "success")
            else:
                raise e
        
        # Check if NAT config already exists on the router
        try:
            router = compute_service.routers().get(
                project=PROJECT_ID,
                region=REGION,
                router=router_name
            ).execute()
            
            existing_nats = router.get('nats', [])
            nat_exists = any(n.get('name') == nat_name for n in existing_nats)
            
            if nat_exists:
                yield log_msg(f"  ✓ NAT config '{nat_name}' already exists", "info")
            else:
                yield log_msg(f"  Adding NAT config '{nat_name}' to router...")
                
                # Add NAT configuration to the router
                nat_config = {
                    'name': nat_name,
                    'natIpAllocateOption': 'AUTO_ONLY',
                    'sourceSubnetworkIpRangesToNat': 'ALL_SUBNETWORKS_ALL_IP_RANGES',
                    'logConfig': {
                        'enable': False,
                        'filter': 'ALL'
                    }
                }
                
                router['nats'] = existing_nats + [nat_config]
                
                operation = compute_service.routers().patch(
                    project=PROJECT_ID,
                    region=REGION,
                    router=router_name,
                    body=router
                ).execute()
                
                # Wait for operation
                yield log_msg("  Waiting for NAT configuration...")
                while True:
                    result = compute_service.regionOperations().get(
                        project=PROJECT_ID,
                        region=REGION,
                        operation=operation['name']
                    ).execute()
                    if result['status'] == 'DONE':
                        break
                    time.sleep(2)
                
                yield log_msg(f"  ✓ NAT config '{nat_name}' added", "success")
        except Exception as e:
            raise e
        
        yield log_msg("  Cloud NAT enables outbound internet for private VMs", "info")
        yield log_msg("  Workbench can now install packages (torch, transformers)", "info")
        yield step_complete()
    except Exception as e:
        yield step_error(str(e))


def execute_provision_workbench():
    """
    Provision a Vertex AI Workbench instance for researchers.
    Uses the Notebooks API v2 (notebooks.googleapis.com/v2) to create a Workbench Instance.
    Tries zones from PREFERRED_ZONES in order; on STOCKOUT, skips to next zone.
    Updates global REGION/ZONE so all downstream endpoints match the resolved zone.
    If instance already exists in the current ZONE, returns its URL.
    """
    global REGION, ZONE

    yield log_msg(f"Provisioning Vertex AI Workbench: {WORKBENCH_INSTANCE_NAME}...")

    try:
        credentials, project = default()

        # Enable the Notebooks API if not already enabled
        yield log_msg("  Enabling notebooks.googleapis.com API...")
        try:
            service_usage = discovery.build('serviceusage', 'v1', credentials=credentials)
            service_usage.services().enable(
                name=f'projects/{PROJECT_ID}/services/notebooks.googleapis.com'
            ).execute()
            yield log_msg("  ✓ Notebooks API enabled", "success")
        except Exception as e:
            if "already enabled" in str(e).lower():
                yield log_msg("  ✓ Notebooks API already enabled", "info")
            else:
                yield log_msg(f"  ⚠ Notebooks API: {str(e)[:80]}", "info")

        notebooks_service = discovery.build('notebooks', 'v2', credentials=credentials)
        workbench_url = f"https://console.cloud.google.com/vertex-ai/workbench/instances?project={PROJECT_ID}"
        jupyter_url = None

        # Check if instance already exists in the current default zone
        instance_name = f"projects/{PROJECT_ID}/locations/{ZONE}/instances/{WORKBENCH_INSTANCE_NAME}"
        yield log_msg(f"  Checking for existing instance in {ZONE}...")
        try:
            instance = notebooks_service.projects().locations().instances().get(
                name=instance_name
            ).execute()
            state = instance.get('state', 'UNKNOWN')
            yield log_msg(f"  ✓ Workbench instance already exists (state: {state})", "info")
            if 'proxyUri' in instance:
                jupyter_url = instance['proxyUri']
                yield log_msg(f"  JupyterLab URL: {jupyter_url}", "success")
            yield stream_sse({
                "log": f"Workbench ready: {WORKBENCH_INSTANCE_NAME}",
                "type": "success",
                "workbenchUrl": workbench_url,
                "jupyterUrl": jupyter_url,
                "instanceName": WORKBENCH_INSTANCE_NAME,
                "status": "complete"
            })
            return
        except Exception as e:
            err_str = str(e)
            if 'notFound' in err_str.lower() or '404' in err_str:
                yield log_msg("  Instance not found, will create new one...", "info")
            elif 'SERVICE_DISABLED' in err_str or 'has not been used' in err_str:
                yield log_msg("  ⏳ Notebooks API propagating, proceeding to create...", "info")
                time.sleep(15)
                notebooks_service = discovery.build('notebooks', 'v2', credentials=credentials)
            else:
                raise e

        # --- Zone fallback loop: try each zone in PREFERRED_ZONES ---
        sa_email = f"{SERVICE_ACCOUNT_NAME}@{PROJECT_ID}.iam.gserviceaccount.com"

        startup_script = f'''#!/bin/bash
set -e
echo "=== MedSigLIP A100 Workbench Setup ==="
nvidia-smi || echo "WARNING: nvidia-smi not found, GPU drivers may need install"
mkdir -p /home/jupyter/medsiglip-workspace

# Create the MedSigLIP pathology notebook
cat > /home/jupyter/medsiglip-workspace/MedSigLIP_Pathology_Pipeline.ipynb << 'NOTEBOOK'
{{
  "cells": [
    {{
      "cell_type": "markdown",
      "metadata": {{}},
      "source": ["# MedSigLIP Pathology Pipeline on A100\\n", "\\n", "This notebook runs the complete MedSigLIP pipeline on GPU:\\n", "1. Load model from HuggingFace\\n", "2. Download NCT-CRC-HE-100K dataset\\n", "3. Zero-shot baseline on CRC-VAL-HE-7K\\n", "4. Fine-tune with HuggingFace Trainer\\n", "5. Evaluate & compare pretrained vs fine-tuned\\n", "\\n", "Based on: https://github.com/google-health/medsiglip/blob/main/notebooks/fine_tune_with_hugging_face.ipynb"]
    }},
    {{
      "cell_type": "markdown",
      "metadata": {{}},
      "source": ["## Step 1: Setup & GPU Verification"]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": ["!pip install -q --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124\\n", "!pip install -q --force-reinstall \\"transformers==4.45.2\\"\\n", "!pip install -q datasets huggingface_hub accelerate evaluate\\n", "!pip install -q google-cloud-aiplatform google-cloud-storage\\n", "!pip install -q scikit-learn matplotlib pillow tqdm\\n", "!pip install -q sentencepiece protobuf\\n", "\\n", "# After finishes, restart kernel"]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": ["import torch\\n", "print(f\\"GPU: {{torch.cuda.get_device_name(0)}}\\")", "\\n", "print(f\\"VRAM: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB\\")", "\\n", "assert torch.cuda.is_available(), \\"No GPU found!\\"\\n", "\\n", "PROJECT_ID = \\"{PROJECT_ID}\\"\\n", "BUCKET_NAME = \\"{BUCKET_NAME}\\""]
    }},
    {{
      "cell_type": "markdown",
      "metadata": {{}},
      "source": ["## Step 2: Create GCS Bucket"]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": ["!gcloud storage buckets describe gs://{BUCKET_NAME} 2>/dev/null || gcloud storage buckets create gs://{BUCKET_NAME} --project={PROJECT_ID}\\n", "print(f\\"Bucket ready: gs://{BUCKET_NAME}\\")"]
    }},
    {{
      "cell_type": "markdown",
      "metadata": {{}},
      "source": ["## Step 3: HuggingFace Auth & Load MedSigLIP"]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": ["# Retrieve read token from https://huggingface.co/settings/tokens\\n", "\\n", "from huggingface_hub import get_token\\n", "if get_token() is None:\\n", "    from huggingface_hub import notebook_login\\n", "    notebook_login()\\n", "\\n", "from transformers import SiglipProcessor, SiglipModel\\n", "\\n", "model_id = \\"google/medsiglip-448\\"\\n", "model = SiglipModel.from_pretrained(model_id)\\n", "processor = SiglipProcessor.from_pretrained(model_id)\\n", "print(f\\"Loaded {{model_id}}\\")\\n", "\\n", "# Write status marker\\n", "import json, datetime\\n", "from google.cloud import storage as gcs\\n", "client = gcs.Client()\\n", "bucket = client.bucket(\\"{BUCKET_NAME}\\")\\n", "bucket.blob(\\"status/load-model.json\\").upload_from_string(\\n", "    json.dumps({{\\"step\\": \\"load-model\\", \\"status\\": \\"complete\\", \\"timestamp\\": datetime.datetime.now().isoformat()}})\\n", ")"]
    }},
    {{
      "cell_type": "markdown",
      "metadata": {{}},
      "source": ["## Step 4: Download NCT-CRC-HE-100K + CRC-VAL-HE-7K"]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": ["!wget -nc -q \\"https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip\\"\\n", "!wget -nc -q \\"https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip\\"\\n", "!unzip -qn NCT-CRC-HE-100K.zip\\n", "!unzip -qn CRC-VAL-HE-7K.zip\\n", "print(\\"Datasets downloaded and extracted.\\")\\n", "\\n", "# Write status marker\\n", "bucket.blob(\\"status/download-dataset.json\\").upload_from_string(\\n", "    json.dumps({{\\"step\\": \\"download-dataset\\", \\"status\\": \\"complete\\", \\"timestamp\\": datetime.datetime.now().isoformat()}})\\n", ")"]
    }},
    {{
      "cell_type": "markdown",
      "metadata": {{}},
      "source": ["## Step 5: Prepare Training Dataset"]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": ["from datasets import load_dataset\\n", "from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode\\n", "\\n", "train_size = 9000\\n", "validation_size = 1000\\n", "\\n", "data = load_dataset(\\"./NCT-CRC-HE-100K\\", split=\\"train\\")\\n", "data = data.train_test_split(train_size=train_size, test_size=validation_size, shuffle=True, seed=42)\\n", "data[\\"validation\\"] = data.pop(\\"test\\")\\n", "\\n", "TISSUE_CLASSES = [\\n", "    \\"adipose\\", \\"background\\", \\"debris\\", \\"lymphocytes\\", \\"mucus\\",\\n", "    \\"smooth muscle\\", \\"normal colon mucosa\\", \\"cancer-associated stroma\\",\\n", "    \\"colorectal adenocarcinoma epithelium\\"\\n", "]\\n", "\\n", "size = processor.image_processor.size[\\"height\\"]\\n", "mean = processor.image_processor.image_mean\\n", "std = processor.image_processor.image_std\\n", "\\n", "_transform = Compose([\\n", "    Resize((size, size), interpolation=InterpolationMode.BILINEAR),\\n", "    ToTensor(),\\n", "    Normalize(mean=mean, std=std),\\n", "])\\n", "\\n", "def preprocess(examples):\\n", "    pixel_values = [_transform(image.convert(\\"RGB\\")) for image in examples[\\"image\\"]]\\n", "    captions = [TISSUE_CLASSES[label] for label in examples[\\"label\\"]]\\n", "    inputs = processor.tokenizer(captions, max_length=64, padding=\\"max_length\\", truncation=True, return_attention_mask=True)\\n", "    inputs[\\"pixel_values\\"] = pixel_values\\n", "    return inputs\\n", "\\n", "data = data.map(preprocess, batched=True, remove_columns=[\\"image\\", \\"label\\"])\\n", "print(f\\"Train: {{len(data[chr(116)+chr(114)+chr(97)+chr(105)+chr(110)])}}, Val: {{len(data[chr(118)+chr(97)+chr(108)+chr(105)+chr(100)+chr(97)+chr(116)+chr(105)+chr(111)+chr(110)])}}\\")"]
    }},
    {{
      "cell_type": "markdown",
      "metadata": {{}},
      "source": ["## Step 6: Zero-Shot Baseline (Pretrained Model)"]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": ["import io, evaluate\\n", "from PIL import Image\\n", "\\n", "test_data = load_dataset(\\"./CRC-VAL-HE-7K\\", split=\\"train\\")\\n", "test_data = test_data.shuffle(seed=42).select(range(1000))\\n", "test_batches = test_data.batch(batch_size=64)\\n", "\\n", "accuracy_metric = evaluate.load(\\"accuracy\\")\\n", "f1_metric = evaluate.load(\\"f1\\")\\n", "REFERENCES = test_data[\\"label\\"]\\n", "\\n", "def compute_metrics(predictions):\\n", "    metrics = {{}}\\n", "    metrics.update(accuracy_metric.compute(predictions=predictions, references=REFERENCES))\\n", "    metrics.update(f1_metric.compute(predictions=predictions, references=REFERENCES, average=\\"weighted\\"))\\n", "    return metrics\\n", "\\n", "pt_model = SiglipModel.from_pretrained(model_id, device_map=\\"auto\\")\\n", "pt_predictions = []\\n", "for batch in test_batches:\\n", "    images = [Image.open(io.BytesIO(image[\\"bytes\\"])) for image in batch[\\"image\\"]]\\n", "    inputs = processor(text=TISSUE_CLASSES, images=images, padding=\\"max_length\\", return_tensors=\\"pt\\").to(\\"cuda\\")\\n", "    with torch.no_grad():\\n", "        outputs = pt_model(**inputs)\\n", "    pt_predictions.extend(outputs.logits_per_image.argmax(axis=1).tolist())\\n", "\\n", "pt_metrics = compute_metrics(pt_predictions)\\n", "print(f\\"Pretrained baseline: {{pt_metrics}}\\")"]
    }},
    {{
      "cell_type": "markdown",
      "metadata": {{}},
      "source": ["## Step 7: Fine-Tune with Contrastive Loss"]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": ["from transformers import TrainingArguments, Trainer\\n", "\\n", "def collate_fn(examples):\\n", "    pixel_values = torch.tensor([ex[\\"pixel_values\\"] for ex in examples])\\n", "    input_ids = torch.tensor([ex[\\"input_ids\\"] for ex in examples])\\n", "    attention_mask = torch.tensor([ex[\\"attention_mask\\"] for ex in examples])\\n", "    return {{\\"pixel_values\\": pixel_values, \\"input_ids\\": input_ids, \\"attention_mask\\": attention_mask, \\"return_loss\\": True}}\\n", "\\n", "training_args = TrainingArguments(\\n", "    output_dir=\\"medsiglip-448-ft-crc100k\\",\\n", "    num_train_epochs=2,\\n", "    per_device_train_batch_size=8,\\n", "    per_device_eval_batch_size=8,\\n", "    gradient_accumulation_steps=8,\\n", "    logging_steps=50,\\n", "    save_strategy=\\"epoch\\",\\n", "    eval_strategy=\\"steps\\",\\n", "    eval_steps=50,\\n", "    learning_rate=1e-4,\\n", "    weight_decay=0.01,\\n", "    warmup_steps=5,\\n", "    lr_scheduler_type=\\"cosine\\",\\n", "    push_to_hub=False,\\n", "    report_to=\\"tensorboard\\",\\n", ")\\n", "\\n", "trainer = Trainer(\\n", "    model=model,\\n", "    args=training_args,\\n", "    train_dataset=data[\\"train\\"],\\n", "    eval_dataset=data[\\"validation\\"].shuffle().select(range(200)),\\n", "    data_collator=collate_fn,\\n", ")\\n", "\\n", "trainer.train()\\n", "trainer.save_model()\\n", "print(\\"Fine-tuning complete! Model saved to medsiglip-448-ft-crc100k/\\")\\n", "\\n", "# Write status marker\\n", "bucket.blob(\\"status/fine-tune.json\\").upload_from_string(\\n", "    json.dumps({{\\"step\\": \\"fine-tune\\", \\"status\\": \\"complete\\", \\"timestamp\\": datetime.datetime.now().isoformat()}})\\n", ")"]
    }},
    {{
      "cell_type": "markdown",
      "metadata": {{}},
      "source": ["## Step 8: Evaluate Fine-Tuned Model & Compare"]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": ["# Re-create test batches (iterator was consumed in baseline eval)\n", "test_data = load_dataset(\"./CRC-VAL-HE-7K\", split=\"train\")\n", "test_data = test_data.shuffle(seed=42).select(range(1000))\n", "test_batches = test_data.batch(batch_size=64)\n", "REFERENCES = test_data[\"label\"]\n", "\n", "ft_model = SiglipModel.from_pretrained(\\"medsiglip-448-ft-crc100k\\", device_map=\\"auto\\")\\n", "\\n", "ft_predictions = []\\n", "for batch in test_batches:\\n", "    images = [Image.open(io.BytesIO(image[\\"bytes\\"])) for image in batch[\\"image\\"]]\\n", "    inputs = processor(text=TISSUE_CLASSES, images=images, padding=\\"max_length\\", return_tensors=\\"pt\\").to(\\"cuda\\")\\n", "    with torch.no_grad():\\n", "        outputs = ft_model(**inputs)\\n", "    ft_predictions.extend(outputs.logits_per_image.argmax(axis=1).tolist())\\n", "\\n", "ft_metrics = compute_metrics(ft_predictions)\\n", "print(f\\"Pretrained: {{pt_metrics}}\\")\\n", "print(f\\"Fine-tuned: {{ft_metrics}}\\")\\n", "print(f\\"Accuracy improvement: {{ft_metrics['accuracy'] - pt_metrics['accuracy']:.4f}}\\")"]
    }},
    {{
      "cell_type": "markdown",
      "metadata": {{}},
      "source": ["## Step 9: Save Results to GCS"]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "outputs": [],
      "source": ["import json as _json\\n", "results = {{\\"pretrained\\": pt_metrics, \\"fine_tuned\\": ft_metrics}}\\n", "with open(\\"evaluation_results.json\\", \\"w\\") as f:\\n", "    _json.dump(results, f, indent=2)\\n", "\\n", "!gsutil cp evaluation_results.json gs://{BUCKET_NAME}/results/\\n", "!gsutil cp -r medsiglip-448-ft-crc100k/ gs://{BUCKET_NAME}/medsiglip-finetune/\\n", "print(f\\"Results saved to gs://{BUCKET_NAME}/results/\\")\\n", "print(f\\"Checkpoints saved to gs://{BUCKET_NAME}/medsiglip-finetune/\\")"]
    }}
  ],
  "metadata": {{
    "kernelspec": {{
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    }}
  }},
  "nbformat": 4,
  "nbformat_minor": 4
}}
NOTEBOOK

chown -R jupyter:jupyter /home/jupyter/medsiglip-workspace
echo "=== Setup complete. Open medsiglip-workspace/ in JupyterLab ==="
'''

        yield log_msg(f"  Trying {len(PREFERRED_ZONES)} zones for A100 availability...", "info")

        for zone_candidate in PREFERRED_ZONES:
            candidate_region = zone_candidate.rsplit('-', 1)[0]  # e.g. 'asia-northeast3-b' → 'asia-northeast3'

            instance_body = {
                'gceSetup': {
                    'machineType': 'a2-highgpu-1g',
                    'acceleratorConfigs': [{'type': 'NVIDIA_TESLA_A100', 'coreCount': '1'}],
                    'serviceAccounts': [{'email': sa_email, 'scopes': ['https://www.googleapis.com/auth/cloud-platform']}],
                    'networkInterfaces': [{
                        'network': f'projects/{PROJECT_ID}/global/networks/default',
                        'subnet': f'projects/{PROJECT_ID}/regions/{candidate_region}/subnetworks/default',
                        'nicType': 'VIRTIO_NET'
                    }],
                    'disablePublicIp': True,
                    'metadata': {'startup-script': startup_script, 'proxy-mode': 'service_account'},
                    'bootDisk': {'diskSizeGb': '300', 'diskType': 'PD_SSD'},
                    'vmImage': {'project': 'cloud-notebooks-managed', 'name': 'workbench-instances-v20260122'},
                    'shieldedInstanceConfig': {'enableSecureBoot': False, 'enableVtpm': True, 'enableIntegrityMonitoring': True}
                }
            }

            yield log_msg(f"  → Trying zone: {zone_candidate} (region: {candidate_region})...", "info")

            try:
                operation = notebooks_service.projects().locations().instances().create(
                    parent=f"projects/{PROJECT_ID}/locations/{zone_candidate}",
                    instanceId=WORKBENCH_INSTANCE_NAME,
                    body=instance_body
                ).execute()
            except Exception as e:
                err_str = str(e)
                if 'STOCKOUT' in err_str or 'does not have enough resources' in err_str:
                    yield log_msg(f"  ✗ {zone_candidate}: STOCKOUT — no A100 capacity", "error")
                    continue
                elif 'already exists' in err_str.lower():
                    yield log_msg(f"  ✓ Instance already exists in {zone_candidate}", "info")
                    REGION = candidate_region
                    ZONE = zone_candidate
                    yield stream_sse({
                        "log": f"Workbench ready: {WORKBENCH_INSTANCE_NAME}",
                        "type": "success",
                        "workbenchUrl": workbench_url,
                        "instanceName": WORKBENCH_INSTANCE_NAME,
                        "status": "complete"
                    })
                    return
                else:
                    yield log_msg(f"  ✗ {zone_candidate}: {err_str[:120]}", "error")
                    continue

            # Creation request accepted — update globals to this zone
            REGION = candidate_region
            ZONE = zone_candidate
            yield log_msg(f"  ✓ Creation accepted in {zone_candidate}!", "success")
            yield log_msg(f"  Machine: a2-highgpu-1g (A100 40GB GPU), Zone: {ZONE}", "info")
            yield log_msg(f"  Region/Zone updated globally → {REGION}/{ZONE}", "info")

            operation_name = operation.get('name')
            yield log_msg(f"  Operation: {operation_name.split('/')[-1]}", "info")

            # Poll for operation completion
            instance_name = f"projects/{PROJECT_ID}/locations/{ZONE}/instances/{WORKBENCH_INSTANCE_NAME}"
            max_wait = 600
            poll_interval = 15
            elapsed = 0

            while elapsed < max_wait:
                op_result = notebooks_service.projects().locations().operations().get(
                    name=operation_name
                ).execute()

                if op_result.get('done'):
                    if 'error' in op_result:
                        err_msg = op_result['error'].get('message', 'Unknown error')
                        if 'STOCKOUT' in err_msg or 'does not have enough resources' in err_msg:
                            yield log_msg(f"  ✗ {zone_candidate}: STOCKOUT during provisioning", "error")
                            break  # break inner while → continue to next zone
                        yield step_error(f"Failed in {zone_candidate}: {err_msg}")
                        return

                    yield log_msg("  ✓ Workbench instance created successfully!", "success")

                    instance = notebooks_service.projects().locations().instances().get(
                        name=instance_name
                    ).execute()

                    if 'proxyUri' in instance:
                        jupyter_url = instance['proxyUri']
                        yield log_msg(f"  JupyterLab URL: {jupyter_url}", "success")

                    yield stream_sse({
                        "log": f"Workbench ready: {WORKBENCH_INSTANCE_NAME} in {ZONE}",
                        "type": "success",
                        "workbenchUrl": workbench_url,
                        "jupyterUrl": jupyter_url,
                        "instanceName": WORKBENCH_INSTANCE_NAME,
                        "resolvedRegion": REGION,
                        "resolvedZone": ZONE,
                        "status": "complete"
                    })
                    return

                elapsed += poll_interval
                yield log_msg(f"  Provisioning in {zone_candidate}... ({elapsed}s elapsed)", "info")
                time.sleep(poll_interval)
            else:
                # max_wait exceeded but no error — workbench still provisioning
                yield log_msg(f"  ⚠ Still provisioning in {zone_candidate} (check console)", "info")
                yield stream_sse({
                    "log": f"Workbench provisioning in progress ({ZONE})",
                    "type": "info",
                    "workbenchUrl": workbench_url,
                    "instanceName": WORKBENCH_INSTANCE_NAME,
                    "resolvedRegion": REGION,
                    "resolvedZone": ZONE,
                    "status": "complete"
                })
                return

        # All zones exhausted
        yield step_error(f"All {len(PREFERRED_ZONES)} zones stocked out. No A100 capacity available.")

    except Exception as e:
        print(f"[ERROR] Workbench provisioning failed: {str(e)}")
        yield step_error(str(e))


def execute_create_bucket():
    """Create GCS bucket using google-cloud-storage"""
    yield log_msg(f"Creating GCS bucket: gs://{BUCKET_NAME}...")
    
    try:
        client = storage.Client(project=PROJECT_ID)
        
        try:
            bucket = client.get_bucket(BUCKET_NAME)
            yield log_msg(f"  Bucket already exists: gs://{BUCKET_NAME}", "info")
        except gcp_exceptions.NotFound:
            bucket = client.create_bucket(BUCKET_NAME, location=REGION)
            yield log_msg(f"  Created bucket: gs://{BUCKET_NAME} in {REGION}", "success")
        
        yield log_msg(f"  Location: {bucket.location}", "info")
        yield step_complete()
    except Exception as e:
        yield step_error(str(e))




def task_update(task_id: str, status: str, message: str = ""):
    """Send a task-specific status update SSE event"""
    return stream_sse({
        "type": "task_update",
        "task": task_id,
        "status": status,
        "message": message
    })




def execute_check_vertex_ai_status():
    """
    Check Vertex AI endpoint and model deployment status.
    Used as a generic status checker for MedSigLIP pipeline steps.
    These steps are researcher-driven from notebook cells — this just checks current state.
    """
    yield log_msg("Checking Vertex AI resources...")
    
    try:
        credentials, project = default()
        ai_service = discovery.build('aiplatform', 'v1', credentials=credentials)
        
        # List endpoints
        parent = f"projects/{PROJECT_ID}/locations/{REGION}"
        endpoints_response = ai_service.projects().locations().endpoints().list(
            parent=parent
        ).execute()
        
        endpoints = endpoints_response.get('endpoints', [])
        medsiglip_endpoints = [e for e in endpoints if 'medsiglip' in e.get('displayName', '').lower()]
        
        yield log_msg(f"  Found {len(endpoints)} total endpoints, {len(medsiglip_endpoints)} MedSigLIP", "info")
        
        for ep in medsiglip_endpoints:
            name = ep.get('displayName', 'unknown')
            deployed = len(ep.get('deployedModels', []))
            yield log_msg(f"  • {name}: {deployed} model(s) deployed", "success" if deployed > 0 else "info")
        
        # List models
        models_response = ai_service.projects().locations().models().list(
            parent=parent
        ).execute()
        
        models = models_response.get('models', [])
        medsiglip_models = [m for m in models if 'medsiglip' in m.get('displayName', '').lower()]
        
        yield log_msg(f"  Found {len(medsiglip_models)} MedSigLIP model(s) in registry", "info")
        
        # Check GCS for pipeline artifacts
        try:
            client = storage.Client(project=PROJECT_ID)
            bucket = client.get_bucket(BUCKET_NAME)
            
            prefixes = ['medsiglip-finetune/', 'results/', 'embeddings/']
            for prefix in prefixes:
                blobs = list(bucket.list_blobs(prefix=prefix, max_results=5))
                if blobs:
                    yield log_msg(f"  • gs://{BUCKET_NAME}/{prefix} — {len(blobs)} file(s)", "success")
        except Exception:
            pass
        
        yield step_complete()
    except Exception as e:
        yield log_msg(f"  Could not check Vertex AI: {str(e)[:100]}", "info")
        yield step_complete()


# Map step IDs to executor functions
STEP_EXECUTORS = {
    # Infrastructure setup phase (platform team)
    'enable-apis': execute_enable_apis,
    'create-sa': execute_create_service_account,
    'iam-roles': execute_iam_roles,
    'org-policies': execute_configure_org_policies,
    'create-network': execute_create_network,
    'configure-cloud-nat': execute_configure_cloud_nat,
    # Researcher environment provisioning
    'provision-workbench': execute_provision_workbench,
    # Researcher workflow (triggered from notebook cells, but we visualize)
    'storage-bucket': execute_create_bucket,
    # MedSigLIP pipeline steps (researcher-driven from notebook, polled via GCS markers)
    'load-model': execute_check_vertex_ai_status,
    'download-dataset': execute_check_vertex_ai_status,
    'fine-tune': execute_check_vertex_ai_status,
    'save-results': execute_check_vertex_ai_status,
}


@app.route('/api/execute', methods=['POST'])
def execute_step():
    """Execute a workflow step and stream output via SSE"""
    data = request.get_json()
    step_id = data.get('stepId', '')
    phase = data.get('phase', 'setup')

    print(f"\n{'='*60}")
    print(f"Executing step: {step_id} (phase: {phase})")
    print(f"{'='*60}\n")

    def generate():
        if step_id in STEP_EXECUTORS:
            yield from STEP_EXECUTORS[step_id]()
        else:
            yield log_msg(f"Unknown step: {step_id}", "error")
            yield step_error(f"Unknown step: {step_id}")

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "project": PROJECT_ID}


@app.route('/api/config', methods=['GET'])
def get_config():
    """
    Return project configuration to the frontend.
    All values are driven by environment variables so colleagues can
    export their own PROJECT_ID and run the same codebase.
    """
    return jsonify({
        "projectId": PROJECT_ID,
        "bucketName": BUCKET_NAME,
        "region": REGION,
        "zone": ZONE,
        "serviceAccountName": SERVICE_ACCOUNT_NAME,
        "workbenchInstanceName": WORKBENCH_INSTANCE_NAME,
        "consoleBaseUrl": "https://console.cloud.google.com",
    })


@app.route('/api/poll-jobs', methods=['GET'])
def poll_jobs():
    """
    Poll Vertex AI for endpoint and model status.
    Used by frontend to animate MedSigLIP pipeline progress in real-time.
    
    Returns JSON with task statuses mapped to MedSigLIP pipeline steps.
    Infers step completion from: deployed endpoints, GCS artifacts (embeddings,
    fine-tuned checkpoints, evaluation results).
    """
    print(f"\n[POLL] Polling Vertex AI endpoints and GCS artifacts...")
    
    try:
        credentials, _ = default()
        ai_service = discovery.build('aiplatform', 'v1', credentials=credentials)
        
        parent = f"projects/{PROJECT_ID}/locations/{REGION}"
        
        task_statuses = {
            'deploy-endpoint': 'pending',
            'generate-embeddings': 'pending',
            'zero-shot': 'pending',
            'fine-tune': 'pending',
            'evaluate-model': 'pending',
        }
        
        # Check endpoints for deployed MedSigLIP models
        endpoints_response = ai_service.projects().locations().endpoints().list(
            parent=parent
        ).execute()
        
        endpoints = endpoints_response.get('endpoints', [])
        medsiglip_endpoints = [e for e in endpoints if 'medsiglip' in e.get('displayName', '').lower()]
        
        if medsiglip_endpoints:
            has_deployed = any(len(e.get('deployedModels', [])) > 0 for e in medsiglip_endpoints)
            task_statuses['deploy-endpoint'] = 'complete' if has_deployed else 'running'
        
        # Check GCS for pipeline artifacts to infer step completion
        try:
            client = storage.Client(project=PROJECT_ID)
            bucket = client.get_bucket(BUCKET_NAME)
            
            # Embeddings folder → generate-embeddings complete
            embeddings = list(bucket.list_blobs(prefix="embeddings/", max_results=1))
            if embeddings:
                task_statuses['generate-embeddings'] = 'complete'
            
            # Zero-shot results → zero-shot complete
            zs_results = list(bucket.list_blobs(prefix="zero-shot/", max_results=1))
            if zs_results:
                task_statuses['zero-shot'] = 'complete'
            
            # Fine-tune checkpoints → fine-tune complete
            ft_ckpts = list(bucket.list_blobs(prefix="medsiglip-finetune/", max_results=1))
            if ft_ckpts:
                task_statuses['fine-tune'] = 'complete'
            
            # Evaluation results → evaluate-model complete
            eval_results = list(bucket.list_blobs(prefix="results/", max_results=1))
            if eval_results:
                task_statuses['evaluate-model'] = 'complete'
        except Exception:
            pass
        
        pipeline_tasks = list(task_statuses.keys())
        all_complete = all(task_statuses[t] == 'complete' for t in pipeline_tasks)
        
        response_data = {
            'jobs': [],
            'taskStatuses': task_statuses,
            'totalEndpoints': len(medsiglip_endpoints),
            'pipelineComplete': all_complete
        }
        
        print(f"[POLL] MedSigLIP statuses: {task_statuses}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"[POLL ERROR] {str(e)}")
        return jsonify({
            'error': str(e),
            'jobs': [],
            'taskStatuses': {
                'deploy-endpoint': 'pending',
                'generate-embeddings': 'pending',
                'zero-shot': 'pending',
                'fine-tune': 'pending',
                'evaluate-model': 'pending',
            }
        }), 500


@app.route('/api/workbench-status', methods=['GET'])
def get_workbench_status():
    """
    Get the current status and URL of the Vertex AI Workbench instance.
    Uses v2 API with zone-based location.
    Used by frontend to display link to workbench and check if it's ready.
    """
    print(f"\n[WORKBENCH] Checking workbench status (v2 API, zone: {ZONE})...")
    
    try:
        credentials, project = default()
        # Use v2 API for Workbench Instances
        notebooks_service = discovery.build('notebooks', 'v2', credentials=credentials)
        
        # v2 API uses zone for location
        instance_name = f"projects/{PROJECT_ID}/locations/{ZONE}/instances/{WORKBENCH_INSTANCE_NAME}"
        
        try:
            instance = notebooks_service.projects().locations().instances().get(
                name=instance_name
            ).execute()
            
            state = instance.get('state', 'UNKNOWN')
            proxy_uri = instance.get('proxyUri', None)
            
            workbench_url = f"https://console.cloud.google.com/vertex-ai/workbench/instances?project={PROJECT_ID}"
            
            response_data = {
                'exists': True,
                'state': state,
                'instanceName': WORKBENCH_INSTANCE_NAME,
                'workbenchUrl': workbench_url,
                'jupyterUrl': proxy_uri,
                'ready': state == 'ACTIVE'
            }
            
            print(f"[WORKBENCH] Instance state: {state}, ready: {state == 'ACTIVE'}")
            return jsonify(response_data)
            
        except Exception as e:
            if 'notFound' in str(e).lower() or '404' in str(e):
                return jsonify({
                    'exists': False,
                    'state': 'NOT_FOUND',
                    'instanceName': WORKBENCH_INSTANCE_NAME,
                    'ready': False
                })
            raise e
            
    except Exception as e:
        print(f"[WORKBENCH ERROR] {str(e)}")
        return jsonify({
            'error': str(e),
            'exists': False,
            'ready': False
        }), 500


@app.route('/api/poll-bucket-status', methods=['GET'])
def poll_bucket_status():
    """
    Poll for bucket existence and metadata.
    Used by frontend to detect when researcher creates the bucket from notebook.
    Returns bucket info if exists, or not_found status.
    """
    print(f"\n[POLL-BUCKET] Checking bucket: gs://{BUCKET_NAME}")
    
    try:
        client = storage.Client(project=PROJECT_ID)
        
        try:
            bucket = client.get_bucket(BUCKET_NAME)
            
            # Get some metadata about the bucket
            blob_count = 0
            scratch_files = []
            try:
                blobs = list(bucket.list_blobs(prefix="scratch/", max_results=50))
                blob_count = len(blobs)
                scratch_files = [{'name': b.name, 'size': b.size, 'updated': b.updated.isoformat() if b.updated else None} for b in blobs[:10]]
            except Exception:
                pass
            
            response_data = {
                'exists': True,
                'bucketName': BUCKET_NAME,
                'bucketUrl': f'https://console.cloud.google.com/storage/browser/{BUCKET_NAME}?project={PROJECT_ID}',
                'location': bucket.location,
                'storageClass': bucket.storage_class,
                'created': bucket.time_created.isoformat() if bucket.time_created else None,
                'scratchFileCount': blob_count,
                'scratchFiles': scratch_files,
                'status': 'complete'
            }
            
            print(f"[POLL-BUCKET] Bucket exists: {BUCKET_NAME}, scratch files: {blob_count}")
            return jsonify(response_data)
            
        except gcp_exceptions.NotFound:
            print(f"[POLL-BUCKET] Bucket not found: {BUCKET_NAME}")
            return jsonify({
                'exists': False,
                'bucketName': BUCKET_NAME,
                'status': 'pending'
            })
            
    except Exception as e:
        print(f"[POLL-BUCKET ERROR] {str(e)}")
        return jsonify({
            'error': str(e),
            'exists': False,
            'status': 'error'
        }), 500


@app.route('/api/poll-pipeline-logs', methods=['GET'])
def poll_pipeline_logs():
    """
    Poll Cloud Logging for Vertex AI prediction and training logs.
    Returns recent log entries for MedSigLIP pipeline execution.
    Used by frontend to display real-time logs from notebook-triggered pipeline.
    """
    print(f"\n[POLL-LOGS] Polling Cloud Logging for Vertex AI logs...")
    
    try:
        from google.cloud import logging as cloud_logging
        
        client = cloud_logging.Client(project=PROJECT_ID)
        
        # Query for Vertex AI prediction and training logs
        filter_str = '''
            resource.type="aiplatform.googleapis.com/Endpoint" OR
            resource.type="aiplatform.googleapis.com/PipelineJob" OR
            resource.type="ml_job" OR
            (resource.type="gce_instance" AND textPayload:("medsiglip" OR "MedSigLIP" OR "embedding"))
        '''
        
        entries = list(client.list_entries(
            filter_=filter_str,
            order_by=cloud_logging.DESCENDING,
            max_results=50
        ))
        
        logs = []
        for entry in entries:
            log_entry = {
                'timestamp': entry.timestamp.isoformat() if entry.timestamp else None,
                'severity': entry.severity if hasattr(entry, 'severity') else 'INFO',
                'message': str(entry.payload) if entry.payload else '',
                'resource': entry.resource.type if entry.resource else 'unknown',
                'labels': dict(entry.labels) if entry.labels else {}
            }
            logs.append(log_entry)
        
        # Infer pipeline status from log patterns
        pipeline_status = 'unknown'
        for log in logs:
            msg = log.get('message', '').upper()
            if 'COMPLETED' in msg or 'SUCCEEDED' in msg:
                pipeline_status = 'complete'
                break
            elif 'ERROR' in msg or 'FAILED' in msg:
                pipeline_status = 'error'
                break
            elif 'RUNNING' in msg or 'DEPLOYED' in msg or 'PREDICT' in msg:
                pipeline_status = 'running'
        
        response_data = {
            'logs': logs,
            'logCount': len(logs),
            'pipelineStatus': pipeline_status
        }
        
        print(f"[POLL-LOGS] Found {len(logs)} log entries, pipeline status: {pipeline_status}")
        return jsonify(response_data)
        
    except ImportError:
        print(f"[POLL-LOGS] google-cloud-logging not installed")
        return jsonify({
            'logs': [],
            'logCount': 0,
            'pipelineStatus': 'unknown',
            'error': 'Cloud Logging client not available'
        })
    except Exception as e:
        print(f"[POLL-LOGS ERROR] {str(e)}")
        return jsonify({
            'error': str(e),
            'logs': [],
            'logCount': 0,
            'pipelineStatus': 'unknown'
        }), 500


@app.route('/api/poll-all', methods=['GET'])
def poll_all():
    """
    Combined polling endpoint that returns status for all researcher-triggered resources.
    This is called by the frontend in monitoring mode after workbench is provisioned.
    Returns: bucket status, GCS marker-based task statuses, and workbench status.
    
    Infers MedSigLIP pipeline step completion from GCS marker files:
    - status/load-model.json → load-model complete
    - status/download-dataset.json → download-dataset complete
    - status/fine-tune.json → fine-tune complete
    - results/ + medsiglip-finetune/ prefixes → save-results complete
    """
    print(f"\n[POLL-ALL] Combined polling for all MedSigLIP resources...")
    
    result = {
        'bucket': None,
        'jobs': None,
        'workbench': None,
        'pipelineRunning': False,
        'allComplete': False
    }
    
    # 1. Check bucket status
    try:
        client = storage.Client(project=PROJECT_ID)
        try:
            bucket = client.get_bucket(BUCKET_NAME)
            result['bucket'] = {
                'exists': True,
                'status': 'complete',
                'location': bucket.location,
            }
        except gcp_exceptions.NotFound:
            result['bucket'] = {'exists': False, 'status': 'pending'}
    except Exception as e:
        result['bucket'] = {'exists': False, 'status': 'error', 'error': str(e)[:100]}
    
    # 2. Check GCS marker files for pipeline step statuses
    task_statuses = {
        'load-model': 'pending',
        'download-dataset': 'pending',
        'fine-tune': 'pending',
        'save-results': 'pending',
    }
    
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.get_bucket(BUCKET_NAME)
        
        # Check marker files
        marker_map = {
            'status/load-model.json': 'load-model',
            'status/download-dataset.json': 'download-dataset',
            'status/fine-tune.json': 'fine-tune',
        }
        
        for marker_path, task_id in marker_map.items():
            if bucket.blob(marker_path).exists():
                task_statuses[task_id] = 'complete'
                print(f"[POLL-ALL]   ✓ {task_id}: marker found ({marker_path})")
            else:
                print(f"[POLL-ALL]   ○ {task_id}: no marker yet")
        
        # save-results: check for actual artifacts (results/ + medsiglip-finetune/)
        results_blobs = list(bucket.list_blobs(prefix="results/", max_results=1))
        checkpoint_blobs = list(bucket.list_blobs(prefix="medsiglip-finetune/", max_results=1))
        if results_blobs and checkpoint_blobs:
            task_statuses['save-results'] = 'complete'
            print(f"[POLL-ALL]   ✓ save-results: artifacts found in results/ and medsiglip-finetune/")
        else:
            print(f"[POLL-ALL]   ○ save-results: waiting for artifacts (results/: {len(results_blobs)}, finetune/: {len(checkpoint_blobs)})")
    except gcp_exceptions.NotFound:
        print(f"[POLL-ALL] Bucket {BUCKET_NAME} not found, all tasks pending")
    except Exception as e:
        print(f"[POLL-ALL] GCS marker check error: {str(e)}")
    
    pipeline_running = any(s == 'running' for s in task_statuses.values())
    all_complete = all(s == 'complete' for s in task_statuses.values())
    
    result['jobs'] = {
        'taskStatuses': task_statuses,
        'recentJobs': [],
    }
    result['pipelineRunning'] = pipeline_running
    result['allComplete'] = all_complete
    
    # 3. Check workbench status (with timeout)
    try:
        import httplib2 as httplib2_nb
        import google_auth_httplib2 as gah_nb
        credentials, _ = default()
        http_nb = gah_nb.AuthorizedHttp(credentials, http=httplib2_nb.Http(timeout=10))
        notebooks_service = discovery.build('notebooks', 'v2', http=http_nb)
        instance_name = f"projects/{PROJECT_ID}/locations/{ZONE}/instances/{WORKBENCH_INSTANCE_NAME}"
        
        try:
            instance = notebooks_service.projects().locations().instances().get(name=instance_name).execute()
            result['workbench'] = {
                'exists': True,
                'state': instance.get('state', 'UNKNOWN'),
                'ready': instance.get('state') == 'ACTIVE',
                'proxyUri': instance.get('proxyUri')
            }
        except:
            result['workbench'] = {'exists': False, 'ready': False}
    except Exception as e:
        result['workbench'] = {'error': str(e)[:100]}
    
    print(f"[POLL-ALL] Bucket: {result['bucket'].get('status')}, Pipeline running: {result['pipelineRunning']}, All complete: {result['allComplete']}")
    print(f"[POLL-ALL] Task statuses: {task_statuses}")
    return jsonify(result)


if __name__ == '__main__':
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     MedSigLIP Pathology Workload Visualizer - Backend         ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Project ID:  {PROJECT_ID:<45} ║
    ║  Bucket:      gs://{BUCKET_NAME:<42} ║
    ║  Region:      {REGION:<45} ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    Server starting on http://localhost:5000
    Using Python GCP client libraries for all operations.
    """)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
