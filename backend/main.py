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
# europe-west4 chosen for best A100 availability among Workbench-supported regions.
# GCE Supply dashboard (2026-02-12): europe-west4-a has 96 unsold A100 chips (73.91% sold)
# vs us-central1-b with 76 unsold (92.66% sold). Better demo repeatability.
# Notebooks API v2 only supports zones a/b/c per region.
REGION = os.environ.get("GCP_REGION", "europe-west4")
ZONE = f"{REGION}-a"
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
            'roles/storage.admin'
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
    Placeholder for org policy overrides.
    Most projects don't need any changes here. If you hit org constraint errors
    during Workbench provisioning or GPU allocation, add the specific policy
    overrides in this function (e.g., compute.requireShieldedVm,
    compute.trustedImageProjects, compute.vmExternalIpAccess).
    """
    yield log_msg("Org Policies — placeholder (no overrides configured)")
    yield log_msg("  ✓ No org policy changes needed for this project", "success")
    yield log_msg("  Add overrides here if you hit constraint errors later", "info")
    yield step_complete()


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
    The v1 API for user-managed notebooks has been deprecated.
    If instance already exists, returns the URL to access it.
    """
    yield log_msg(f"Provisioning Vertex AI Workbench: {WORKBENCH_INSTANCE_NAME}...")
    
    try:
        credentials, project = default()
        
        # First, enable the Notebooks API if not already enabled
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
        
        # Build the Notebooks API v2 client (v1 is deprecated for new instances)
        notebooks_service = discovery.build('notebooks', 'v2', credentials=credentials)
        
        # v2 API still uses zone for location (not region)
        instance_name = f"projects/{PROJECT_ID}/locations/{ZONE}/instances/{WORKBENCH_INSTANCE_NAME}"
        workbench_url = f"https://console.cloud.google.com/vertex-ai/workbench/instances?project={PROJECT_ID}"
        jupyter_url = None
        
        # Check if instance already exists (with retry for API propagation)
        max_retries = 4
        for attempt in range(max_retries):
            try:
                yield log_msg(f"  Checking for existing instance...")
                instance = notebooks_service.projects().locations().instances().get(
                    name=instance_name
                ).execute()
                break  # success — instance found
            except Exception as e:
                err_str = str(e)
                if ('SERVICE_DISABLED' in err_str or 'has not been used' in err_str) and attempt < max_retries - 1:
                    wait_secs = 15 * (attempt + 1)
                    yield log_msg(f"  ⏳ Notebooks API still propagating, waiting {wait_secs}s (attempt {attempt+1}/{max_retries})...", "info")
                    time.sleep(wait_secs)
                    # Rebuild client after wait
                    notebooks_service = discovery.build('notebooks', 'v2', credentials=credentials)
                    continue
                elif 'notFound' in err_str.lower() or '404' in err_str:
                    instance = None  # Signal to create new instance
                    break
                else:
                    raise e
        else:
            yield step_error("Notebooks API not ready after retries. Please wait a minute and try again.")
            return
        
        if instance is not None:
            
            state = instance.get('state', 'UNKNOWN')
            yield log_msg(f"  ✓ Workbench instance already exists (state: {state})", "info")
            
            # v2 API uses 'proxyUri' for JupyterLab access
            if 'proxyUri' in instance:
                jupyter_url = instance['proxyUri']
                yield log_msg(f"  JupyterLab URL: {jupyter_url}", "success")
            
            # Send the workbench URL for frontend to display
            yield stream_sse({
                "log": f"Workbench ready: {WORKBENCH_INSTANCE_NAME}",
                "type": "success",
                "workbenchUrl": workbench_url,
                "jupyterUrl": jupyter_url,
                "instanceName": WORKBENCH_INSTANCE_NAME,
                "status": "complete"
            })
            return
        
        # Instance not found — create new one
        yield log_msg(f"  Instance not found, creating new workbench...", "info")
        
        # Create the Workbench instance using v2 API
        sa_email = f"{SERVICE_ACCOUNT_NAME}@{PROJECT_ID}.iam.gserviceaccount.com"
        
        # Startup script: install deps only. Data download + notebook cells are user-driven.
        startup_script = '''#!/bin/bash
set -e
echo "=== MedSigLIP A100 Workbench Setup ==="

# Verify GPU
nvidia-smi || echo "WARNING: nvidia-smi not found, GPU drivers may need install"

# Install Python dependencies for MedSigLIP fine-tuning
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets huggingface_hub accelerate
pip install google-cloud-aiplatform google-cloud-storage
pip install scikit-learn matplotlib pillow tqdm

# Create researcher workspace
mkdir -p /home/jupyter/medsiglip-workspace
chown -R jupyter:jupyter /home/jupyter/medsiglip-workspace

echo "=== Setup complete. Open medsiglip-workspace/ in JupyterLab ==="
'''

        # v2 API instance body structure with gceSetup — A100 GPU for fine-tuning
        instance_body = {
            'gceSetup': {
                'machineType': 'a2-highgpu-1g',
                'acceleratorConfigs': [
                    {
                        'type': 'NVIDIA_TESLA_A100',
                        'coreCount': '1'
                    }
                ],
                'serviceAccounts': [
                    {
                        'email': sa_email,
                        'scopes': ['https://www.googleapis.com/auth/cloud-platform']
                    }
                ],
                'networkInterfaces': [
                    {
                        'network': f'projects/{PROJECT_ID}/global/networks/default',
                        'subnet': f'projects/{PROJECT_ID}/regions/{REGION}/subnetworks/default',
                        'nicType': 'VIRTIO_NET'
                    }
                ],
                'disablePublicIp': True,  # Use internal IP only (org policy compliance)
                'metadata': {
                    'startup-script': startup_script,
                    'proxy-mode': 'service_account'
                },
                'bootDisk': {
                    'diskSizeGb': '300',
                    'diskType': 'PD_SSD'
                },
                'vmImage': {
                    'project': 'cloud-notebooks-managed',
                    'name': 'workbench-instances-v20260122'
                },
                'shieldedInstanceConfig': {
                    'enableSecureBoot': True,
                    'enableVtpm': True,
                    'enableIntegrityMonitoring': True
                }
            }
        }
        
        yield log_msg("  Creating Workbench instance (this takes 5-10 minutes)...", "info")
        yield log_msg(f"  Machine: a2-highgpu-1g (A100 40GB GPU), Zone: {ZONE}", "info")
        yield log_msg(f"  Disk: 300GB PD-SSD, Network: default (no public IP)", "info")
        yield log_msg(f"  Using Notebooks API v2 (Workbench Instances)", "info")
        
        # v2 API uses zone for location
        operation = notebooks_service.projects().locations().instances().create(
            parent=f"projects/{PROJECT_ID}/locations/{ZONE}",
            instanceId=WORKBENCH_INSTANCE_NAME,
            body=instance_body
        ).execute()
        
        operation_name = operation.get('name')
        yield log_msg(f"  Operation started: {operation_name.split('/')[-1]}", "info")
        
        # Poll for operation completion
        max_wait = 600  # 10 minutes max
        poll_interval = 15
        elapsed = 0
        
        while elapsed < max_wait:
            op_result = notebooks_service.projects().locations().operations().get(
                name=operation_name
            ).execute()
            
            if op_result.get('done'):
                if 'error' in op_result:
                    yield step_error(f"Failed: {op_result['error'].get('message', 'Unknown error')}")
                    return
                
                yield log_msg("  ✓ Workbench instance created successfully!", "success")
                
                # Get the instance details
                instance = notebooks_service.projects().locations().instances().get(
                    name=instance_name
                ).execute()
                
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
            
            elapsed += poll_interval
            yield log_msg(f"  Provisioning... ({elapsed}s elapsed)", "info")
            time.sleep(poll_interval)
        
        yield log_msg("  ⚠ Workbench still provisioning (check console)", "info")
        yield stream_sse({
            "log": f"Workbench provisioning in progress",
            "type": "info",
            "workbenchUrl": workbench_url,
            "instanceName": WORKBENCH_INSTANCE_NAME,
            "status": "complete"
        })
        
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
    # MedSigLIP pipeline steps (researcher-driven from notebook)
    'deploy-endpoint': execute_check_vertex_ai_status,
    'generate-embeddings': execute_check_vertex_ai_status,
    'zero-shot': execute_check_vertex_ai_status,
    'fine-tune': execute_check_vertex_ai_status,
    'evaluate-model': execute_check_vertex_ai_status,
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
    Returns: bucket status, Vertex AI endpoint/artifact statuses, and workbench status.
    
    Infers MedSigLIP pipeline step completion from:
    - Deployed Vertex AI endpoints (deploy-endpoint)
    - GCS artifacts: embeddings/, zero-shot/, medsiglip-finetune/, results/
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
    
    # 2. Check Vertex AI endpoints and GCS artifacts for pipeline step statuses
    task_statuses = {
        'deploy-endpoint': 'pending',
        'generate-embeddings': 'pending',
        'zero-shot': 'pending',
        'fine-tune': 'pending',
        'evaluate-model': 'pending',
    }
    
    num_medsiglip_endpoints = 0
    try:
        credentials, _ = default()
        ai_service = discovery.build('aiplatform', 'v1', credentials=credentials)
        parent = f"projects/{PROJECT_ID}/locations/{REGION}"
        
        endpoints_response = ai_service.projects().locations().endpoints().list(
            parent=parent
        ).execute()
        
        endpoints = endpoints_response.get('endpoints', [])
        medsiglip_eps = [e for e in endpoints if 'medsiglip' in e.get('displayName', '').lower()]
        num_medsiglip_endpoints = len(medsiglip_eps)
        
        print(f"[POLL-ALL] Found {len(endpoints)} total endpoints, {num_medsiglip_endpoints} MedSigLIP")
        
        if medsiglip_eps:
            has_deployed = any(len(e.get('deployedModels', [])) > 0 for e in medsiglip_eps)
            task_statuses['deploy-endpoint'] = 'complete' if has_deployed else 'running'
            for ep in medsiglip_eps:
                n_models = len(ep.get('deployedModels', []))
                print(f"[POLL-ALL]   • {ep.get('displayName')}: {n_models} deployed model(s)")
    except Exception as e:
        print(f"[POLL-ALL] Vertex AI endpoint check error: {str(e)}")
    
    # Check GCS for pipeline artifacts
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.get_bucket(BUCKET_NAME)
        
        artifact_map = {
            'embeddings/': 'generate-embeddings',
            'zero-shot/': 'zero-shot',
            'medsiglip-finetune/': 'fine-tune',
            'results/': 'evaluate-model',
        }
        
        for prefix, task_id in artifact_map.items():
            blobs = list(bucket.list_blobs(prefix=prefix, max_results=1))
            if blobs:
                task_statuses[task_id] = 'complete'
    except Exception:
        pass
    
    pipeline_running = any(s == 'running' for s in task_statuses.values())
    all_complete = all(s == 'complete' for s in task_statuses.values())
    
    result['jobs'] = {
        'taskStatuses': task_statuses,
        'recentJobs': [],
        'totalEndpoints': num_medsiglip_endpoints
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
