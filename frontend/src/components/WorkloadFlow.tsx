import React, { useState, useCallback, useEffect, useRef } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  ConnectionLineType,
  useReactFlow,
  ReactFlowProvider,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { SetupStepNode } from './SetupStepNode';
import { PipelineTaskNode } from './PipelineTaskNode';
import { GroupNode } from './GroupNode';
import { ExecutionBox } from './ExecutionBox';
import { CostInfoTooltip } from './CostInfoTooltip';
import './WorkloadFlow.css';

const nodeTypes = {
  setupStep: SetupStepNode,
  pipelineTask: PipelineTaskNode,
  groupNode: GroupNode,
};

const GCP_DIFFERENTIATORS: Record<string, string> = {
  'deploy-endpoint': `🚀 Model Garden One-Click Deploy:
• Deploy MedSigLIP (medsiglip-448) directly from Vertex AI Model Garden
• Pre-configured serving container with 448×448 medical image support
• Dedicated endpoint with auto-scaling and per-second billing
• T4 GPU inference at ~$0.35/hr (Spot: ~$0.11/hr)
• Other clouds: manually build container, configure GPU drivers, set up LB`,
  'generate-embeddings': `🧬 768-dim Contrastive Embeddings:
• Supports GCS URIs, DICOMweb URIs, or base64 image bytes as input
• Native Healthcare API integration — feed DICOM URIs directly
• Online prediction for real-time, batch prediction for large-scale
• Batch: submit JSONL with 100K images, auto-provisioned T4 GPUs
• Per-second billing with no minimum runtime`,
  'zero-shot': `🎯 Zero-Shot Classification:
• Classify pathology images with text prompts — no training data needed
• Measures cosine distance between image and text embeddings
• 9 tissue classes from NCT-CRC-HE histopathology dataset
• Outperforms data-efficient methods at low label counts
• No additional compute or model training required`,
  'fine-tune': `🔧 Contrastive Fine-Tuning:
• Joint vision + text encoder training with HF Trainer
• NCT-CRC-HE-100K dataset (100K histopathology patches)
• ~3 hours on A100 GPU (40GB+ VRAM required)
• Spot VMs reduce A100 cost from $2.93/hr to $0.88/hr (70% savings)
• Checkpoints saved to GCS, push to HuggingFace Hub`,
  'evaluate-model': `📊 Model Evaluation:
• CRC-VAL-HE-7K test set (no overlap with training data)
• Accuracy + weighted F1 metrics
• Compare pretrained vs fine-tuned zero-shot performance
• Integrated with Vertex AI Experiments for tracking runs
• Reproducible evaluation with fixed seeds`,
  'provision-workbench': `🔬 Vertex AI Workbench:
• Fully managed JupyterLab with GPU support (T4/A100)
• Pre-installed: transformers, torch, google-cloud-aiplatform
• Secure access via IAP — no public IP required
• Integrated with GCS for seamless data access
• Cost: ~$0.19/hr for n1-standard-4 (+ GPU add-on)`,
  'storage-bucket': `📦 Cloud Storage for Medical AI:
• Store DICOM images, embeddings, and fine-tuned checkpoints
• DICOMweb URIs flow directly into MedSigLIP predict API
• Fastest time-to-first-byte among all cloud storage providers
• Data encrypted by default with CMEK available

🌐 Dual-Region Storage:
• Single bucket namespace spanning two regions
• Strong consistency with RTO of zero
• Other clouds require separate buckets with eventual consistency`,
};

// Infrastructure setup steps (platform team) - vertical column on left
const INFRA_STEPS = [
  { id: 'enable-apis', label: 'Enable APIs', command: 'Vertex AI, Compute, Healthcare, IAM', icon: 'api' },
  { id: 'create-sa', label: 'Create Service Account', command: 'medsiglip-pipeline-sa', icon: 'person' },
  { id: 'iam-roles', label: 'Add IAM Roles', command: '5 roles granted', icon: 'security' },
  { id: 'org-policies', label: 'Configure Org Policies', command: 'VM + GPU image exceptions', icon: 'policy' },
  { id: 'create-network', label: 'Create VPC Network', command: 'default + firewall + PGA', icon: 'lan' },
  { id: 'configure-cloud-nat', label: 'Configure Cloud NAT', command: 'Router + NAT for outbound', icon: 'router' },
];

interface StepStatus {
  status: 'pending' | 'running' | 'complete' | 'error';
  logs: Array<{ timestamp: string; message: string; type: 'info' | 'success' | 'error' }>;
}

interface ProjectConfig {
  projectId: string;
  bucketName: string;
  region: string;
  zone: string;
  serviceAccountName: string;
  workbenchInstanceName: string;
  consoleBaseUrl: string;
}

const DEFAULT_CONFIG: ProjectConfig = {
  projectId: '',
  bucketName: '',
  region: 'europe-west4',
  zone: 'europe-west4-a',
  serviceAccountName: 'medsiglip-pipeline-sa',
  workbenchInstanceName: 'medsiglip-researcher-workbench',
  consoleBaseUrl: 'https://console.cloud.google.com',
};

interface WorkloadFlowInnerProps {
  onComplete?: () => void;
}

const WorkloadFlowInner: React.FC<WorkloadFlowInnerProps> = ({ onComplete }) => {
  const [stepStatuses, setStepStatuses] = useState<Record<string, StepStatus>>({});
  const [isRunning, setIsRunning] = useState(false);
  const [selectedStep, setSelectedStep] = useState<string | null>(null);
  const [currentPhase, setCurrentPhase] = useState<'setup' | 'pipeline'>('setup');
  const [workbenchUrl, setWorkbenchUrl] = useState<string | null>(null);
  const [isMonitoringMode, setIsMonitoringMode] = useState(false);
  const [jupyterUrl, setJupyterUrl] = useState<string | null>(null);
  const [config, setConfig] = useState<ProjectConfig>(DEFAULT_CONFIG);
  const abortControllerRef = useRef<AbortController | null>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const { setViewport } = useReactFlow();

  // Fetch project config from backend on mount
  useEffect(() => {
    fetch('/api/config')
      .then(res => res.json())
      .then(data => {
        console.log('[CONFIG] Loaded project config:', JSON.stringify(data, null, 2));
        setConfig(data);
      })
      .catch(err => console.error('[CONFIG] Failed to load config:', err));
  }, []);

  // Layout constants
  const INFRA_START_X = 50;
  const INFRA_START_Y = 30;
  const INFRA_Y_GAP = 90;
  
  // IAM Roles Y position (index 2) - Launch Pipeline aligns with this
  const IAM_ROLES_Y = INFRA_START_Y + 2 * INFRA_Y_GAP;  // = 210
  
  // Workbench: below VPC Network, left-aligned with infra column
  const WORKBENCH_X = INFRA_START_X;
  const WORKBENCH_Y = INFRA_START_Y + INFRA_STEPS.length * INFRA_Y_GAP + 80;
  
  // Storage Bucket: directly to the right of workbench, same Y level
  const BUCKET_X = WORKBENCH_X + 450;  // Gap for 400px nodes
  const BUCKET_Y = WORKBENCH_Y;
  
  // Launch Pipeline: aligned vertically with IAM Roles, same X as bucket
  const LAUNCH_X = BUCKET_X;
  const LAUNCH_Y = IAM_ROLES_Y;  // Center vertically with IAM Roles
  
  // Pipeline tasks flow to the right with spacing for 400px nodes
  const HORIZONTAL_GAP = 450;  // Gap for 400px nodes
  const PARALLEL_Y_GAP = 140;  // Vertical spacing between parallel tasks

  const zoomToSetup = useCallback(() => {
    setViewport({ x: 100, y: 0, zoom: 1.2 }, { duration: 800 });
  }, [setViewport]);

  const zoomToPipeline = useCallback(() => {
    setViewport({ x: -100, y: 50, zoom: 0.7 }, { duration: 800 });
  }, [setViewport]);

  const generateNodes = useCallback((): Node[] => {
    const nodes: Node[] = [];
    
    // Calculate group box dimensions
    const secondParallelX = LAUNCH_X + HORIZONTAL_GAP * 2;
    const nodeHeight = 80;
    const padding = 30;
    const leftPadding = 80;  // Extra left padding for edge clearance and labels
    
    // IT Group Box: contains infra steps + launch pipeline + pipeline tasks
    const itBoxX = INFRA_START_X - leftPadding;
    const itBoxY = INFRA_START_Y - padding - 20;  // Extra top for label
    const itBoxWidth = secondParallelX + 400 + padding - itBoxX + 40;  // extends to rightmost task + padding
    const itBoxHeight = WORKBENCH_Y - itBoxY - 50;  // ends clearly above workbench row
    
    nodes.push({
      id: 'group-it',
      type: 'groupNode',
      position: { x: itBoxX, y: itBoxY },
      zIndex: -1,
      selectable: false,
      draggable: false,
      data: {
        label: 'IT',
        icon: 'admin_panel_settings',
        width: itBoxWidth,
        height: itBoxHeight,
        groupType: 'it',
      },
    });
    
    // Researcher Group Box: contains workbench + storage bucket
    const researcherBoxX = WORKBENCH_X - leftPadding;
    const researcherBoxY = WORKBENCH_Y - padding - 20;  // Extra top for label
    const researcherBoxWidth = BUCKET_X + 400 + padding - researcherBoxX + 40;
    const researcherBoxHeight = nodeHeight + padding * 2 + 20;
    
    nodes.push({
      id: 'group-researcher',
      type: 'groupNode',
      position: { x: researcherBoxX, y: researcherBoxY },
      zIndex: -1,
      selectable: false,
      draggable: false,
      data: {
        label: 'Researcher',
        icon: 'science',
        width: researcherBoxWidth,
        height: researcherBoxHeight,
        groupType: 'researcher',
      },
    });

    // Infrastructure setup nodes (vertical stack on left)
    INFRA_STEPS.forEach((step, index) => {
      nodes.push({
        id: step.id,
        type: 'setupStep',
        position: { x: INFRA_START_X, y: INFRA_START_Y + index * INFRA_Y_GAP },
        data: {
          label: step.label,
          command: step.command,
          icon: step.icon,
          status: stepStatuses[step.id]?.status || 'pending',
          isSelected: selectedStep === step.id,
          onClick: () => setSelectedStep(step.id),
        },
      });
    });

    // Provision Workbench: below VPC, left side of researcher box
    const defaultWorkbenchUrl = `${config.consoleBaseUrl}/vertex-ai/workbench/instances?project=${config.projectId}`;
    nodes.push({
      id: 'provision-workbench',
      type: 'pipelineTask',
      position: { x: WORKBENCH_X, y: WORKBENCH_Y },
      data: {
        label: 'Provision Workbench',
        command: 'Vertex AI Workbench (GPU)',
        icon: 'terminal',
        status: stepStatuses['provision-workbench']?.status || 'pending',
        isSelected: selectedStep === 'provision-workbench',
        onClick: () => setSelectedStep('provision-workbench'),
        tooltip: GCP_DIFFERENTIATORS['provision-workbench'],
        batchJobUrl: workbenchUrl || defaultWorkbenchUrl,
      },
    });

    // Storage Bucket: DICOM images, embeddings, checkpoints
    nodes.push({
      id: 'storage-bucket',
      type: 'pipelineTask',
      position: { x: BUCKET_X, y: BUCKET_Y },
      data: {
        label: 'Storage Bucket',
        command: `gs://${config.bucketName}`,
        icon: 'cloud_upload',
        status: stepStatuses['storage-bucket']?.status || 'pending',
        isSelected: selectedStep === 'storage-bucket',
        onClick: () => setSelectedStep('storage-bucket'),
        tooltip: GCP_DIFFERENTIATORS['storage-bucket'],
        batchJobUrl: `${config.consoleBaseUrl}/storage/browser/${config.bucketName}?project=${config.projectId}`,
      },
    });

    // Deploy MedSigLIP Endpoint: aligned with IAM Roles row, same X as bucket
    nodes.push({
      id: 'deploy-endpoint',
      type: 'pipelineTask',
      position: { x: LAUNCH_X, y: LAUNCH_Y },
      data: {
        label: 'Deploy MedSigLIP',
        command: 'medsiglip-448 → Vertex AI Endpoint',
        icon: 'rocket_launch',
        status: stepStatuses['deploy-endpoint']?.status || 'pending',
        isSelected: selectedStep === 'deploy-endpoint',
        onClick: () => setSelectedStep('deploy-endpoint'),
        tooltip: GCP_DIFFERENTIATORS['deploy-endpoint'],
        batchJobUrl: `${config.consoleBaseUrl}/vertex-ai/online-prediction/endpoints?project=${config.projectId}`,
      },
    });

    // Generate Embeddings: first step after deploy
    const firstParallelX = LAUNCH_X + HORIZONTAL_GAP;
    
    nodes.push({
      id: 'generate-embeddings',
      type: 'pipelineTask',
      position: { x: firstParallelX, y: LAUNCH_Y - PARALLEL_Y_GAP / 2 - 30 },
      data: {
        label: 'Generate Embeddings',
        command: '768-dim image + text vectors',
        icon: 'data_array',
        status: stepStatuses['generate-embeddings']?.status || 'pending',
        isSelected: selectedStep === 'generate-embeddings',
        onClick: () => setSelectedStep('generate-embeddings'),
        tooltip: GCP_DIFFERENTIATORS['generate-embeddings'],
        batchJobUrl: `${config.consoleBaseUrl}/vertex-ai/online-prediction/endpoints?project=${config.projectId}`,
      },
    });

    // Zero-Shot Classification: parallel with embeddings
    nodes.push({
      id: 'zero-shot',
      type: 'pipelineTask',
      position: { x: firstParallelX, y: LAUNCH_Y + PARALLEL_Y_GAP / 2 + 30 },
      data: {
        label: 'Zero-Shot Classification',
        command: 'Text-prompt tissue classification',
        icon: 'category',
        status: stepStatuses['zero-shot']?.status || 'pending',
        isSelected: selectedStep === 'zero-shot',
        onClick: () => setSelectedStep('zero-shot'),
        tooltip: GCP_DIFFERENTIATORS['zero-shot'],
      },
    });

    // Fine-Tune Model: after embeddings
    nodes.push({
      id: 'fine-tune',
      type: 'pipelineTask',
      position: { x: secondParallelX, y: LAUNCH_Y - PARALLEL_Y_GAP / 2 - 30 },
      data: {
        label: 'Fine-Tune Model',
        command: 'HF Trainer on NCT-CRC-HE-100K',
        icon: 'model_training',
        status: stepStatuses['fine-tune']?.status || 'pending',
        isSelected: selectedStep === 'fine-tune',
        onClick: () => setSelectedStep('fine-tune'),
        tooltip: GCP_DIFFERENTIATORS['fine-tune'],
      },
    });

    // Evaluate Model: after fine-tune and zero-shot
    nodes.push({
      id: 'evaluate-model',
      type: 'pipelineTask',
      position: { x: secondParallelX, y: LAUNCH_Y + PARALLEL_Y_GAP / 2 + 30 },
      data: {
        label: 'Evaluate Model',
        command: 'Accuracy + F1 on CRC-VAL-HE-7K',
        icon: 'assessment',
        status: stepStatuses['evaluate-model']?.status || 'pending',
        isSelected: selectedStep === 'evaluate-model',
        onClick: () => setSelectedStep('evaluate-model'),
        tooltip: GCP_DIFFERENTIATORS['evaluate-model'],
      },
    });

    return nodes;
  }, [stepStatuses, selectedStep, workbenchUrl, WORKBENCH_Y, BUCKET_Y, LAUNCH_Y]);

  const generateEdges = useCallback((): Edge[] => {
    const edges: Edge[] = [];

    // Infrastructure edges (vertical flow)
    INFRA_STEPS.slice(0, -1).forEach((step, index) => {
      edges.push({
        id: `e-${step.id}-${INFRA_STEPS[index + 1].id}`,
        source: step.id, target: INFRA_STEPS[index + 1].id,
        sourceHandle: 'source-bottom', targetHandle: 'target-top',
        type: 'straight',
        animated: stepStatuses[step.id]?.status === 'complete',
        style: { stroke: stepStatuses[step.id]?.status === 'complete' ? '#4CAF50' : '#DADCE0', strokeWidth: 2 },
      });
    });

    // Cloud NAT (bottom) → Provision Workbench (top) - smooth bezier curve
    edges.push({
      id: 'e-nat-workbench',
      source: 'configure-cloud-nat', target: 'provision-workbench',
      sourceHandle: 'source-bottom', targetHandle: 'target-top',
      type: 'default',
      animated: stepStatuses['configure-cloud-nat']?.status === 'complete',
      style: { stroke: stepStatuses['configure-cloud-nat']?.status === 'complete' ? '#4CAF50' : '#DADCE0', strokeWidth: 2 },
    });

    // Workbench (right) → Storage Bucket (left)
    edges.push({
      id: 'e-workbench-bucket',
      source: 'provision-workbench', target: 'storage-bucket',
      sourceHandle: 'source-right', targetHandle: 'target-left',
      type: 'smoothstep',
      animated: stepStatuses['provision-workbench']?.status === 'complete',
      style: { stroke: stepStatuses['provision-workbench']?.status === 'complete' ? '#4CAF50' : '#DADCE0', strokeWidth: 2 },
    });

    // Storage Bucket (top) → Deploy Endpoint (bottom) - vertical line
    edges.push({
      id: 'e-bucket-deploy',
      source: 'storage-bucket', target: 'deploy-endpoint',
      sourceHandle: 'source-top', targetHandle: 'target-bottom',
      type: 'straight',
      animated: stepStatuses['storage-bucket']?.status === 'complete',
      style: { stroke: stepStatuses['storage-bucket']?.status === 'complete' ? '#4CAF50' : '#DADCE0', strokeWidth: 2 },
    });

    // Deploy Endpoint → Generate Embeddings and Zero-Shot (parallel)
    ['generate-embeddings', 'zero-shot'].forEach(taskId => {
      const isRunning = stepStatuses[taskId]?.status === 'running';
      const isComplete = stepStatuses[taskId]?.status === 'complete';
      edges.push({
        id: `e-deploy-${taskId}`,
        source: 'deploy-endpoint', target: taskId,
        sourceHandle: 'source-right', targetHandle: 'target-left',
        type: 'smoothstep',
        animated: isRunning || isComplete,
        style: { stroke: isComplete ? '#4CAF50' : isRunning ? '#1A73E8' : '#DADCE0', strokeWidth: 2 },
      });
    });

    // Generate Embeddings → Fine-Tune
    edges.push({
      id: 'e-embeddings-finetune',
      source: 'generate-embeddings', target: 'fine-tune',
      sourceHandle: 'source-right', targetHandle: 'target-left',
      type: 'smoothstep',
      animated: stepStatuses['generate-embeddings']?.status === 'complete' || stepStatuses['fine-tune']?.status === 'running',
      style: { stroke: stepStatuses['fine-tune']?.status === 'complete' ? '#4CAF50' : stepStatuses['generate-embeddings']?.status === 'complete' ? '#1A73E8' : '#DADCE0', strokeWidth: 2 },
    });

    // Zero-Shot → Evaluate Model
    edges.push({
      id: 'e-zeroshot-evaluate',
      source: 'zero-shot', target: 'evaluate-model',
      sourceHandle: 'source-right', targetHandle: 'target-left',
      type: 'smoothstep',
      animated: stepStatuses['zero-shot']?.status === 'complete' || stepStatuses['evaluate-model']?.status === 'running',
      style: { stroke: stepStatuses['evaluate-model']?.status === 'complete' ? '#4CAF50' : stepStatuses['zero-shot']?.status === 'complete' ? '#1A73E8' : '#DADCE0', strokeWidth: 2 },
    });

    // Fine-Tune → Evaluate Model
    edges.push({
      id: 'e-finetune-evaluate',
      source: 'fine-tune', target: 'evaluate-model',
      sourceHandle: 'source-bottom', targetHandle: 'target-top',
      type: 'smoothstep',
      animated: stepStatuses['fine-tune']?.status === 'complete' || stepStatuses['evaluate-model']?.status === 'running',
      style: { stroke: stepStatuses['evaluate-model']?.status === 'complete' ? '#4CAF50' : stepStatuses['fine-tune']?.status === 'complete' ? '#1A73E8' : '#DADCE0', strokeWidth: 2 },
    });

    // Fine-Tune → Bucket (loop back - checkpoints saved to GCS)
    const fineTuneComplete = stepStatuses['fine-tune']?.status === 'complete';
    edges.push({
      id: 'e-finetune-bucket',
      source: 'fine-tune', target: 'storage-bucket',
      sourceHandle: 'source-right', targetHandle: 'target-right',
      type: 'smoothstep',
      animated: fineTuneComplete,
      style: { 
        stroke: fineTuneComplete ? '#9C27B0' : '#DADCE0', 
        strokeWidth: 2,
        strokeDasharray: fineTuneComplete ? '0' : '5,5',
      },
    });

    // Evaluate → Bucket (loop back - results saved to GCS)
    const evaluateComplete = stepStatuses['evaluate-model']?.status === 'complete';
    edges.push({
      id: 'e-evaluate-bucket',
      source: 'evaluate-model', target: 'storage-bucket',
      sourceHandle: 'source-right', targetHandle: 'target-right',
      type: 'smoothstep',
      animated: evaluateComplete,
      style: { 
        stroke: evaluateComplete ? '#9C27B0' : '#DADCE0', 
        strokeWidth: 2,
        strokeDasharray: evaluateComplete ? '0' : '5,5',
      },
      label: evaluateComplete ? '📊 Results in Bucket' : '',
      labelStyle: { fill: '#9C27B0', fontWeight: 600, fontSize: 11 },
      labelBgStyle: { fill: 'white', fillOpacity: 0.9 },
    });

    return edges;
  }, [stepStatuses]);

  const [nodes, setNodes, onNodesChange] = useNodesState(generateNodes());
  const [edges, setEdges, onEdgesChange] = useEdgesState(generateEdges());

  useEffect(() => {
    setNodes(generateNodes());
    setEdges(generateEdges());
  }, [stepStatuses, selectedStep, workbenchUrl, generateNodes, generateEdges, setNodes, setEdges]);

  const addLog = (stepId: string, message: string, type: 'info' | 'success' | 'error' = 'info') => {
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
    setStepStatuses(prev => ({
      ...prev,
      [stepId]: { ...prev[stepId], logs: [...(prev[stepId]?.logs || []), { timestamp, message, type }] },
    }));
  };

  const runStep = async (stepId: string, stepLabel: string, signal: AbortSignal) => {
    setStepStatuses(prev => ({ ...prev, [stepId]: { status: 'running', logs: prev[stepId]?.logs || [] } }));
    setSelectedStep(stepId);
    addLog(stepId, `Starting: ${stepLabel}`, 'info');

    try {
      const response = await fetch('/api/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stepId, phase: currentPhase }),
        signal,
      });

      if (!response.ok) throw new Error(`HTTP error: ${response.status}`);

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) throw new Error('No response body');

      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.substring(6));
              
              if (data.workbenchUrl) {
                setWorkbenchUrl(data.workbenchUrl);
              }
              
              if (data.type === 'task_update' && data.task) {
                const taskStatus = data.status as 'running' | 'complete' | 'error';
                setStepStatuses(prev => ({
                  ...prev,
                  [data.task]: { 
                    status: taskStatus, 
                    logs: [...(prev[data.task]?.logs || []), {
                      timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
                      message: data.message || `${data.task.toUpperCase()}: ${taskStatus}`,
                      type: taskStatus === 'error' ? 'error' : taskStatus === 'complete' ? 'success' : 'info'
                    }]
                  }
                }));
              }
              
              if (data.log) addLog(stepId, data.log, data.type || 'info');
              
              if (data.status === 'complete') {
                setStepStatuses(prev => ({ ...prev, [stepId]: { ...prev[stepId], status: 'complete' } }));
              } else if (data.status === 'error') {
                throw new Error(data.message);
              }
            } catch (e) {}
          }
        }
      }
      return true;
    } catch (error: any) {
      if (error.name === 'AbortError') { addLog(stepId, 'Aborted', 'error'); return false; }
      addLog(stepId, `Error: ${error.message}`, 'error');
      setStepStatuses(prev => ({ ...prev, [stepId]: { ...prev[stepId], status: 'error' } }));
      return false;
    }
  };

  // Polling function to check researcher-triggered resources
  const pollResources = useCallback(async () => {
    console.log('[POLL] Polling for researcher resources...');
    try {
      const response = await fetch('/api/poll-all');
      if (!response.ok) return;
      
      const data = await response.json();
      console.log('[POLL] Response:', JSON.stringify(data, null, 2));
      
      // Update bucket status
      if (data.bucket) {
        const bucketStatus = data.bucket.exists ? 'complete' : 'pending';
        setStepStatuses(prev => {
          // Only update if status changed
          if (prev['storage-bucket']?.status !== bucketStatus) {
            const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
            const logs = prev['storage-bucket']?.logs || [];
            const newLogs = bucketStatus === 'complete' 
              ? [...logs, { timestamp, message: `✓ Bucket detected: gs://${config.bucketName} (${data.bucket.location})`, type: 'success' as const }]
              : logs;
            return {
              ...prev,
              'storage-bucket': { status: bucketStatus, logs: newLogs }
            };
          }
          return prev;
        });
      }
      
      // Update pipeline task statuses from Vertex AI jobs
      if (data.jobs?.taskStatuses) {
        const taskStatuses = data.jobs.taskStatuses;
        const pipelineTasks = ['deploy-endpoint', 'generate-embeddings', 'zero-shot', 'fine-tune', 'evaluate-model'];
        
        setStepStatuses(prev => {
          const updates: Record<string, StepStatus> = {};
          let hasUpdates = false;
          
          for (const taskId of pipelineTasks) {
            const newStatus = taskStatuses[taskId] as 'pending' | 'running' | 'complete';
            if (newStatus && prev[taskId]?.status !== newStatus) {
              hasUpdates = true;
              const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
              const logs = prev[taskId]?.logs || [];
              const statusMessage = newStatus === 'complete' 
                ? `✓ ${taskId} completed successfully`
                : newStatus === 'running'
                ? `▶ ${taskId} running on Vertex AI...`
                : '';
              
              updates[taskId] = {
                status: newStatus,
                logs: statusMessage ? [...logs, { timestamp, message: statusMessage, type: newStatus === 'complete' ? 'success' as const : 'info' as const }] : logs
              };
            }
          }
          
          // Also update deploy-endpoint if any task is running
          if (data.pipelineRunning && prev['deploy-endpoint']?.status !== 'running') {
            hasUpdates = true;
            const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
            updates['deploy-endpoint'] = {
              status: 'running',
              logs: [...(prev['deploy-endpoint']?.logs || []), { timestamp, message: '▶ MedSigLIP pipeline detected running on Vertex AI', type: 'info' as const }]
            };
          } else if (data.allComplete && prev['deploy-endpoint']?.status !== 'complete') {
            hasUpdates = true;
            const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
            updates['deploy-endpoint'] = {
              status: 'complete',
              logs: [...(prev['deploy-endpoint']?.logs || []), { timestamp, message: '✓ MedSigLIP pipeline completed successfully!', type: 'success' as const }]
            };
          }
          
          return hasUpdates ? { ...prev, ...updates } : prev;
        });
      }
      
      // If all tasks complete, stop polling
      if (data.allComplete) {
        console.log('[POLL] All tasks complete, stopping polling');
        stopPolling();
        if (onComplete) onComplete();
      }
      
    } catch (error) {
      console.error('[POLL] Error polling resources:', error);
    }
  }, [onComplete]);

  // Start polling for researcher resources
  const startPolling = useCallback(() => {
    console.log('[MONITORING] Starting polling mode...');
    setIsMonitoringMode(true);
    
    // Initial poll immediately
    pollResources();
    
    // Then poll every 5 seconds
    pollingIntervalRef.current = setInterval(pollResources, 5000);
  }, [pollResources]);

  // Stop polling
  const stopPolling = useCallback(() => {
    console.log('[MONITORING] Stopping polling mode');
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
    setIsMonitoringMode(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  const runAllSteps = async () => {
    if (abortControllerRef.current) abortControllerRef.current.abort();
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    setIsRunning(true);
    stopPolling(); // Ensure no old polling is running

    zoomToSetup();

    // Run infrastructure setup steps
    setCurrentPhase('setup');
    for (const step of INFRA_STEPS) {
      if (abortController.signal.aborted) break;
      const success = await runStep(step.id, step.label, abortController.signal);
      if (!success) break;
    }

    // Provision workbench
    if (!abortController.signal.aborted) {
      const success = await runStep('provision-workbench', 'Provision Workbench', abortController.signal);
      if (!success) { 
        setIsRunning(false); 
        return; 
      }
    }

    // After workbench is provisioned, STOP and enter monitoring mode
    // The researcher will run the remaining steps from the notebook cells
    setIsRunning(false);
    
    // Add log indicating we're now in monitoring mode
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
    setStepStatuses(prev => ({
      ...prev,
      'provision-workbench': {
        ...prev['provision-workbench'],
        logs: [
          ...(prev['provision-workbench']?.logs || []),
          { timestamp, message: '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', type: 'info' as const },
          { timestamp, message: '🔬 Workbench ready! Open JupyterLab to continue.', type: 'success' as const },
          { timestamp, message: '📓 Run notebook cells to create bucket & launch pipeline', type: 'info' as const },
          { timestamp, message: '👁️ Monitoring mode active - watching for changes...', type: 'info' as const },
        ]
      },
      'storage-bucket': {
        status: 'pending',
        logs: [{ timestamp, message: '⏳ Waiting for researcher to create bucket from notebook...', type: 'info' as const }]
      },
      'deploy-endpoint': {
        status: 'pending',
        logs: [{ timestamp, message: '⏳ Waiting for researcher to deploy MedSigLIP from notebook...', type: 'info' as const }]
      }
    }));

    // Zoom to show full view including researcher area
    zoomToPipeline();
    
    // Start polling for bucket and pipeline status
    startPolling();
  };

  const stopExecution = () => {
    if (abortControllerRef.current) abortControllerRef.current.abort();
    stopPolling();
    setIsRunning(false);
  };

  const selectedStepLogs = selectedStep ? stepStatuses[selectedStep]?.logs || [] : [];
  
  const allSteps = [
    ...INFRA_STEPS,
    { id: 'provision-workbench', label: 'Provision Workbench', command: 'Vertex AI Workbench (GPU)', icon: 'terminal' },
    { id: 'storage-bucket', label: 'Storage Bucket', command: `gs://${config.bucketName}`, icon: 'cloud_upload' },
    { id: 'deploy-endpoint', label: 'Deploy MedSigLIP', command: 'medsiglip-448 → Vertex AI Endpoint', icon: 'rocket_launch' },
    { id: 'generate-embeddings', label: 'Generate Embeddings', command: '768-dim image + text vectors', icon: 'data_array' },
    { id: 'zero-shot', label: 'Zero-Shot Classification', command: 'Text-prompt tissue classification', icon: 'category' },
    { id: 'fine-tune', label: 'Fine-Tune Model', command: 'HF Trainer on NCT-CRC-HE-100K', icon: 'model_training' },
    { id: 'evaluate-model', label: 'Evaluate Model', command: 'Accuracy + F1 on CRC-VAL-HE-7K', icon: 'assessment' },
  ];
  const selectedStepData = allSteps.find(s => s.id === selectedStep);

  return (
    <div className="workload-flow-wrapper">
      <div className="workload-header">
        <div className="header-left">
          <span className="material-symbols-outlined header-icon">biotech</span>
          <div>
            <h1 className="title-large">MedSigLIP on Google Cloud</h1>
            <p className="body-medium header-subtitle">
              {isMonitoringMode 
                ? '👁️ Monitoring Mode — Watching for researcher actions in notebook'
                : 'Medical Foundation Model Workload Visualization'}
            </p>
          </div>
        </div>
        <div className="header-right">
          {isMonitoringMode && (
            <div className="monitoring-indicator">
              <span className="material-symbols-outlined pulse-icon">visibility</span>
              <span className="label-medium">Monitoring</span>
            </div>
          )}
          {isRunning ? (
            <button className="stop-button" onClick={stopExecution}>
              <span className="material-symbols-outlined">stop</span>
              <span className="label-large">Stop</span>
            </button>
          ) : isMonitoringMode ? (
            <button className="stop-button" onClick={stopPolling} style={{ background: '#5F6368' }}>
              <span className="material-symbols-outlined">visibility_off</span>
              <span className="label-large">Stop Monitoring</span>
            </button>
          ) : (
            <button className="run-button" onClick={runAllSteps}>
              <span className="material-symbols-outlined">play_arrow</span>
              <span className="label-large">Run Workflow</span>
            </button>
          )}
        </div>
      </div>

      <div className="workload-content">
        <div className="flow-container">
          <CostInfoTooltip />
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            connectionLineType={ConnectionLineType.SmoothStep}
            fitView
            fitViewOptions={{ padding: 0.15 }}
            proOptions={{ hideAttribution: true }}
          >
            <Background color="#E8EAED" gap={16} />
            <Controls />
          </ReactFlow>
        </div>

        <div className="logs-panel">
          <ExecutionBox
            title={selectedStepData?.label || 'Select a step'}
            command={selectedStepData?.command || ''}
            status={selectedStep ? stepStatuses[selectedStep]?.status || 'pending' : 'pending'}
            logs={selectedStepLogs}
          />
        </div>
      </div>
    </div>
  );
};

interface WorkloadFlowProps {
  onComplete?: () => void;
}

export const WorkloadFlow: React.FC<WorkloadFlowProps> = ({ onComplete }) => {
  return (
    <ReactFlowProvider>
      <WorkloadFlowInner onComplete={onComplete} />
    </ReactFlowProvider>
  );
};
