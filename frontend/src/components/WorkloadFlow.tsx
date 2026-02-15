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
  'deploy-endpoint': `🧠 Why GCP — Model Garden:
• MedSigLIP is a Google-developed model — available exclusively in Vertex AI Model Garden
• One-click from Model Garden → running on GPU, no container building
• AWS/Azure: model not available; you'd manually download, containerize, deploy
• HuggingFace weights available but GCP gets optimized serving containers first`,
  'generate-embeddings': `📥 Why GCP — Healthcare API Integration:
• Native DICOMweb URIs flow directly into MedSigLIP — no format conversion
• Healthcare API stores DICOM natively; AWS HealthLake requires FHIR conversion
• GCS supports composite uploads for large pathology slides (up to 5 TiB objects)
• Batch prediction: submit 100K images as JSONL, auto-provisioned GPUs
• Azure: no native DICOM → embedding pipeline; requires custom orchestration`,
  'zero-shot': `🎯 Why GCP — Zero-Shot on Medical Data:
• MedSigLIP trained on medical image-text pairs — not general-purpose CLIP
• Outperforms OpenAI CLIP and BiomedCLIP on pathology benchmarks
• No equivalent medical foundation model available on AWS or Azure
• Text-prompt classification: no labeled training data needed`,
  'fine-tune': `🔧 Why GCP — GPU Availability & Cost:
• A100 40GB on-demand: $2.93/hr (GCP) vs $3.67/hr (AWS p4d) vs $3.40/hr (Azure)
• Spot/Preemptible A100: $0.88/hr — 70% savings, best spot pricing among clouds
• Per-second billing (GCP) vs per-hour billing (AWS)
• Cloud NAT + no public IP: secure fine-tuning without internet exposure
• Checkpoints stream directly to GCS — no S3 gateway or egress config needed`,
  'evaluate-model': `📊 Why GCP — Experiment Tracking:
• Vertex AI Experiments: native run tracking with lineage to model artifacts
• Results auto-linked to GCS bucket, model registry, and training job
• AWS SageMaker Experiments exists but lacks model lineage integration
• Azure ML: experiment tracking requires separate workspace configuration`,
  'provision-workbench': `🔬 Why GCP — Vertex AI Workbench:
• Managed JupyterLab with IAP auth — zero-config secure access, no SSH tunnels
• GPU hot-attach: resize from CPU to A100 without recreating the instance
• Pre-authenticated to all GCP services (GCS, Vertex AI, BigQuery) — no credential files
• AWS SageMaker Studio: cold starts of 3-5 min; Workbench boots in ~90s
• Azure ML Compute: requires VNet peering for private access; GCP uses IAP natively`,
  'storage-bucket': `📦 Why GCP — Cloud Storage:
• Fastest time-to-first-byte among all cloud object stores (independent benchmarks)
• Strong consistency on all operations — no eventual consistency surprises
• DICOMweb URIs addressable directly from Healthcare API → MedSigLIP

🌐 Dual-Region Storage:
• Single bucket namespace spanning two regions with zero RTO
• AWS: requires separate buckets + cross-region replication (eventual consistency)
• Azure: GRS replication has 15-minute RPO, not zero`,
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
  region: 'asia-northeast3',
  zone: 'asia-northeast3-b',
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
  
  // Workbench: below infra column
  const WORKBENCH_X = INFRA_START_X;
  const WORKBENCH_Y = INFRA_START_Y + INFRA_STEPS.length * INFRA_Y_GAP + 80;
  
  // Storage Bucket: directly to the right of workbench, same Y level
  const HORIZONTAL_GAP = 450;  // Gap for 400px nodes
  const BUCKET_X = WORKBENCH_X + HORIZONTAL_GAP;
  const BUCKET_Y = WORKBENCH_Y;
  
  // Pipeline tasks flow to the right of bucket
  const PIPELINE_START_X = BUCKET_X + HORIZONTAL_GAP;
  const PARALLEL_Y_OFFSET = 70;  // Vertical offset for parallel tasks above/below center

  const zoomToSetup = useCallback(() => {
    setViewport({ x: 100, y: 0, zoom: 1.2 }, { duration: 800 });
  }, [setViewport]);

  const zoomToPipeline = useCallback(() => {
    setViewport({ x: -100, y: 50, zoom: 0.7 }, { duration: 800 });
  }, [setViewport]);

  const generateNodes = useCallback((): Node[] => {
    const nodes: Node[] = [];
    
    // Calculate group box dimensions
    const nodeHeight = 80;
    const padding = 30;
    const leftPadding = 80;  // Extra left padding for edge clearance and labels
    const saveResultsX = PIPELINE_START_X + HORIZONTAL_GAP * 2;
    
    // IT Group Box: contains only infra steps (vertical column)
    const itBoxX = INFRA_START_X - leftPadding;
    const itBoxY = INFRA_START_Y - padding - 20;  // Extra top for label
    const itBoxWidth = 400 + leftPadding + padding + 40;  // covers infra nodes
    const itBoxHeight = (INFRA_STEPS.length - 1) * INFRA_Y_GAP + nodeHeight + padding * 2 + 20;
    
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
    
    // Researcher Group Box: contains workbench + bucket + all pipeline nodes
    const researcherBoxX = WORKBENCH_X - leftPadding;
    const researcherBoxY = WORKBENCH_Y - PARALLEL_Y_OFFSET - padding - 20;  // covers top parallel node
    const researcherBoxWidth = saveResultsX + 400 + padding - researcherBoxX + 40;
    const researcherBoxHeight = (PARALLEL_Y_OFFSET * 2) + nodeHeight + padding * 2 + 20;
    
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

    // Load MedSigLIP: parallel top, right of bucket
    nodes.push({
      id: 'load-model',
      type: 'pipelineTask',
      position: { x: PIPELINE_START_X, y: WORKBENCH_Y - PARALLEL_Y_OFFSET },
      data: {
        label: 'Load MedSigLIP',
        command: 'from_pretrained("google/medsiglip-448")',
        icon: 'neurology',
        status: stepStatuses['load-model']?.status || 'pending',
        isSelected: selectedStep === 'load-model',
        onClick: () => setSelectedStep('load-model'),
        tooltip: GCP_DIFFERENTIATORS['load-model'],
      },
    });

    // Download Dataset: parallel bottom, right of bucket
    nodes.push({
      id: 'download-dataset',
      type: 'pipelineTask',
      position: { x: PIPELINE_START_X, y: WORKBENCH_Y + PARALLEL_Y_OFFSET },
      data: {
        label: 'Download Dataset',
        command: 'NCT-CRC-HE-100K from Zenodo',
        icon: 'download',
        status: stepStatuses['download-dataset']?.status || 'pending',
        isSelected: selectedStep === 'download-dataset',
        onClick: () => setSelectedStep('download-dataset'),
        tooltip: GCP_DIFFERENTIATORS['download-dataset'],
      },
    });

    // Fine-Tune: after both parallel tasks converge
    nodes.push({
      id: 'fine-tune',
      type: 'pipelineTask',
      position: { x: PIPELINE_START_X + HORIZONTAL_GAP, y: WORKBENCH_Y },
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

    // Save Results to Bucket: terminal step
    nodes.push({
      id: 'save-results',
      type: 'pipelineTask',
      position: { x: saveResultsX, y: WORKBENCH_Y },
      data: {
        label: 'Save Results to Bucket',
        command: 'Checkpoints + eval → GCS',
        icon: 'cloud_upload',
        status: stepStatuses['save-results']?.status || 'pending',
        isSelected: selectedStep === 'save-results',
        onClick: () => setSelectedStep('save-results'),
        tooltip: GCP_DIFFERENTIATORS['save-results'],
        batchJobUrl: `${config.consoleBaseUrl}/storage/browser/${config.bucketName}?project=${config.projectId}`,
      },
    });

    return nodes;
  }, [stepStatuses, selectedStep, workbenchUrl, config, WORKBENCH_Y, BUCKET_Y]);

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

    // Storage Bucket → Load MedSigLIP (parallel fan-out)
    ['load-model', 'download-dataset'].forEach(taskId => {
      const isRunning = stepStatuses[taskId]?.status === 'running';
      const isComplete = stepStatuses[taskId]?.status === 'complete';
      edges.push({
        id: `e-bucket-${taskId}`,
        source: 'storage-bucket', target: taskId,
        sourceHandle: 'source-right', targetHandle: 'target-left',
        type: 'smoothstep',
        animated: stepStatuses['storage-bucket']?.status === 'complete' || isRunning || isComplete,
        style: { stroke: isComplete ? '#4CAF50' : isRunning ? '#1A73E8' : stepStatuses['storage-bucket']?.status === 'complete' ? '#4CAF50' : '#DADCE0', strokeWidth: 2 },
      });
    });

    // Load MedSigLIP → Fine-Tune (convergence)
    edges.push({
      id: 'e-loadmodel-finetune',
      source: 'load-model', target: 'fine-tune',
      sourceHandle: 'source-right', targetHandle: 'target-left',
      type: 'smoothstep',
      animated: stepStatuses['load-model']?.status === 'complete' || stepStatuses['fine-tune']?.status === 'running',
      style: { stroke: stepStatuses['fine-tune']?.status === 'complete' ? '#4CAF50' : stepStatuses['load-model']?.status === 'complete' ? '#1A73E8' : '#DADCE0', strokeWidth: 2 },
    });

    // Download Dataset → Fine-Tune (convergence)
    edges.push({
      id: 'e-dataset-finetune',
      source: 'download-dataset', target: 'fine-tune',
      sourceHandle: 'source-right', targetHandle: 'target-left',
      type: 'smoothstep',
      animated: stepStatuses['download-dataset']?.status === 'complete' || stepStatuses['fine-tune']?.status === 'running',
      style: { stroke: stepStatuses['fine-tune']?.status === 'complete' ? '#4CAF50' : stepStatuses['download-dataset']?.status === 'complete' ? '#1A73E8' : '#DADCE0', strokeWidth: 2 },
    });

    // Fine-Tune → Save Results
    edges.push({
      id: 'e-finetune-saveresults',
      source: 'fine-tune', target: 'save-results',
      sourceHandle: 'source-right', targetHandle: 'target-left',
      type: 'smoothstep',
      animated: stepStatuses['fine-tune']?.status === 'complete' || stepStatuses['save-results']?.status === 'running',
      style: { stroke: stepStatuses['save-results']?.status === 'complete' ? '#4CAF50' : stepStatuses['fine-tune']?.status === 'complete' ? '#1A73E8' : '#DADCE0', strokeWidth: 2 },
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
      
      // Update pipeline task statuses from GCS markers
      if (data.jobs?.taskStatuses) {
        const taskStatuses = data.jobs.taskStatuses;
        const pipelineTasks = ['load-model', 'download-dataset', 'fine-tune', 'save-results'];
        
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
                ? `▶ ${taskId} running...`
                : '';
              
              updates[taskId] = {
                status: newStatus,
                logs: statusMessage ? [...logs, { timestamp, message: statusMessage, type: newStatus === 'complete' ? 'success' as const : 'info' as const }] : logs
              };
            }
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

    // Provision workbench (may resolve to a different zone via fallback)
    if (!abortController.signal.aborted) {
      const success = await runStep('provision-workbench', 'Provision Workbench', abortController.signal);
      if (!success) { 
        setIsRunning(false); 
        return; 
      }
    }

    // Re-fetch config after workbench provisioning — backend may have resolved
    // to a different zone via STOCKOUT fallback, updating REGION/ZONE globals
    try {
      const configRes = await fetch('/api/config');
      const newConfig = await configRes.json();
      console.log('[CONFIG] Re-fetched after workbench:', JSON.stringify(newConfig, null, 2));
      setConfig(newConfig);
    } catch (err) {
      console.error('[CONFIG] Failed to re-fetch config after workbench:', err);
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
      'load-model': {
        status: 'pending',
        logs: [{ timestamp, message: '⏳ Waiting for researcher to load MedSigLIP from notebook...', type: 'info' as const }]
      },
      'download-dataset': {
        status: 'pending',
        logs: [{ timestamp, message: '⏳ Waiting for researcher to download dataset from notebook...', type: 'info' as const }]
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
    { id: 'load-model', label: 'Load MedSigLIP', command: 'from_pretrained("google/medsiglip-448")', icon: 'neurology' },
    { id: 'download-dataset', label: 'Download Dataset', command: 'NCT-CRC-HE-100K from Zenodo', icon: 'download' },
    { id: 'fine-tune', label: 'Fine-Tune Model', command: 'HF Trainer on NCT-CRC-HE-100K', icon: 'model_training' },
    { id: 'save-results', label: 'Save Results to Bucket', command: 'Checkpoints + eval → GCS', icon: 'cloud_upload' },
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
