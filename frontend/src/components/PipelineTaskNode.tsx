import React from 'react';
import { Handle, Position } from 'reactflow';
import { TooltipWrapper } from './TooltipWrapper';
import './StepNode.css';

interface PipelineTaskNodeData {
  label: string;
  command: string;
  icon: string;
  status: 'pending' | 'running' | 'complete' | 'error';
  isSelected: boolean;
  onClick: () => void;
  tooltip?: string;
  batchJobUrl?: string;
}

/**
 * @brief Pipeline task node component for the workflow visualization.
 * 
 * @details This component renders a pipeline task (INDEX, FASTQC, QUANT, MULTIQC, Results, Workbench).
 * It supports horizontal (left/right) and vertical (top/bottom) handles for flexible edge connections.
 * The bottom handle is used for the Results → Workbench loop edge.
 */
export const PipelineTaskNode: React.FC<{ data: PipelineTaskNodeData }> = ({ data }) => {
  const getStatusClass = () => {
    switch (data.status) {
      case 'running': return 'node-running';
      case 'complete': return 'node-complete';
      case 'error': return 'node-error';
      default: return 'node-pending';
    }
  };

  const getStatusIcon = () => {
    switch (data.status) {
      case 'running': return <div className="spinner" />;
      case 'complete': return <span className="material-symbols-outlined status-check">check_circle</span>;
      case 'error': return <span className="material-symbols-outlined status-error">error</span>;
      default: return null;
    }
  };

  const handleLinkClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (data.batchJobUrl) {
      window.open(data.batchJobUrl, '_blank');
    }
  };

  return (
    <div 
      className={`step-node pipeline-node ${getStatusClass()} ${data.isSelected ? 'node-selected' : ''}`}
      onClick={data.onClick}
    >
      {/* All handles for flexible edge connections */}
      <Handle type="target" position={Position.Left} id="target-left" />
      <Handle type="target" position={Position.Top} id="target-top" />
      <Handle type="target" position={Position.Bottom} id="target-bottom" />
      <Handle type="target" position={Position.Right} id="target-right" />
      
      <div className="node-content">
        <div className="node-icon-container pipeline-icon-container">
          <span className="material-symbols-outlined node-icon">{data.icon}</span>
        </div>
        <div className="node-info">
          <div className="node-label">
            {data.label}
            {data.batchJobUrl && (
              <span 
                className="material-symbols-outlined link-icon" 
                onClick={handleLinkClick}
                title="Open in Google Cloud Console"
              >
                open_in_new
              </span>
            )}
          </div>
          <div className="node-command">{data.command.substring(0, 30)}{data.command.length > 30 ? '...' : ''}</div>
        </div>
        <div className="node-status">
          {getStatusIcon()}
        </div>
        
        {/* Info icon with tooltip for GCP differentiators */}
        {data.tooltip && (
          <TooltipWrapper content={data.tooltip} delay={100}>
            <span className="material-symbols-outlined info-icon">info</span>
          </TooltipWrapper>
        )}
      </div>

      <Handle type="source" position={Position.Right} id="source-right" />
      <Handle type="source" position={Position.Bottom} id="source-bottom" />
      <Handle type="source" position={Position.Top} id="source-top" />
    </div>
  );
};
