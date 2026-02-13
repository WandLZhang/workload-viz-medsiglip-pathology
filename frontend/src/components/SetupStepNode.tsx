import React from 'react';
import { Handle, Position } from 'reactflow';
import './StepNode.css';

interface SetupStepNodeData {
  label: string;
  command: string;
  icon: string;
  status: 'pending' | 'running' | 'complete' | 'error';
  isSelected: boolean;
  onClick: () => void;
}

export const SetupStepNode: React.FC<{ data: SetupStepNodeData }> = ({ data }) => {
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

  return (
    <div 
      className={`step-node setup-node ${getStatusClass()} ${data.isSelected ? 'node-selected' : ''}`}
      onClick={data.onClick}
    >
      {/* Vertical handles for setup phase: top (input) and bottom (output) */}
      <Handle type="target" position={Position.Top} id="target-top" />
      
      <div className="node-content">
        <div className="node-icon-container">
          <span className="material-symbols-outlined node-icon">{data.icon}</span>
        </div>
        <div className="node-info">
          <div className="node-label">{data.label}</div>
          <div className="node-command">{data.command.substring(0, 30)}{data.command.length > 30 ? '...' : ''}</div>
        </div>
        <div className="node-status">
          {getStatusIcon()}
        </div>
      </div>

      <Handle type="source" position={Position.Bottom} id="source-bottom" />
      <Handle type="source" position={Position.Right} id="source-right" />
      <Handle type="source" position={Position.Left} id="source-left" />
    </div>
  );
};
