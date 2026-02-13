import React from 'react';
import './GroupNode.css';

interface GroupNodeData {
  label: string;
  icon: string;
  width: number;
  height: number;
  groupType: 'it' | 'researcher';
}

export const GroupNode: React.FC<{ data: GroupNodeData }> = ({ data }) => {
  const groupClass = data.groupType === 'it' ? 'group-it' : 'group-researcher';
  
  return (
    <div 
      className={`group-node ${groupClass}`}
      style={{ width: data.width, height: data.height }}
    >
      <div className="group-label">
        <span className="material-symbols-outlined group-icon">{data.icon}</span>
        <span className="group-text">{data.label}</span>
      </div>
    </div>
  );
};
