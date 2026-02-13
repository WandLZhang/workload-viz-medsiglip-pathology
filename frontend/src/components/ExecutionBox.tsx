import React, { useEffect, useRef } from 'react';
import { TooltipWrapper } from './TooltipWrapper';
import './ExecutionBox.css';

interface LogEntry {
  timestamp: string;
  message: string;
  type: 'info' | 'success' | 'error';
}

interface ExecutionBoxProps {
  title: string;
  command: string;
  status: 'pending' | 'running' | 'complete' | 'error';
  logs: LogEntry[];
}

export const ExecutionBox: React.FC<ExecutionBoxProps> = ({
  title,
  command,
  status,
  logs,
}) => {
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs to bottom
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const getStatusDisplay = () => {
    switch (status) {
      case 'running': return 'Executing...';
      case 'complete': return 'Complete';
      case 'error': return 'Error';
      default: return 'Pending';
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'running': return '#1A73E8';
      case 'complete': return '#4CAF50';
      case 'error': return '#f44336';
      default: return '#5F6368';
    }
  };

  return (
    <div className="execution-box">
      {/* Header */}
      <div className="execution-header">
        <div className="execution-title-row">
          <span className="material-symbols-outlined execution-icon">terminal</span>
          <h3 className="title-medium">{title}</h3>
        </div>
        <div 
          className="execution-status-badge"
          style={{ 
            backgroundColor: `${getStatusColor()}15`,
            color: getStatusColor(),
            borderColor: getStatusColor()
          }}
        >
          {status === 'running' && <div className="mini-spinner" />}
          {status === 'complete' && <span className="material-symbols-outlined status-badge-icon">check</span>}
          {status === 'error' && <span className="material-symbols-outlined status-badge-icon">error</span>}
          <span className="label-small">{getStatusDisplay()}</span>
        </div>
      </div>

      {/* Command display */}
      {command && (
        <div className="command-display">
          <span className="command-prompt">$</span>
          <code className="command-text">{command}</code>
        </div>
      )}

      {/* Logs container */}
      <div className="logs-container">
        <div className="logs-header">
          <span className="material-symbols-outlined logs-icon">article</span>
          <span className="label-medium">Execution Logs</span>
          <span className="log-count">{logs.length}</span>
        </div>
        
        <TooltipWrapper
          content={<pre>{logs.map(l => `[${l.timestamp}] ${l.message}`).join('\n')}</pre>}
          position="left"
        >
          <div className="logs-scroll-area">
            {logs.length === 0 ? (
              <div className="logs-empty">
                <span className="material-symbols-outlined">info</span>
                <span>No logs yet. Click "Run" to start execution.</span>
              </div>
            ) : (
              logs.map((log, index) => (
                <div 
                  key={index} 
                  className={`log-entry log-${log.type}`}
                >
                  <span className="log-timestamp">[{log.timestamp}]</span>
                  <span className="log-message">
                    {log.type === 'success' && '✓ '}
                    {log.type === 'error' && '✗ '}
                    {log.message}
                  </span>
                </div>
              ))
            )}
            <div ref={logsEndRef} />
          </div>
        </TooltipWrapper>
      </div>

      {/* Progress indicator for running state */}
      {status === 'running' && (
        <div className="execution-progress">
          <div className="wavy-progress">
            <div className="wavy-bar" />
            <div className="wavy-bar" />
            <div className="wavy-bar" />
            <div className="wavy-bar" />
            <div className="wavy-bar" />
          </div>
        </div>
      )}
    </div>
  );
};
