import React, { useState, useRef, useEffect } from 'react';
import ReactDOM from 'react-dom';
import './Tooltip.css';

interface TooltipWrapperProps {
  content: React.ReactNode;
  children: React.ReactElement;
  position?: 'top' | 'bottom' | 'left' | 'right';
  delay?: number;
}

export const TooltipWrapper: React.FC<TooltipWrapperProps> = ({
  content,
  children,
  position = 'top',
  delay = 200,
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [coords, setCoords] = useState({ top: 0, left: 0 });
  const timeoutRef = useRef<number | null>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const targetRef = useRef<HTMLSpanElement>(null);

  const handleMouseEnter = (e: React.MouseEvent) => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    const { clientX, clientY } = e;
    timeoutRef.current = window.setTimeout(() => {
      setCoords({ top: clientY + 10, left: clientX + 10 });
      setIsVisible(true);
    }, delay);
  };

  const handleMouseLeave = () => {
    timeoutRef.current = window.setTimeout(() => {
      setIsVisible(false);
    }, 500);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const { clientX, clientY } = e;
    setCoords({ top: clientY + 10, left: clientX + 10 });
  };

  const handleCopy = () => {
    const contentToCopy = tooltipRef.current?.innerText || '';
    navigator.clipboard.writeText(contentToCopy);
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsVisible(false);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  return (
    <>
      <span
        ref={targetRef}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onMouseMove={handleMouseMove}
        style={{ position: 'relative', zIndex: 1, pointerEvents: 'all' }}
      >
        {children}
      </span>
      {isVisible &&
        ReactDOM.createPortal(
          <div
            ref={tooltipRef}
            className="tooltip-container"
            style={{ top: `${coords.top}px`, left: `${coords.left}px` }}
            onMouseEnter={() => {
              if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
              }
            }}
            onMouseLeave={() => {
              setIsVisible(false);
            }}
          >
            <div className="tooltip-header">
              <button onClick={handleCopy} className="copy-button">
                Copy
              </button>
            </div>
            <div className="tooltip-content">{content}</div>
          </div>,
          document.body
        )}
    </>
  );
};
