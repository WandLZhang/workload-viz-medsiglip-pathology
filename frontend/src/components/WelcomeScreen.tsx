import React from 'react';
import './WelcomeScreen.css';

interface WelcomeScreenProps {
  onGetStarted: () => void;
  isTransitioning?: boolean;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onGetStarted, isTransitioning }) => {
  return (
    <div className="welcome-screen">
      <div className={`welcome-card ${isTransitioning ? 'fade-out' : ''}`}>
        <div className="welcome-logo">
          <span className="material-symbols-outlined" style={{ fontSize: '48px' }}>
            biotech
          </span>
        </div>
        
        <h1 className="display-small welcome-title">
          <span className="title-part-1">MedSigLIP Pathology</span>{' '}
          <span className="title-part-2">Workload Visualizer</span>
        </h1>
        
        <p className="body-large welcome-subtitle">
          Foundation model embeddings &amp; fine-tuning for DICOM pathology images on Google Cloud
        </p>
        
        <button className="welcome-button" onClick={onGetStarted}>
          <span className="label-large">Get Started</span>
          <span className="material-symbols-outlined" style={{ fontSize: '20px', marginLeft: '8px' }}>
            arrow_forward
          </span>
        </button>
      </div>
    </div>
  );
};
