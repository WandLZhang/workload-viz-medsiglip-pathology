import React, { useState } from 'react';
import { WelcomeScreen } from './components/WelcomeScreen';
import { WorkloadFlow } from './components/WorkloadFlow';
import './App.css';

export const App: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isTransitioning, setIsTransitioning] = useState(false);

  const handleGetStarted = () => {
    setIsTransitioning(true);
    setTimeout(() => {
      setCurrentStep(1);
      setIsTransitioning(false);
    }, 500);
  };

  const handleFlowComplete = () => {
    console.log('Workflow complete');
  };

  return (
    <div>
      {currentStep === 0 && (
        <WelcomeScreen 
          onGetStarted={handleGetStarted} 
          isTransitioning={isTransitioning}
        />
      )}
      {currentStep === 1 && (
        <WorkloadFlow 
          onComplete={handleFlowComplete}
        />
      )}
    </div>
  );
};
