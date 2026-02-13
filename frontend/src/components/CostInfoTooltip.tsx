/**
 * @file CostInfoTooltip.tsx
 * @brief Cost breakdown tooltip showing GCP SKUs and estimated costs for the MedSigLIP pipeline.
 *
 * @details Displays an "i" info icon in the top-right corner of the flowchart viewport.
 * On hover, it expands to show a detailed cost breakdown with actual GCP SKU IDs
 * and pricing for MedSigLIP model deployment, embedding generation, fine-tuning,
 * and evaluation on Vertex AI with GPU-backed endpoints.
 *
 * @author Willis Zhang
 * @date 2026-02-11
 */

import React, { useState } from 'react';
import './CostInfoTooltip.css';

interface SkuInfo {
  skuId: string;
  description: string;
  price: string;
  unit: string;
}

interface CostLineItem {
  component: string;
  resource: string;
  time: string;
  cost: string;
}

const GPU_SKUS: SkuInfo[] = [
  {
    skuId: 'A972-1DF4-4F6E',
    description: 'NVIDIA A100 40GB GPU (On-Demand)',
    price: '$2.934',
    unit: 'GPU-hour',
  },
];

const COMPUTE_SKUS: SkuInfo[] = [
  {
    skuId: 'A2-HIGHGPU-1G',
    description: 'a2-highgpu-1g (12 vCPU, 85 GB RAM, 1x A100)',
    price: '$3.673',
    unit: 'hour (total)',
  },
  {
    skuId: 'PD-SSD-300',
    description: '300 GB PD-SSD Boot Disk',
    price: '$0.17',
    unit: 'GB/month',
  },
];

const STORAGE_SKUS: SkuInfo[] = [
  {
    skuId: 'E5F0-6A5D-7BAD',
    description: 'Standard Storage (Regional)',
    price: '$0.02',
    unit: 'GiB/month',
  },
];

const COST_BREAKDOWN: CostLineItem[] = [
  {
    component: 'A100 Workbench',
    resource: 'a2-highgpu-1g (1x A100 40GB)',
    time: '~4 hr',
    cost: '$14.69',
  },
  {
    component: 'Zero-Shot Eval',
    resource: 'Local GPU inference',
    time: '~15 min',
    cost: '(included)',
  },
  {
    component: 'Fine-Tune',
    resource: 'Local GPU training',
    time: '~2 hr',
    cost: '(included)',
  },
  {
    component: 'Storage',
    resource: '300GB PD-SSD + GCS',
    time: '1 month',
    cost: '$51.10',
  },
];

export const CostInfoTooltip: React.FC = () => {
  const [isExpanded, setIsExpanded] = useState(false);

  const handleMouseEnter = () => setIsExpanded(true);
  const handleMouseLeave = () => setIsExpanded(false);

  const totalCost = COST_BREAKDOWN.reduce((sum, item) => {
    const val = parseFloat(item.cost.replace('$', ''));
    return sum + (isNaN(val) ? 0 : val);
  }, 0);

  return (
    <div
      className="cost-info-tooltip-wrapper"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <button className="cost-info-button" aria-label="View cost breakdown">
        <span className="material-symbols-outlined">info</span>
      </button>

      {isExpanded && (
        <div className="cost-info-panel">
          <div className="cost-info-header">
            <span className="material-symbols-outlined cost-icon">payments</span>
            <h3>Estimated Cost Breakdown</h3>
          </div>

          <div className="cost-info-section">
            <h4>
              <span className="material-symbols-outlined section-icon">memory</span>
              GPU Accelerators
            </h4>
            <table className="sku-table">
              <thead>
                <tr>
                  <th>SKU ID</th>
                  <th>Description</th>
                  <th>Price</th>
                </tr>
              </thead>
              <tbody>
                {GPU_SKUS.map((sku) => (
                  <tr key={sku.skuId}>
                    <td className="sku-id">{sku.skuId}</td>
                    <td>{sku.description}</td>
                    <td className="price">{sku.price}/{sku.unit}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="cost-info-section">
            <h4>
              <span className="material-symbols-outlined section-icon">dns</span>
              Compute (A2 GPU Machine)
            </h4>
            <table className="sku-table">
              <thead>
                <tr>
                  <th>SKU ID</th>
                  <th>Description</th>
                  <th>Price</th>
                </tr>
              </thead>
              <tbody>
                {COMPUTE_SKUS.map((sku) => (
                  <tr key={sku.skuId}>
                    <td className="sku-id">{sku.skuId}</td>
                    <td>{sku.description}</td>
                    <td className="price">{sku.price}/{sku.unit}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="cost-info-section">
            <h4>
              <span className="material-symbols-outlined section-icon">cloud_upload</span>
              Cloud Storage
            </h4>
            <table className="sku-table">
              <thead>
                <tr>
                  <th>SKU ID</th>
                  <th>Description</th>
                  <th>Price</th>
                </tr>
              </thead>
              <tbody>
                {STORAGE_SKUS.map((sku) => (
                  <tr key={sku.skuId}>
                    <td className="sku-id">{sku.skuId}</td>
                    <td>{sku.description}</td>
                    <td className="price">{sku.price}/{sku.unit}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="cost-info-section">
            <h4>
              <span className="material-symbols-outlined section-icon">calculate</span>
              MedSigLIP Pipeline Cost Estimate
            </h4>
            <table className="cost-table">
              <thead>
                <tr>
                  <th>Component</th>
                  <th>Resource</th>
                  <th>Time</th>
                  <th>Cost</th>
                </tr>
              </thead>
              <tbody>
                {COST_BREAKDOWN.map((item) => (
                  <tr key={item.component}>
                    <td className="component">{item.component}</td>
                    <td>{item.resource}</td>
                    <td>{item.time}</td>
                    <td className="cost">{item.cost}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="cost-total">
            <div className="total-row">
              <span className="total-label">
                <span className="material-symbols-outlined">target</span>
                Total Estimated Cost:
              </span>
              <span className="total-value">~${totalCost.toFixed(2)}</span>
            </div>
            <div className="total-note">All inference + fine-tuning runs locally on A100 workbench</div>
          </div>

          <div className="cost-info-footer">
            <div className="batch-note">
              <span className="material-symbols-outlined">check_circle</span>
              Stop the workbench when done to avoid idle charges
            </div>
            <a
              href="https://cloud.google.com/vertex-ai/pricing"
              target="_blank"
              rel="noopener noreferrer"
              className="sku-link"
            >
              <span className="material-symbols-outlined">open_in_new</span>
              View Vertex AI Pricing
            </a>
          </div>
        </div>
      )}
    </div>
  );
};
