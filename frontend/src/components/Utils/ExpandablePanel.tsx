import React, { ReactNode } from 'react';
import { IconButton } from '@mui/material';
import { ExpandMore as ExpandIcon, ExpandLess as CollapseIcon } from '@mui/icons-material';

interface ExpandablePanelProps {
  title: string;
  expanded: boolean;
  onToggle: () => void;
  children: ReactNode;
}

/**
 * ExpandablePanel component following CategoryRetry layout pattern
 * 
 * Visual structure:
 * - Collapsed: Clean header bar with title and expand icon
 * - Expanded: Panel with:
 *   - Header (title + collapse icon)
 *   - Content area (children)
 *   - Consistent spacing and borders
 */
const ExpandablePanel: React.FC<ExpandablePanelProps> = ({ title, expanded, onToggle, children }) => {
  if (!expanded) {
    return (
      <div
        style={{
          padding: '12px',
          borderRadius: '8px',
          backgroundColor: 'var(--bg-secondary)',
          border: '1px solid var(--border)',
          cursor: 'pointer',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          transition: 'all 0.2s',
        }}
        onClick={onToggle}
      >
        <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)' }}>
          {title}
        </div>
        <IconButton size="small">
          <ExpandIcon fontSize="small" />
        </IconButton>
      </div>
    );
  }

  return (
    <div
      style={{
        padding: '12px',
        borderRadius: '8px',
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border)',
      }}
    >
      {/* Header */}
      <div 
        style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          marginBottom: '12px',
          cursor: 'pointer'
        }}
        onClick={onToggle}
      >
        <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)' }}>
          {title}
        </div>
        <IconButton size="small">
          <CollapseIcon fontSize="small" />
        </IconButton>
      </div>

      {/* Content */}
      <div>
        {children}
      </div>
    </div>
  );
};

export default ExpandablePanel;
