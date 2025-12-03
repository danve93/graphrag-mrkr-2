import React, { useState } from 'react';
import { Tooltip, IconButton, Chip, Button } from '@mui/material';
import { RestartAlt as RetryIcon, ExpandMore as ExpandIcon, ExpandLess as CollapseIcon } from '@mui/icons-material';

interface CategoryRetryProps {
  query: string;
  currentCategories: string[];
  onRetry: (query: string, selectedCategories: string[]) => void;
}

const AVAILABLE_CATEGORIES = [
  { id: 'general', label: 'General', color: '#6b7280' },
  { id: 'installation', label: 'Installation', color: '#10b981' },
  { id: 'configuration', label: 'Configuration', color: '#3b82f6' },
  { id: 'troubleshooting', label: 'Troubleshooting', color: '#ef4444' },
  { id: 'api', label: 'API', color: '#8b5cf6' },
  { id: 'conceptual', label: 'Conceptual', color: '#06b6d4' },
  { id: 'quickstart', label: 'Quick Start', color: '#f59e0b' },
  { id: 'reference', label: 'Reference', color: '#ec4899' },
  { id: 'example', label: 'Examples', color: '#14b8a6' },
  { id: 'best_practices', label: 'Best Practices', color: '#a855f7' },
];

const CategoryRetry: React.FC<CategoryRetryProps> = ({ query, currentCategories, onRetry }) => {
  const [expanded, setExpanded] = useState(false);
  const [selectedCategories, setSelectedCategories] = useState<string[]>(currentCategories);

  const handleCategoryToggle = (categoryId: string) => {
    setSelectedCategories((prev) => {
      if (prev.includes(categoryId)) {
        // Don't allow removing all categories
        if (prev.length === 1) return prev;
        return prev.filter((id) => id !== categoryId);
      } else {
        return [...prev, categoryId];
      }
    });
  };

  const handleRetry = () => {
    onRetry(query, selectedCategories);
    setExpanded(false);
  };

  // Suggest alternative categories (exclude current ones)
  const suggestedCategories = AVAILABLE_CATEGORIES
    .filter((cat) => !currentCategories.includes(cat.id))
    .slice(0, 3);

  if (!expanded) {
    return (
      <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
        <Tooltip title="Search in different categories" arrow>
          <Button
            size="small"
            startIcon={<RetryIcon />}
            onClick={() => setExpanded(true)}
            style={{
              fontSize: '0.75rem',
              textTransform: 'none',
              color: 'var(--text-secondary)',
              borderColor: 'var(--border)',
            }}
            variant="outlined"
          >
            Try Different Categories
          </Button>
        </Tooltip>
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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)' }}>
          Select Categories to Search
        </div>
        <IconButton size="small" onClick={() => setExpanded(false)}>
          <CollapseIcon fontSize="small" />
        </IconButton>
      </div>

      {suggestedCategories.length > 0 && (
        <div style={{ marginBottom: '12px' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
            Suggested alternatives:
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {suggestedCategories.map((cat) => (
              <Chip
                key={cat.id}
                label={cat.label}
                size="small"
                onClick={() => handleCategoryToggle(cat.id)}
                style={{
                  backgroundColor: selectedCategories.includes(cat.id) ? `${cat.color}30` : 'transparent',
                  border: `1px solid ${selectedCategories.includes(cat.id) ? cat.color : 'var(--border)'}`,
                  color: selectedCategories.includes(cat.id) ? cat.color : 'var(--text-secondary)',
                  cursor: 'pointer',
                }}
              />
            ))}
          </div>
        </div>
      )}

      <div style={{ marginBottom: '12px' }}>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
          All categories ({selectedCategories.length} selected):
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
          {AVAILABLE_CATEGORIES.map((cat) => (
            <Chip
              key={cat.id}
              label={cat.label}
              size="small"
              onClick={() => handleCategoryToggle(cat.id)}
              style={{
                backgroundColor: selectedCategories.includes(cat.id) ? `${cat.color}30` : 'transparent',
                border: `1px solid ${selectedCategories.includes(cat.id) ? cat.color : 'var(--border)'}`,
                color: selectedCategories.includes(cat.id) ? cat.color : 'var(--text-secondary)',
                cursor: 'pointer',
                opacity: selectedCategories.includes(cat.id) ? 1 : 0.6,
              }}
            />
          ))}
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
        <Button
          size="small"
          onClick={() => setExpanded(false)}
          style={{ textTransform: 'none', color: 'var(--text-secondary)' }}
        >
          Cancel
        </Button>
        <Button
          size="small"
          variant="contained"
          onClick={handleRetry}
          disabled={selectedCategories.length === 0}
          startIcon={<RetryIcon />}
          style={{
            textTransform: 'none',
            backgroundColor: 'var(--accent-primary)',
            color: 'white',
          }}
        >
          Search Again
        </Button>
      </div>
    </div>
  );
};

export default CategoryRetry;
