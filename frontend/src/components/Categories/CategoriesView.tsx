'use client';

import { useState, useEffect } from 'react';
import { FolderTree, Check, X, Trash2, RefreshCw, Sparkles, MessageSquare, Edit3, Save, XCircle } from 'lucide-react';
import { api } from '@/lib/api';

interface Category {
  id: string;
  name: string;
  description: string;
  keywords: string[];
  patterns: string[];
  approved: boolean;
  document_count: number;
  children?: string[];
  created_at?: string;
}

interface ProposedCategory {
  name: string;
  description: string;
  keywords: string[];
  patterns: string[];
  confidence: number;
}

interface CategoryPrompt {
  category: string;
  retrieval_strategy: string;
  generation_template: string;
  format_instructions: string;
  specificity_level: string;
}

type TabType = 'categories' | 'prompts';

export default function CategoriesView() {
  const [activeTab, setActiveTab] = useState<TabType>('categories');
  const [categories, setCategories] = useState<Category[]>([]);
  const [prompts, setPrompts] = useState<Record<string, CategoryPrompt>>({});
  const [proposedCategories, setProposedCategories] = useState<ProposedCategory[]>([]);
  const [loading, setLoading] = useState(false);
  const [promptsLoading, setPromptsLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [generatingProgress, setGeneratingProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [showApprovalQueue, setShowApprovalQueue] = useState(false);
  const [editingPromptCategory, setEditingPromptCategory] = useState<string | null>(null);
  const [promptEditForm, setPromptEditForm] = useState<CategoryPrompt>({
    category: '',
    retrieval_strategy: 'balanced',
    generation_template: '',
    format_instructions: '',
    specificity_level: 'detailed'
  });

  // Listen for section selection from sidebar
  useEffect(() => {
    const handleSectionSelect = (event: CustomEvent<string>) => {
      setActiveTab(event.detail as TabType);
    };

    window.addEventListener('categories-section-select', handleSectionSelect as EventListener);
    return () => {
      window.removeEventListener('categories-section-select', handleSectionSelect as EventListener);
    };
  }, []);

  // Broadcast active section changes to sidebar
  useEffect(() => {
    window.dispatchEvent(new CustomEvent('categories-active-section-changed', { detail: activeTab }));
  }, [activeTab]);

  useEffect(() => {
    loadCategories();
    loadPrompts();
  }, []);

  const loadPrompts = async () => {
    try {
      setPromptsLoading(true);
      setError(null);
      const data = await api.getPrompts();
      setPrompts(data.prompts || {});
    } catch (err: any) {
      setError(err.message || 'Failed to load prompts');
    } finally {
      setPromptsLoading(false);
    }
  };

  const loadCategories = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getCategories(false);
      setCategories(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load categories');
    } finally {
      setLoading(false);
    }
  };

  const generateCategories = async () => {
    try {
      setGenerating(true);
      setError(null);
      setGeneratingProgress(0);

      // Simulated progress: climbs to 95% until the request completes
      const start = Date.now();
      const interval = setInterval(() => {
        setGeneratingProgress((prev) => {
          const elapsed = Math.max(0, Date.now() - start);
          // Ease towards 95%; cap growth per tick
          const target = Math.min(95, 10 + Math.sqrt(elapsed / 800) * 25);
          const next = Math.min(95, Math.max(prev, Math.floor(target)));
          return next;
        });
      }, 250);

      const data = await api.generateCategories(10, 100);
      setProposedCategories(data.categories || []);
      setShowApprovalQueue(true);

      setGeneratingProgress(100);
      clearInterval(interval);
    } catch (err: any) {
      setGeneratingProgress(0);
      setError(err.message || 'Failed to generate categories');
    } finally {
      setGenerating(false);
    }
  };

  const approveProposed = async (proposed: ProposedCategory) => {
    try {
      const createResponse = await api.createCategory({
        name: proposed.name,
        description: proposed.description,
        keywords: proposed.keywords,
        patterns: proposed.patterns || [],
        parent_id: null
      });

      await api.approveCategory(createResponse.id);
      setProposedCategories(prev => prev.filter(c => c.name !== proposed.name));
      await loadCategories();
    } catch (err: any) {
      setError(err.message || 'Failed to approve category');
    }
  };

  const rejectProposed = (proposed: ProposedCategory) => {
    setProposedCategories(prev => prev.filter(c => c.name !== proposed.name));
  };

  const deleteCategory = async (categoryId: string) => {
    if (!confirm('Are you sure you want to delete this category?')) return;

    try {
      await api.deleteCategory(categoryId);
      await loadCategories();
    } catch (err: any) {
      setError(err.message || 'Failed to delete category');
    }
  };

  const approveCategory = async (categoryId: string) => {
    try {
      await api.approveCategory(categoryId);
      await loadCategories();
    } catch (err: any) {
      setError(err.message || 'Failed to approve category');
    }
  };

  const autoCategorize = async () => {
    try {
      setLoading(true);
      const data = await api.autoCategorizeDocuments(10);
      alert(`Auto-categorization complete:\n${JSON.stringify(data.statistics, null, 2)}`);
      await loadCategories();
    } catch (err: any) {
      setError(err.message || 'Failed to auto-categorize documents');
    } finally {
      setLoading(false);
    }
  };

  const approvedCategories = categories.filter(c => c.approved);
  const pendingCategories = categories.filter(c => !c.approved);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editForm, setEditForm] = useState<{ name: string; description: string; keywords: string; patterns: string }>({ name: '', description: '', keywords: '', patterns: '' });

  const startEdit = (category: Category) => {
    setEditingId(category.id);
    setEditForm({
      name: category.name,
      description: category.description,
      keywords: (category.keywords || []).join(', '),
      patterns: (category.patterns || []).join(', '),
    });
  };

  const cancelEdit = () => {
    setEditingId(null);
  };

  const saveEdit = async (categoryId: string) => {
    try {
      const payload = {
        name: editForm.name,
        description: editForm.description,
        keywords: editForm.keywords.split(',').map(k => k.trim()).filter(Boolean),
        patterns: editForm.patterns.split(',').map(p => p.trim()).filter(Boolean),
      };
      await api.updateCategory(categoryId, payload);
      setEditingId(null);
      await loadCategories();
    } catch (err: any) {
      setError(err.message || 'Failed to update category');
    }
  };

  // Prompt management functions
  const startEditPrompt = (category: string, prompt: CategoryPrompt) => {
    setEditingPromptCategory(category);
    setPromptEditForm({ ...prompt, category });
  };

  const cancelEditPrompt = () => {
    setEditingPromptCategory(null);
  };

  const savePrompt = async () => {
    try {
      if (!editingPromptCategory) return;

      const payload = {
        retrieval_strategy: promptEditForm.retrieval_strategy,
        generation_template: promptEditForm.generation_template,
        format_instructions: promptEditForm.format_instructions,
        specificity_level: promptEditForm.specificity_level
      };

      await api.updatePrompt(editingPromptCategory, payload);
      setEditingPromptCategory(null);
      await loadPrompts();
    } catch (err: any) {
      setError(err.message || 'Failed to update prompt');
    }
  };

  const deletePromptTemplate = async (category: string) => {
    if (!confirm(`Delete prompt template for "${category}"?`)) return;
    try {
      await api.deletePrompt(category);
      await loadPrompts();
    } catch (err: any) {
      setError(err.message || 'Failed to delete prompt');
    }
  };

  const reloadAllPrompts = async () => {
    try {
      await api.reloadPrompts();
      await loadPrompts();
      alert('Prompts reloaded from config file');
    } catch (err: any) {
      setError(err.message || 'Failed to reload prompts');
    }
  };

  return (
    <div className="h-full flex flex-col" style={{ background: 'var(--bg-primary)' }}>
      {/* Header */}
      <div style={{ borderBottom: '1px solid var(--border)', padding: 'var(--space-6)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: 'var(--space-2)' }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '8px',
            backgroundColor: 'var(--accent-subtle)',
            border: '1px solid var(--accent-primary)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            {activeTab === 'categories' ? (
              <FolderTree className="w-6 h-6" style={{ color: 'var(--accent-primary)' }} />
            ) : (
              <MessageSquare className="w-6 h-6" style={{ color: 'var(--accent-primary)' }} />
            )}
          </div>
          <div style={{ flex: 1 }}>
            <h1 className="font-display" style={{ fontSize: 'var(--text-2xl)', fontWeight: 700, color: 'var(--text-primary)' }}>
              {activeTab === 'categories' ? 'Document Categories' : 'Category Prompts'}
            </h1>
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
              {activeTab === 'categories'
                ? 'LLM-generated taxonomy for query routing and document organization'
                : 'Category-specific prompt templates for improved answer quality'
              }
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {activeTab === 'categories' ? (
              <>
                <button
                  onClick={() => setShowApprovalQueue(!showApprovalQueue)}
                  className="button-secondary flex items-center gap-2"
                  style={{ fontSize: '0.75rem' }}
                >
                  {showApprovalQueue ? <X className="w-4 h-4" /> : <Check className="w-4 h-4" />}
                  {showApprovalQueue ? 'Hide Queue' : `Approval Queue (${proposedCategories.length + pendingCategories.length})`}
                </button>
                <button
                  onClick={generateCategories}
                  disabled={generating}
                  className="button-primary flex items-center gap-2"
                  style={{ fontSize: '0.75rem' }}
                >
                  <Sparkles className="w-4 h-4" />
                  {generating ? `Generating ${generatingProgress}%` : 'Generate Categories'}
                </button>
              </>
            ) : (
              <button
                onClick={reloadAllPrompts}
                disabled={promptsLoading}
                className="button-secondary flex items-center gap-2"
                style={{ fontSize: '0.75rem' }}
              >
                <RefreshCw className="w-4 h-4" />
                Reload from File
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mx-6 mt-4 p-4 rounded-lg" style={{ background: 'var(--error-bg)', border: '1px solid var(--error-border)' }}>
          <p className="text-sm" style={{ color: 'var(--error-text)' }}>{error}</p>
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6 pb-28">
        {activeTab === 'categories' ? renderCategoriesTab() : renderPromptsTab()}
      </div>
    </div>
  );

  function renderCategoriesTab() {
    return (
      <>
        {/* Approval Queue */}
        {showApprovalQueue && (proposedCategories.length > 0 || pendingCategories.length > 0) && (
          <div className="mb-6 p-4 rounded-lg" style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border)' }}>
            <h2 className="text-lg font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>
              Approval Queue
            </h2>

            {/* Proposed by LLM */}
            {proposedCategories.length > 0 && (
              <div className="mb-4">
                <h3 className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
                  LLM Generated ({proposedCategories.length})
                </h3>
                <div className="space-y-2">
                  {proposedCategories.map((proposed, idx) => (
                    <div key={idx} className="p-3 rounded" style={{ background: 'var(--bg-secondary)' }}>
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <h4 className="font-medium" style={{ color: 'var(--text-primary)' }}>{proposed.name}</h4>
                          <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>{proposed.description}</p>
                          <div className="flex flex-wrap gap-1 mt-2">
                            {proposed.keywords.slice(0, 5).map((kw, i) => (
                              <span key={i} className="px-2 py-0.5 text-xs rounded" style={{ background: 'var(--accent-subtle)', color: 'var(--accent-primary)' }}>
                                {kw}
                              </span>
                            ))}
                          </div>
                          <p className="text-xs mt-2" style={{ color: 'var(--text-secondary)' }}>
                            Confidence: {(proposed.confidence * 100).toFixed(0)}%
                          </p>
                        </div>
                        <div className="flex gap-2">
                          <button onClick={() => approveProposed(proposed)} className="button-primary text-sm">
                            <Check className="w-3 h-3 inline mr-1" />
                            Approve
                          </button>
                          <button onClick={() => rejectProposed(proposed)} className="button-secondary text-sm">
                            <X className="w-3 h-3 inline mr-1" />
                            Reject
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Pending Approval */}
            {pendingCategories.length > 0 && (
              <div>
                <h3 className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
                  Pending Approval ({pendingCategories.length})
                </h3>
                <div className="space-y-2">
                  {pendingCategories.map((category) => (
                    <div key={category.id} className="p-3 rounded flex items-start justify-between" style={{ background: 'var(--bg-secondary)' }}>
                      <div className="flex-1">
                        <h4 className="font-medium" style={{ color: 'var(--text-primary)' }}>{category.name}</h4>
                        <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>{category.description}</p>
                      </div>
                      <button onClick={() => approveCategory(category.id)} className="button-primary text-sm">
                        <Check className="w-3 h-3 inline mr-1" />
                        Approve
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Categories Grid */}
        {loading && !generating ? (
          <div className="text-center py-12" style={{ color: 'var(--text-secondary)' }}>
            <RefreshCw className="w-8 h-8 mx-auto mb-2 animate-spin" />
            <p>Loading categories...</p>
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
                Approved Categories ({approvedCategories.length})
              </h2>
              <button onClick={autoCategorize} disabled={loading} className="button-secondary text-sm">
                Auto-Categorize Documents
              </button>
            </div>

            {approvedCategories.length === 0 ? (
              <div className="text-center py-12" style={{ color: 'var(--text-secondary)' }}>
                <FolderTree className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>No categories yet. Generate some to get started!</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {approvedCategories.map((category) => (
                  <div
                    key={category.id}
                    className="p-4 rounded-lg flex flex-col min-w-0"
                    style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}
                  >
                    {/* Header with title and actions */}
                    <div className="flex flex-col gap-2 mb-2">
                      {editingId === category.id ? (
                        <div className="flex-1 space-y-2">
                          <input className="input w-full" value={editForm.name} onChange={(e) => setEditForm({ ...editForm, name: e.target.value })} />
                          <textarea className="textarea w-full" value={editForm.description} onChange={(e) => setEditForm({ ...editForm, description: e.target.value })} />
                        </div>
                      ) : (
                        <h3 className="font-semibold text-lg truncate" style={{ color: 'var(--text-primary)' }} title={category.name}>
                          {category.name}
                        </h3>
                      )}
                      <div className="flex items-center gap-2 flex-shrink-0">
                        {editingId === category.id ? (
                          <>
                            <button onClick={() => saveEdit(category.id)} className="button-secondary text-xs">Save</button>
                            <button onClick={cancelEdit} className="button-secondary text-xs">Cancel</button>
                          </>
                        ) : (
                          <>
                            <button onClick={() => startEdit(category)} className="button-secondary text-xs">Edit</button>
                            <button onClick={() => deleteCategory(category.id)} className="p-1 rounded hover:bg-red-100 dark:hover:bg-red-900/20 transition-colors" title="Delete">
                              <Trash2 className="w-4 h-4 text-red-600" />
                            </button>
                          </>
                        )}
                      </div>
                    </div>

                    {/* Edit form or description */}
                    {editingId === category.id ? (
                      <div className="space-y-2 mb-3">
                        <label className="text-xs" style={{ color: 'var(--text-secondary)' }}>Keywords (comma-separated)</label>
                        <input className="input w-full" value={editForm.keywords} onChange={(e) => setEditForm({ ...editForm, keywords: e.target.value })} />
                        <label className="text-xs" style={{ color: 'var(--text-secondary)' }}>Patterns (comma-separated)</label>
                        <input className="input w-full" value={editForm.patterns} onChange={(e) => setEditForm({ ...editForm, patterns: e.target.value })} />
                      </div>
                    ) : (
                      <p className="text-sm mb-3 line-clamp-2 flex-grow" style={{ color: 'var(--text-secondary)' }}>
                        {category.description}
                      </p>
                    )}

                    {/* Keywords */}
                    <div className="flex flex-wrap gap-2 mb-3">
                      {category.keywords.slice(0, 3).map((keyword, i) => (
                        <span key={i} className="px-2 py-1 text-xs rounded truncate max-w-full" style={{ background: 'var(--accent-subtle)', color: 'var(--accent-primary)' }}>
                          {keyword}
                        </span>
                      ))}
                      {category.keywords.length > 3 && (
                        <span className="px-2 py-1 text-xs flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>
                          +{category.keywords.length - 3} more
                        </span>
                      )}
                    </div>

                    {/* Document count */}
                    <div className="text-xs mt-auto" style={{ color: 'var(--text-secondary)' }}>
                      {category.document_count} document{category.document_count !== 1 ? 's' : ''}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </>
    );
  }

  function renderPromptsTab() {
    const promptEntries = Object.entries(prompts);

    return (
      <>
        {promptsLoading ? (
          <div className="text-center py-12" style={{ color: 'var(--text-secondary)' }}>
            <RefreshCw className="w-8 h-8 mx-auto mb-2 animate-spin" />
            <p>Loading prompts...</p>
          </div>
        ) : promptEntries.length === 0 ? (
          <div className="text-center py-12" style={{ color: 'var(--text-secondary)' }}>
            <MessageSquare className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No prompts configured.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {promptEntries.map(([category, prompt]) => (
              <div key={category} className="p-4 rounded-lg" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <h3 className="font-semibold text-lg capitalize" style={{ color: 'var(--text-primary)' }}>
                      {category}
                    </h3>
                    <div className="flex gap-3 mt-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
                      <span>Strategy: <strong>{prompt.retrieval_strategy}</strong></span>
                      <span>•</span>
                      <span>Level: <strong>{prompt.specificity_level}</strong></span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {editingPromptCategory === category ? (
                      <>
                        <button onClick={savePrompt} className="button-primary text-sm flex items-center gap-1">
                          <Save className="w-3 h-3" />
                          Save
                        </button>
                        <button onClick={cancelEditPrompt} className="button-secondary text-sm flex items-center gap-1">
                          <XCircle className="w-3 h-3" />
                          Cancel
                        </button>
                      </>
                    ) : (
                      <>
                        <button onClick={() => startEditPrompt(category, prompt)} className="button-secondary text-sm flex items-center gap-1">
                          <Edit3 className="w-3 h-3" />
                          Edit
                        </button>
                        {category !== 'default' && (
                          <button onClick={() => deletePromptTemplate(category)} className="p-1 rounded hover:bg-red-100 dark:hover:bg-red-900/20 transition-colors" title="Delete">
                            <Trash2 className="w-4 h-4 text-red-600" />
                          </button>
                        )}
                      </>
                    )}
                  </div>
                </div>

                {editingPromptCategory === category ? (
                  <div className="space-y-3">
                    <div>
                      <label className="text-sm font-medium mb-1 block" style={{ color: 'var(--text-primary)' }}>Retrieval Strategy</label>
                      <select
                        className="input"
                        value={promptEditForm.retrieval_strategy}
                        onChange={(e) => setPromptEditForm({ ...promptEditForm, retrieval_strategy: e.target.value })}
                      >
                        <option value="balanced">Balanced</option>
                        <option value="step_back">Step Back (Procedural)</option>
                        <option value="ppr">PPR (Technical Reference)</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-sm font-medium mb-1 block" style={{ color: 'var(--text-primary)' }}>Specificity Level</label>
                      <select
                        className="input"
                        value={promptEditForm.specificity_level}
                        onChange={(e) => setPromptEditForm({ ...promptEditForm, specificity_level: e.target.value })}
                      >
                        <option value="concise">Concise</option>
                        <option value="detailed">Detailed</option>
                        <option value="prescriptive">Prescriptive</option>
                        <option value="technical">Technical</option>
                        <option value="explanatory">Explanatory</option>
                        <option value="comprehensive">Comprehensive</option>
                        <option value="practical">Practical</option>
                        <option value="advisory">Advisory</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-sm font-medium mb-1 block" style={{ color: 'var(--text-primary)' }}>Generation Template</label>
                      <textarea
                        className="textarea font-mono text-xs"
                        rows={6}
                        value={promptEditForm.generation_template}
                        onChange={(e) => setPromptEditForm({ ...promptEditForm, generation_template: e.target.value })}
                        placeholder="Include {query} and {context} placeholders"
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium mb-1 block" style={{ color: 'var(--text-primary)' }}>Format Instructions</label>
                      <textarea
                        className="textarea text-sm"
                        rows={3}
                        value={promptEditForm.format_instructions}
                        onChange={(e) => setPromptEditForm({ ...promptEditForm, format_instructions: e.target.value })}
                        placeholder="Guidance for output format and structure"
                      />
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div>
                      <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-secondary)' }}>Generation Template</label>
                      <pre className="p-3 rounded text-xs font-mono overflow-x-auto" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}>
                        {prompt.generation_template}
                      </pre>
                    </div>
                    <div>
                      <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-secondary)' }}>Format Instructions</label>
                      <p className="text-sm p-3 rounded" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}>
                        {prompt.format_instructions}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </>
    );
  }
}
