import { create } from 'zustand';

interface GraphEditorState {
    mode: 'select' | 'connect' | 'heal' | 'prune' | 'orphan';
    selectedNodeId: string | null;
    isSafetyModalOpen: boolean;

    setMode: (mode: 'select' | 'connect' | 'heal' | 'prune' | 'orphan') => void;
    setSelectedNodeId: (id: string | null) => void;
    setSafetyModalOpen: (isOpen: boolean) => void;
}

export const useGraphEditorStore = create<GraphEditorState>((set) => ({
    mode: 'select',
    selectedNodeId: null,
    isSafetyModalOpen: false,

    setMode: (mode) => set({ mode }),
    setSelectedNodeId: (id) => set({ selectedNodeId: id }),
    setSafetyModalOpen: (isOpen) => set({ isSafetyModalOpen: isOpen }),
}));
