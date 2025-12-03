'use client';

import DatabaseTab from './DatabaseTab';
import { useChatStore } from '@/store/chatStore';

export default function DatabaseSidebarContent() {
  const isConnected = useChatStore((s) => s.isConnected);

  return (
    <div className={`flex-1 overflow-y-auto overscroll-contain p-6 transition-all duration-300 ${
      !isConnected ? 'blur-sm pointer-events-none' : ''
    }`}>
      <DatabaseTab />
    </div>
  );
}
