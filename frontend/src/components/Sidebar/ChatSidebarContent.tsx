'use client';

import { PlusCircleIcon } from '@heroicons/react/24/outline';
import { useChatStore } from '@/store/chatStore';
import HistoryTab from './HistoryTab';

export default function ChatSidebarContent() {
  const clearChat = useChatStore((state) => state.clearChat);
  const setActiveView = useChatStore((state) => state.setActiveView);
  const isConnected = useChatStore((s) => s.isConnected);

  return (
    <div className={`flex-1 flex flex-col min-h-0 transition-all duration-300 ${
      !isConnected ? 'blur-sm pointer-events-none' : ''
    }`}>
      {/* New Chat button */}
      <div className="px-6 py-3 border-b border-secondary-200 dark:border-secondary-700">
        <button
          type="button"
          onClick={() => {
            clearChat();
            setActiveView('chat');
          }}
          className="w-full button-primary"
          aria-label="Start a new chat"
        >
          <PlusCircleIcon className="w-5 h-5" />
          New Chat
        </button>
      </div>

      {/* History */}
      <div className="flex-1 overflow-y-auto overscroll-contain p-6">
        <HistoryTab />
      </div>
    </div>
  );
}
