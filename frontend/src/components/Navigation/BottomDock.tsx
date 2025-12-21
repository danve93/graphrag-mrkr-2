'use client';

import { useState, useEffect } from 'react';
import { useChatStore } from '@/store/chatStore';
import { Dock, DockIcon, DockItem, DockLabel } from '@/../components/motion-primitives/dock';
import {
  MessageSquare,
  Database,
  Network,
  FolderTree,
  Route,
  Code2,
  Sliders,
  BookOpen,
  Cog,
  Menu,
  X,
  Activity,
  Users,
} from 'lucide-react';

type ActiveView = 'chat' | 'document' | 'graph' | 'chatTuning' | 'ragTuning' | 'categories' | 'routing' | 'structuredKg' | 'documentation' | 'metrics' | 'adminSharedChats';

export interface DockItemConfig {
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  view: ActiveView;
}

const dockItems: DockItemConfig[] = [
  { title: 'Chat', icon: MessageSquare, view: 'chat' },
  { title: 'Database', icon: Database, view: 'document' },
  { title: 'Graph', icon: Network, view: 'graph' },
  { title: 'Categories', icon: FolderTree, view: 'categories' },
  { title: 'Structured KG', icon: Code2, view: 'structuredKg' },
  { title: 'Chat Tuning', icon: Cog, view: 'chatTuning' },
  { title: 'RAG Tuning', icon: Sliders, view: 'ragTuning' },
  { title: 'Metrics', icon: Activity, view: 'metrics' },
  { title: 'Shared Chats', icon: Users, view: 'adminSharedChats' }, // Admin only, filtered below
  { title: 'Documentation', icon: BookOpen, view: 'documentation' },
];

export default function BottomDock() {
  const activeView = useChatStore((state) => state.activeView);
  const setActiveView = useChatStore((state) => state.setActiveView);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Keyboard shortcuts: Cmd/Ctrl+1-9 for navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Check for Cmd (Mac) or Ctrl (Windows/Linux) + number key
      if ((e.metaKey || e.ctrlKey) && e.key >= '1' && e.key <= '9') {
        e.preventDefault();
        const index = parseInt(e.key) - 1;
        if (index < dockItems.length) {
          setActiveView(dockItems[index].view);
          if (isMobile) {
            setIsMenuOpen(false);
          }
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [setActiveView, isMobile]);

  const handleNavigation = (view: ActiveView) => {
    setActiveView(view);
    if (isMobile) {
      setIsMenuOpen(false);
    }
  };

  const user = useChatStore((state) => state.user);

  const visibleDockItems = dockItems.filter(item => {
    if (item.view === 'adminSharedChats' && user?.role !== 'admin') return false;
    return true;
  });

  // Mobile Bottom Sheet
  if (isMobile) {
    return (
      <>
        {/* Mobile Menu Button */}
        <button
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          className="fixed bottom-4 right-4 z-50 w-14 h-14 rounded-full text-white shadow-lg flex items-center justify-center active:scale-95 transition-transform focus:outline-none focus:ring-2 focus:ring-offset-2"
          style={{ backgroundColor: 'var(--accent-primary)', '--tw-ring-color': 'var(--accent-primary)' } as React.CSSProperties}
          aria-label={isMenuOpen ? "Close navigation menu" : "Open navigation menu"}
          aria-expanded={isMenuOpen}
          aria-controls="mobile-navigation-menu"
        >
          {isMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
        </button>

        {/* Bottom Sheet Overlay */}
        {isMenuOpen && (
          <div
            className="fixed inset-0 bg-black/50 z-40 animate-in fade-in duration-200"
            onClick={() => setIsMenuOpen(false)}
          />
        )}

        {/* Bottom Sheet */}
        <div
          id="mobile-navigation-menu"
          role="dialog"
          aria-label="Navigation menu"
          className={`fixed bottom-0 left-0 right-0 z-50 rounded-t-2xl shadow-2xl transition-transform duration-300 backdrop-blur-xl ${isMenuOpen ? 'translate-y-0' : 'translate-y-full'
            }`}
          style={{
            backgroundColor: 'var(--bg-secondary)',
            borderTop: '1px solid var(--border)',
          }}
        >
          <div className="p-6 pb-8">
            {/* Handle bar */}
            <div className="w-12 h-1 rounded-full mx-auto mb-6" style={{ backgroundColor: 'var(--text-tertiary)' }} aria-hidden="true" />

            {/* Grid of navigation items */}
            <div className="grid grid-cols-3 gap-4">
              {visibleDockItems.map((item) => {
                const isActive = activeView === item.view;
                const Icon = item.icon;

                return (
                  <button
                    key={item.view}
                    onClick={() => handleNavigation(item.view)}
                    className="flex flex-col items-center justify-center p-4 rounded-xl transition-all active:scale-95 focus:outline-none focus:ring-2"
                    style={{
                      minHeight: '88px',
                      minWidth: '88px',
                      backgroundColor: isActive ? 'var(--accent-subtle)' : 'var(--bg-tertiary)',
                      color: isActive ? 'var(--accent-primary)' : 'var(--text-secondary)',
                      '--tw-ring-color': 'var(--accent-primary)',
                    } as React.CSSProperties}
                    aria-label={`Navigate to ${item.title}`}
                    aria-current={isActive ? 'page' : undefined}
                  >
                    <Icon className="w-7 h-7 mb-2" />
                    <span className="text-xs font-medium text-center">{item.title}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </>
    );
  }

  // Desktop Dock
  return (
    <Dock
      className="fixed bottom-4 left-1/2 z-50 -translate-x-1/2 bg-white/80 backdrop-blur-md dark:bg-neutral-900/80 border border-gray-200 dark:border-neutral-800 shadow-lg"
      magnification={60}
      distance={120}
      panelHeight={56}
    >
      {visibleDockItems.map((item, index) => {
        const isActive = activeView === item.view;
        const Icon = item.icon;
        const shortcutKey = index + 1;

        return (
          <DockItem
            key={item.view}
            onClick={() => handleNavigation(item.view)}
            className={`cursor-pointer transition-all duration-200 relative ${isActive
              ? 'scale-110 text-[var(--accent-primary)]'
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
              }`}
          >
            <DockLabel className="text-xs">
              {item.title}
              <span className="ml-1 opacity-60 text-[10px]">⌘{shortcutKey}</span>
            </DockLabel>
            <DockIcon>
              <Icon className="w-full h-full" />
              {isActive && (
                <div
                  className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-1 h-1 rounded-full animate-pulse"
                  style={{ backgroundColor: 'var(--accent-primary)', boxShadow: '0 0 8px var(--accent-primary)' }}
                />
              )}
            </DockIcon>
          </DockItem>
        );
      })}
    </Dock>
  );
}
