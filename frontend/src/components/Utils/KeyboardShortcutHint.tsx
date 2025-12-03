'use client';

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function KeyboardShortcutHint() {
  const [show, setShow] = useState(false);
  const [hasInteracted, setHasInteracted] = useState(false);

  useEffect(() => {
    // Check if user has seen the hint before
    const seen = localStorage.getItem('keyboard-shortcut-hint-seen');
    if (seen) {
      setHasInteracted(true);
      return;
    }

    // Show hint after 3 seconds
    const timer = setTimeout(() => {
      if (!hasInteracted) {
        setShow(true);
      }
    }, 3000);

    // Hide after 8 seconds
    const hideTimer = setTimeout(() => {
      setShow(false);
      localStorage.setItem('keyboard-shortcut-hint-seen', 'true');
    }, 11000);

    return () => {
      clearTimeout(timer);
      clearTimeout(hideTimer);
    };
  }, [hasInteracted]);

  const isMac = typeof window !== 'undefined' && navigator.platform.includes('Mac');

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
          className="fixed bottom-24 left-1/2 transform -translate-x-1/2 z-40 bg-gray-900 dark:bg-gray-800 text-white px-4 py-3 rounded-lg shadow-xl border border-gray-700"
          style={{ maxWidth: '90vw', width: 'auto' }}
        >
          <button
            onClick={() => {
              setShow(false);
              localStorage.setItem('keyboard-shortcut-hint-seen', 'true');
            }}
            className="absolute -top-2 -right-2 w-6 h-6 bg-gray-700 hover:bg-gray-600 rounded-full flex items-center justify-center text-xs transition-colors"
            aria-label="Dismiss hint"
          >
            ×
          </button>
          <div className="flex items-center gap-2 text-sm">
            <span>💡</span>
            <span>
              Press <kbd className="px-2 py-1 bg-gray-800 dark:bg-gray-700 rounded border border-gray-600 text-xs font-mono">{isMac ? '⌘' : 'Ctrl'}</kbd> + 
              <kbd className="px-2 py-1 bg-gray-800 dark:bg-gray-700 rounded border border-gray-600 text-xs font-mono ml-1">1-9</kbd> 
              <span className="ml-1">to navigate quickly</span>
            </span>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
