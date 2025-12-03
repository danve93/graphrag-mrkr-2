'use client';

export default function SettingsView() {
  return (
    <div className="h-full flex items-center justify-center bg-secondary-50 dark:bg-secondary-900 pb-28">
      <div className="text-center">
        <h1 className="text-2xl font-bold mb-4 text-secondary-900 dark:text-secondary-100">
          Settings
        </h1>
        <p className="text-secondary-600 dark:text-secondary-400">
          Application settings and configuration
        </p>
        <p className="text-sm text-secondary-500 dark:text-secondary-500 mt-2">
          Coming in Phase 0, M0.5
        </p>
      </div>
    </div>
  );
}
