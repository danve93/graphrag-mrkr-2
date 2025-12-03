'use client';

export default function GenericSidebarContent({ title, description }: { title: string; description: string }) {
  return (
    <div className="flex-1 p-6">
      <h2 className="text-lg font-semibold mb-2 text-secondary-900 dark:text-secondary-100">
        {title}
      </h2>
      <p className="text-sm text-secondary-600 dark:text-secondary-400">
        {description}
      </p>
    </div>
  );
}
