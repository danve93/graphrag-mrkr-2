'use client';

import { useState, useEffect } from 'react';
import { BookOpen, ExternalLink } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function DocumentationView() {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);

  // Listen for file selection events from the sidebar
  useEffect(() => {
    const handleFileSelect = (event: CustomEvent<string>) => {
      loadFile(event.detail);
    };

    window.addEventListener('documentation-file-select', handleFileSelect as EventListener);
    return () => {
      window.removeEventListener('documentation-file-select', handleFileSelect as EventListener);
    };
  }, []);

  const loadFile = async (path: string) => {
    setIsLoading(true);
    setSelectedFile(path);
    // Broadcast the selected file to the sidebar
    window.dispatchEvent(new CustomEvent('documentation-file-changed', { detail: path }));
    try {
      const response = await fetch(`/api/documentation/${path}`);
      if (response.ok) {
        const content = await response.text();
        setFileContent(content);
      } else {
        setFileContent(`# Documentation Not Available\n\nThe file \`${path}\` could not be loaded.\n\nIn production, documentation files should be served via the API endpoint.`);
      }
    } catch (error) {
      setFileContent(`# Error Loading Documentation\n\nFailed to load \`${path}\`.\n\n\`\`\`\n${error}\n\`\`\``);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-full bg-secondary-50 dark:bg-secondary-900 pb-28 overflow-y-auto">
      {selectedFile ? (
        <div className="max-w-4xl mx-auto p-8">
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#f27a03]"></div>
            </div>
          ) : (
            <article className="prose prose-neutral dark:prose-invert max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  h1: ({ children }) => (
                    <h1 className="text-3xl font-bold mb-4 text-secondary-900 dark:text-secondary-100 border-b border-gray-200 dark:border-neutral-800 pb-3">
                      {children}
                    </h1>
                  ),
                  h2: ({ children }) => (
                    <h2 className="text-2xl font-bold mt-8 mb-3 text-secondary-900 dark:text-secondary-100">
                      {children}
                    </h2>
                  ),
                  h3: ({ children }) => (
                    <h3 className="text-xl font-bold mt-6 mb-2 text-secondary-900 dark:text-secondary-100">
                      {children}
                    </h3>
                  ),
                  code: ({ inline, children, ...props }: any) => 
                    inline ? (
                      <code className="px-1.5 py-0.5 rounded bg-gray-100 dark:bg-neutral-800 text-[#f27a03] text-sm font-mono" {...props}>
                        {children}
                      </code>
                    ) : (
                      <code className="block p-4 rounded-lg bg-gray-100 dark:bg-neutral-800 text-sm font-mono overflow-x-auto" {...props}>
                        {children}
                      </code>
                    ),
                  pre: ({ children }) => (
                    <pre className="my-4 rounded-lg overflow-hidden">{children}</pre>
                  ),
                  a: ({ href, children }) => {
                    // Check if link is to another documentation file
                    const isDocLink = href?.endsWith('.md');
                    
                    if (isDocLink && href) {
                      // Extract the documentation path - must be exact path
                      let docPath = href;
                      
                      // Remove .md extension (backend will add it back)
                      docPath = docPath.replace(/\.md$/, '');
                      
                      // Remove leading ./ or / if present (but keep directory structure)
                      docPath = docPath.replace(/^\.\//, '');
                      docPath = docPath.replace(/^\//, '');
                      
                      return (
                        <button
                          onClick={() => loadFile(docPath)}
                          className="text-[#f27a03] hover:underline cursor-pointer"
                        >
                          {children}
                        </button>
                      );
                    }
                    
                    // External links open in new tab
                    return (
                      <a
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[#f27a03] hover:underline inline-flex items-center gap-1"
                      >
                        {children}
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    );
                  },
                  table: ({ children }) => (
                    <div className="overflow-x-auto my-4">
                      <table className="min-w-full divide-y divide-gray-200 dark:divide-neutral-800">
                        {children}
                      </table>
                    </div>
                  ),
                  th: ({ children }) => (
                    <th className="px-4 py-2 bg-gray-50 dark:bg-neutral-800 text-left text-xs font-semibold text-secondary-700 dark:text-secondary-300 uppercase tracking-wider">
                      {children}
                    </th>
                  ),
                  td: ({ children }) => (
                    <td className="px-4 py-2 text-sm text-secondary-600 dark:text-secondary-400 border-t border-gray-200 dark:border-neutral-800">
                      {children}
                    </td>
                  ),
                }}
              >
                {fileContent}
              </ReactMarkdown>
            </article>
          )}
        </div>
      ) : (
        <div className="flex items-center justify-center h-full">
          <div className="text-center max-w-md">
            <div className="w-16 h-16 rounded-full bg-orange-50 dark:bg-orange-900/20 flex items-center justify-center mx-auto mb-4">
              <BookOpen className="w-8 h-8 text-[#f27a03]" />
            </div>
            <h3 className="text-xl font-bold mb-2 text-secondary-900 dark:text-secondary-100">
              Welcome to Documentation
            </h3>
            <p className="text-secondary-600 dark:text-secondary-400">
              Select a document from the sidebar to get started with Amber&apos;s comprehensive documentation.
            </p>
            <div className="mt-6 p-4 bg-gray-50 dark:bg-neutral-800 rounded-lg text-left">
              <p className="text-sm text-secondary-600 dark:text-secondary-400 mb-2">
                <strong>Quick Links:</strong>
              </p>
              <ul className="text-sm space-y-1 text-secondary-600 dark:text-secondary-400">
                <li>• Getting Started Guide</li>
                <li>• Features Overview</li>
                <li>• API Reference</li>
                <li>• Configuration Options</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
