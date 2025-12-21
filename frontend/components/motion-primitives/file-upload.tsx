'use client';

import {
    motion,
    AnimatePresence,
} from 'framer-motion';
import { useState, useRef, ChangeEvent, DragEvent, ReactNode } from 'react';
import { cn } from '../../src/lib/utils';
import { CloudUpload, FileText, X, Check } from 'lucide-react';

interface FileUploadProps {
    onFilesSelected: (files: File[]) => void;
    accept?: string;
    multiple?: boolean;
    maxFiles?: number;
    maxSizeMB?: number;
    className?: string;
    children?: ReactNode;
    disabled?: boolean;
}

interface UploadedFile {
    file: File;
    id: string;
    progress: number;
    status: 'pending' | 'uploading' | 'complete' | 'error';
}

export function FileUpload({
    onFilesSelected,
    accept = '*',
    multiple = true,
    maxFiles = 10,
    maxSizeMB = 50,
    className,
    children,
    disabled = false,
}: FileUploadProps) {
    const [isDragging, setIsDragging] = useState(false);
    const [files, setFiles] = useState<UploadedFile[]>([]);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        if (!disabled) setIsDragging(true);
    };

    const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    };

    const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
    };

    const validateFiles = (fileList: FileList): File[] => {
        const validFiles: File[] = [];
        const maxSizeBytes = maxSizeMB * 1024 * 1024;

        for (let i = 0; i < fileList.length && validFiles.length < maxFiles; i++) {
            const file = fileList[i];
            if (file.size <= maxSizeBytes) {
                validFiles.push(file);
            }
        }

        return validFiles;
    };

    const handleDrop = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);

        if (disabled) return;

        const droppedFiles = validateFiles(e.dataTransfer.files);
        if (droppedFiles.length > 0) {
            addFiles(droppedFiles);
        }
    };

    const handleFileInput = (e: ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            const selectedFiles = validateFiles(e.target.files);
            if (selectedFiles.length > 0) {
                addFiles(selectedFiles);
            }
        }
        // Reset input
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const addFiles = (newFiles: File[]) => {
        const uploadedFiles: UploadedFile[] = newFiles.map((file) => ({
            file,
            id: `${file.name}-${Date.now()}-${Math.random()}`,
            progress: 0,
            status: 'pending',
        }));

        setFiles((prev) => [...prev, ...uploadedFiles].slice(0, maxFiles));
        onFilesSelected(newFiles);
    };

    const removeFile = (id: string) => {
        setFiles((prev) => prev.filter((f) => f.id !== id));
    };

    const openFilePicker = () => {
        if (!disabled && fileInputRef.current) {
            fileInputRef.current.click();
        }
    };

    return (
        <div className={cn('w-full', className)}>
            {/* Drop zone */}
            <motion.div
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onClick={openFilePicker}
                animate={{
                    scale: isDragging ? 1.02 : 1,
                    borderColor: isDragging ? 'var(--accent-primary)' : undefined,
                }}
                className={cn(
                    'relative cursor-pointer rounded-xl border-2 border-dashed p-8',
                    'transition-colors duration-200',
                    'border-gray-300 dark:border-neutral-700',
                    'hover:border-gray-400 dark:hover:border-neutral-600',
                    'bg-gray-50/50 dark:bg-neutral-900/50',
                    isDragging && 'border-[var(--accent-primary)] bg-[var(--accent-subtle)]',
                    disabled && 'opacity-50 cursor-not-allowed'
                )}
            >
                {/* Grid pattern background */}
                <div
                    className="absolute inset-0 pointer-events-none opacity-30"
                    style={{
                        backgroundImage: `radial-gradient(circle, var(--accent-primary) 1px, transparent 1px)`,
                        backgroundSize: '24px 24px',
                    }}
                />

                {/* Content */}
                <div className="relative flex flex-col items-center gap-4 text-center">
                    <motion.div
                        animate={{
                            y: isDragging ? -8 : 0,
                            scale: isDragging ? 1.1 : 1,
                        }}
                        className={cn(
                            'w-16 h-16 rounded-full flex items-center justify-center',
                            'bg-gray-100 dark:bg-neutral-800',
                            isDragging && 'bg-[var(--accent-subtle)]'
                        )}
                    >
                        <CloudUpload
                            className={cn(
                                'w-8 h-8',
                                isDragging ? 'text-[var(--accent-primary)]' : 'text-gray-400 dark:text-gray-500'
                            )}
                        />
                    </motion.div>

                    {children || (
                        <>
                            <div>
                                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                                    <span className="text-[var(--accent-primary)]">Click to upload</span> or drag and drop
                                </p>
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                    Max {maxFiles} files, up to {maxSizeMB}MB each
                                </p>
                            </div>
                        </>
                    )}
                </div>

                <input
                    ref={fileInputRef}
                    type="file"
                    accept={accept}
                    multiple={multiple}
                    onChange={handleFileInput}
                    className="hidden"
                    disabled={disabled}
                />
            </motion.div>

            {/* File list */}
            <AnimatePresence>
                {files.length > 0 && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-4 space-y-2"
                    >
                        {files.map((uploadedFile) => (
                            <motion.div
                                key={uploadedFile.id}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: 20 }}
                                className={cn(
                                    'flex items-center gap-3 p-3 rounded-lg',
                                    'bg-gray-50 dark:bg-neutral-800/50',
                                    'border border-gray-200 dark:border-neutral-700'
                                )}
                            >
                                <div className="w-10 h-10 rounded-lg bg-gray-100 dark:bg-neutral-800 flex items-center justify-center">
                                    <FileText className="w-5 h-5 text-gray-500" />
                                </div>
                                <div className="flex-1 min-w-0">
                                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300 truncate">
                                        {uploadedFile.file.name}
                                    </p>
                                    <p className="text-xs text-gray-500 dark:text-gray-400">
                                        {(uploadedFile.file.size / 1024 / 1024).toFixed(2)} MB
                                    </p>
                                </div>
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        removeFile(uploadedFile.id);
                                    }}
                                    className="p-1.5 rounded-lg hover:bg-gray-200 dark:hover:bg-neutral-700 transition-colors"
                                >
                                    <X className="w-4 h-4 text-gray-500" />
                                </button>
                            </motion.div>
                        ))}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

export default FileUpload;
