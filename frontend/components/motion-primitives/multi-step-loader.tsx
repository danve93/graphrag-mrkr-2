'use client';

import {
    motion,
    AnimatePresence,
} from 'framer-motion';
import { useEffect, useState } from 'react';
import { cn } from '../../src/lib/utils';
import { Check, Circle, Loader2 } from 'lucide-react';

export interface LoaderStep {
    id: string;
    label: string;
    description?: string;
}

interface MultiStepLoaderProps {
    steps: LoaderStep[];
    currentStep: number;
    className?: string;
    showProgress?: boolean;
    onComplete?: () => void;
}

export function MultiStepLoader({
    steps,
    currentStep,
    className,
    showProgress = true,
    onComplete,
}: MultiStepLoaderProps) {
    const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set());
    const progress = ((currentStep + 1) / steps.length) * 100;

    useEffect(() => {
        if (currentStep > 0) {
            setCompletedSteps((prev) => new Set([...prev, currentStep - 1]));
        }
        if (currentStep >= steps.length && onComplete) {
            onComplete();
        }
    }, [currentStep, steps.length, onComplete]);

    const getStepStatus = (index: number): 'pending' | 'active' | 'complete' => {
        if (completedSteps.has(index) || index < currentStep) return 'complete';
        if (index === currentStep) return 'active';
        return 'pending';
    };

    return (
        <div className={cn('w-full max-w-md', className)}>
            {/* Progress bar */}
            {showProgress && (
                <div className="mb-6">
                    <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-2">
                        <span>Progress</span>
                        <span>{Math.round(progress)}%</span>
                    </div>
                    <div className="h-2 bg-gray-200 dark:bg-neutral-800 rounded-full overflow-hidden">
                        <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${progress}%` }}
                            transition={{ duration: 0.5, ease: 'easeOut' }}
                            className="h-full bg-[var(--accent-primary)] rounded-full"
                        />
                    </div>
                </div>
            )}

            {/* Steps */}
            <div className="space-y-3">
                <AnimatePresence mode="wait">
                    {steps.map((step, index) => {
                        const status = getStepStatus(index);

                        return (
                            <motion.div
                                key={step.id}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: index * 0.1 }}
                                className={cn(
                                    'flex items-center gap-4 p-4 rounded-lg border transition-colors',
                                    status === 'active' && 'bg-[var(--accent-subtle)] border-[var(--accent-primary)]',
                                    status === 'complete' && 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
                                    status === 'pending' && 'bg-gray-50 dark:bg-neutral-900 border-gray-200 dark:border-neutral-800'
                                )}
                            >
                                {/* Status icon */}
                                <div
                                    className={cn(
                                        'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0',
                                        status === 'active' && 'bg-[var(--accent-primary)]',
                                        status === 'complete' && 'bg-green-500',
                                        status === 'pending' && 'bg-gray-200 dark:bg-neutral-700'
                                    )}
                                >
                                    {status === 'active' && (
                                        <Loader2 className="w-4 h-4 text-white animate-spin" />
                                    )}
                                    {status === 'complete' && (
                                        <motion.div
                                            initial={{ scale: 0 }}
                                            animate={{ scale: 1 }}
                                            transition={{ type: 'spring', stiffness: 500, damping: 25 }}
                                        >
                                            <Check className="w-4 h-4 text-white" />
                                        </motion.div>
                                    )}
                                    {status === 'pending' && (
                                        <Circle className="w-4 h-4 text-gray-400 dark:text-gray-500" />
                                    )}
                                </div>

                                {/* Content */}
                                <div className="flex-1 min-w-0">
                                    <p
                                        className={cn(
                                            'text-sm font-medium',
                                            status === 'active' && 'text-[var(--accent-primary)]',
                                            status === 'complete' && 'text-green-700 dark:text-green-400',
                                            status === 'pending' && 'text-gray-500 dark:text-gray-400'
                                        )}
                                    >
                                        {step.label}
                                    </p>
                                    {step.description && (
                                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                                            {step.description}
                                        </p>
                                    )}
                                </div>

                                {/* Step number */}
                                <span className="text-xs text-gray-400 dark:text-gray-500 flex-shrink-0">
                                    {index + 1}/{steps.length}
                                </span>
                            </motion.div>
                        );
                    })}
                </AnimatePresence>
            </div>
        </div>
    );
}

// Inline loader for simpler use cases
interface InlineLoaderProps {
    steps: string[];
    currentStep: number;
    className?: string;
}

export function InlineLoader({ steps, currentStep, className }: InlineLoaderProps) {
    return (
        <div className={cn('flex items-center gap-2', className)}>
            <Loader2 className="w-4 h-4 animate-spin text-[var(--accent-primary)]" />
            <AnimatePresence mode="wait">
                <motion.span
                    key={currentStep}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.2 }}
                    className="text-sm text-gray-700 dark:text-gray-300"
                >
                    {steps[currentStep] || 'Processing...'}
                </motion.span>
            </AnimatePresence>
            <span className="text-xs text-gray-400">
                ({currentStep + 1}/{steps.length})
            </span>
        </div>
    );
}

export default MultiStepLoader;
