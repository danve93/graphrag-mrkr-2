'use client';

import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

interface SkeletonProps {
    className?: string;
    variant?: 'text' | 'circular' | 'rectangular' | 'rounded';
    width?: string | number;
    height?: string | number;
    animation?: 'pulse' | 'wave' | 'none';
}

export function Skeleton({
    className,
    variant = 'text',
    width,
    height,
    animation = 'pulse',
}: SkeletonProps) {
    const variantClasses = {
        text: 'rounded',
        circular: 'rounded-full',
        rectangular: 'rounded-none',
        rounded: 'rounded-lg',
    };

    const animationClasses = {
        pulse: 'animate-pulse',
        wave: 'skeleton-wave',
        none: '',
    };

    const defaultHeight = variant === 'text' ? '1em' : variant === 'circular' ? width : '100%';

    return (
        <div
            className={cn(
                'bg-gray-200 dark:bg-neutral-800',
                variantClasses[variant],
                animationClasses[animation],
                className
            )}
            style={{ width, height: height || defaultHeight }}
        />
    );
}

// Common skeleton patterns
interface SkeletonTextProps {
    lines?: number;
    className?: string;
    lastLineWidth?: string;
}

export function SkeletonText({ lines = 3, className, lastLineWidth = '60%' }: SkeletonTextProps) {
    return (
        <div className={cn('space-y-2', className)}>
            {Array.from({ length: lines }).map((_, i) => (
                <Skeleton
                    key={i}
                    variant="text"
                    height="0.875rem"
                    width={i === lines - 1 ? lastLineWidth : '100%'}
                />
            ))}
        </div>
    );
}

interface SkeletonAvatarProps {
    size?: number;
    className?: string;
}

export function SkeletonAvatar({ size = 40, className }: SkeletonAvatarProps) {
    return (
        <Skeleton
            variant="circular"
            width={size}
            height={size}
            className={className}
        />
    );
}

// Message skeleton for chat
export function SkeletonMessage({ isUser = false }: { isUser?: boolean }) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={cn('flex', isUser ? 'justify-end' : 'justify-start')}
        >
            <div
                className={cn(
                    'max-w-[80%] p-4 rounded-2xl',
                    isUser
                        ? 'bg-[var(--accent-subtle)] rounded-br-none'
                        : 'bg-gray-100 dark:bg-neutral-800 rounded-bl-none'
                )}
            >
                <SkeletonText lines={isUser ? 1 : 3} lastLineWidth={isUser ? '100%' : '75%'} />
            </div>
        </motion.div>
    );
}

// Card skeleton
export function SkeletonCard({ className }: { className?: string }) {
    return (
        <div
            className={cn(
                'p-4 rounded-lg border border-gray-200 dark:border-neutral-800',
                'bg-white dark:bg-neutral-900',
                className
            )}
        >
            <div className="flex items-start gap-3">
                <SkeletonAvatar size={48} />
                <div className="flex-1 space-y-2">
                    <Skeleton variant="text" height="1rem" width="40%" />
                    <Skeleton variant="text" height="0.75rem" width="25%" />
                </div>
            </div>
            <div className="mt-4">
                <SkeletonText lines={3} />
            </div>
        </div>
    );
}

// Document list skeleton
export function SkeletonDocumentList({ count = 5 }: { count?: number }) {
    return (
        <div className="space-y-3">
            {Array.from({ length: count }).map((_, i) => (
                <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.05 }}
                    className="flex items-center gap-3 p-3 rounded-lg border border-gray-200 dark:border-neutral-800"
                >
                    <Skeleton variant="rounded" width={40} height={40} />
                    <div className="flex-1 space-y-2">
                        <Skeleton variant="text" height="0.875rem" width="60%" />
                        <Skeleton variant="text" height="0.75rem" width="40%" />
                    </div>
                    <Skeleton variant="rounded" width={60} height={24} />
                </motion.div>
            ))}
        </div>
    );
}

export default Skeleton;
