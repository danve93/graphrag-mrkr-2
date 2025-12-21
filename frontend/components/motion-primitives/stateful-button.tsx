'use client';

import {
    motion,
    AnimatePresence,
} from 'framer-motion';
import { useState, useEffect, ReactNode, ButtonHTMLAttributes } from 'react';
import { cn } from '../../src/lib/utils';
import { Loader2, Check, AlertCircle } from 'lucide-react';

export type ButtonState = 'idle' | 'loading' | 'success' | 'error';

interface StatefulButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'> {
    children: ReactNode;
    state?: ButtonState;
    onStateChange?: (state: ButtonState) => void;
    loadingText?: string;
    successText?: string;
    errorText?: string;
    successDuration?: number;
    errorDuration?: number;
    variant?: 'primary' | 'secondary' | 'ghost';
    size?: 'sm' | 'md' | 'lg';
    className?: string;
}

const variantClasses = {
    primary: 'bg-[var(--accent-primary)] text-white hover:bg-[var(--accent-hover)] active:bg-[var(--accent-active)]',
    secondary: 'bg-gray-100 dark:bg-neutral-800 text-gray-900 dark:text-gray-100 hover:bg-gray-200 dark:hover:bg-neutral-700',
    ghost: 'bg-transparent text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-neutral-800',
};

const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm gap-1.5',
    md: 'px-4 py-2 text-sm gap-2',
    lg: 'px-6 py-3 text-base gap-2.5',
};

const iconSizes = {
    sm: 'w-3.5 h-3.5',
    md: 'w-4 h-4',
    lg: 'w-5 h-5',
};

export function StatefulButton({
    children,
    state: externalState,
    onStateChange,
    loadingText,
    successText = 'Done!',
    errorText = 'Error',
    successDuration = 2000,
    errorDuration = 2000,
    variant = 'primary',
    size = 'md',
    className,
    disabled,
    ...props
}: StatefulButtonProps) {
    const [internalState, setInternalState] = useState<ButtonState>('idle');

    const isControlled = externalState !== undefined;
    const state = isControlled ? externalState : internalState;

    // Auto-reset from success/error to idle
    useEffect(() => {
        if (state === 'success') {
            const timer = setTimeout(() => {
                if (isControlled) {
                    onStateChange?.('idle');
                } else {
                    setInternalState('idle');
                }
            }, successDuration);
            return () => clearTimeout(timer);
        }
        if (state === 'error') {
            const timer = setTimeout(() => {
                if (isControlled) {
                    onStateChange?.('idle');
                } else {
                    setInternalState('idle');
                }
            }, errorDuration);
            return () => clearTimeout(timer);
        }
    }, [state, successDuration, errorDuration, isControlled, onStateChange]);

    const isDisabled = disabled || state === 'loading';

    const getContent = () => {
        switch (state) {
            case 'loading':
                return (
                    <>
                        <Loader2 className={cn(iconSizes[size], 'animate-spin')} />
                        <span>{loadingText || children}</span>
                    </>
                );
            case 'success':
                return (
                    <>
                        <Check className={iconSizes[size]} />
                        <span>{successText}</span>
                    </>
                );
            case 'error':
                return (
                    <>
                        <AlertCircle className={iconSizes[size]} />
                        <span>{errorText}</span>
                    </>
                );
            default:
                return children;
        }
    };

    const getStateClasses = () => {
        if (state === 'success') {
            return 'bg-green-500 hover:bg-green-500 text-white';
        }
        if (state === 'error') {
            return 'bg-red-500 hover:bg-red-500 text-white';
        }
        return variantClasses[variant];
    };

    return (
        <motion.div
            whileTap={{ scale: isDisabled ? 1 : 0.98 }}
            className="inline-block"
        >
            <button
                className={cn(
                    'relative inline-flex items-center justify-center font-medium rounded-lg',
                    'transition-colors duration-200',
                    'disabled:opacity-50 disabled:cursor-not-allowed',
                    sizeClasses[size],
                    getStateClasses(),
                    className
                )}
                disabled={isDisabled}
                {...props}
            >
                <AnimatePresence mode="wait" initial={false}>
                    <motion.span
                        key={state}
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -5 }}
                        transition={{ duration: 0.15 }}
                        className="flex items-center gap-inherit"
                        style={{ gap: 'inherit' }}
                    >
                        {getContent()}
                    </motion.span>
                </AnimatePresence>
            </button>
        </motion.div>
    );
}

// Hook for easy state management
export function useStatefulButton() {
    const [state, setState] = useState<ButtonState>('idle');

    const execute = async <T,>(promise: Promise<T>): Promise<T> => {
        setState('loading');
        try {
            const result = await promise;
            setState('success');
            return result;
        } catch (error) {
            setState('error');
            throw error;
        }
    };

    return { state, setState, execute };
}

export default StatefulButton;
