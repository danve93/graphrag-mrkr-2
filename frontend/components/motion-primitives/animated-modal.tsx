'use client';

import {
    motion,
    AnimatePresence,
    type Variants,
} from 'framer-motion';
import {
    createContext,
    useContext,
    useState,
    useCallback,
    useEffect,
    ReactNode,
} from 'react';
import { createPortal } from 'react-dom';
import { cn } from '../../src/lib/utils';
import { X } from 'lucide-react';

// Context for modal state
interface ModalContextValue {
    isOpen: boolean;
    open: () => void;
    close: () => void;
}

const ModalContext = createContext<ModalContextValue | null>(null);

function useModal() {
    const context = useContext(ModalContext);
    if (!context) {
        throw new Error('Modal components must be used within AnimatedModal');
    }
    return context;
}

// Animation variants
const backdropVariants: Variants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1 },
};

const modalVariants: Variants = {
    hidden: {
        opacity: 0,
        scale: 0.95,
        y: 10,
    },
    visible: {
        opacity: 1,
        scale: 1,
        y: 0,
        transition: {
            type: 'spring',
            stiffness: 400,
            damping: 30,
            mass: 0.8,
        },
    },
    exit: {
        opacity: 0,
        scale: 0.95,
        y: 5,
        transition: {
            duration: 0.15,
            ease: 'easeOut',
        },
    },
};

// Main Modal compound component
interface AnimatedModalProps {
    children: ReactNode;
    defaultOpen?: boolean;
    open?: boolean;
    onOpenChange?: (open: boolean) => void;
}

export function AnimatedModal({
    children,
    defaultOpen = false,
    open: controlledOpen,
    onOpenChange,
}: AnimatedModalProps) {
    const [internalOpen, setInternalOpen] = useState(defaultOpen);

    const isControlled = controlledOpen !== undefined;
    const isOpen = isControlled ? controlledOpen : internalOpen;

    const handleOpen = useCallback(() => {
        if (isControlled) {
            onOpenChange?.(true);
        } else {
            setInternalOpen(true);
        }
    }, [isControlled, onOpenChange]);

    const handleClose = useCallback(() => {
        if (isControlled) {
            onOpenChange?.(false);
        } else {
            setInternalOpen(false);
        }
    }, [isControlled, onOpenChange]);

    // Handle escape key
    useEffect(() => {
        const handleEscape = (e: KeyboardEvent) => {
            if (e.key === 'Escape' && isOpen) {
                handleClose();
            }
        };
        document.addEventListener('keydown', handleEscape);
        return () => document.removeEventListener('keydown', handleEscape);
    }, [isOpen, handleClose]);

    // Prevent body scroll when modal is open
    useEffect(() => {
        if (isOpen) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = '';
        }
        return () => {
            document.body.style.overflow = '';
        };
    }, [isOpen]);

    return (
        <ModalContext.Provider value={{ isOpen, open: handleOpen, close: handleClose }}>
            {children}
        </ModalContext.Provider>
    );
}

// Trigger component
interface ModalTriggerProps {
    children: ReactNode;
    asChild?: boolean;
    className?: string;
}

export function ModalTrigger({ children, asChild, className }: ModalTriggerProps) {
    const { open } = useModal();

    if (asChild && typeof children === 'object' && 'props' in (children as any)) {
        // Clone child with onClick handler
        const child = children as React.ReactElement;
        return (
            <child.type
                {...child.props}
                onClick={(e: React.MouseEvent) => {
                    child.props?.onClick?.(e);
                    open();
                }}
            />
        );
    }

    return (
        <button onClick={open} className={className}>
            {children}
        </button>
    );
}

// Content (modal panel)
interface ModalContentProps {
    children: ReactNode;
    className?: string;
    size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
}

const sizeClasses = {
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-lg',
    xl: 'max-w-xl',
    full: 'max-w-4xl',
};

export function ModalContent({ children, className, size = 'md' }: ModalContentProps) {
    const { isOpen, close } = useModal();
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    if (!mounted) return null;

    return createPortal(
        <AnimatePresence mode="wait">
            {isOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center">
                    {/* Backdrop */}
                    <motion.div
                        variants={backdropVariants}
                        initial="hidden"
                        animate="visible"
                        exit="hidden"
                        transition={{ duration: 0.2 }}
                        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                        onClick={close}
                        aria-hidden="true"
                    />

                    {/* Modal */}
                    <motion.div
                        variants={modalVariants}
                        initial="hidden"
                        animate="visible"
                        exit="exit"
                        role="dialog"
                        aria-modal="true"
                        className={cn(
                            'relative z-10 w-full mx-4 rounded-xl shadow-2xl overflow-hidden',
                            'bg-white dark:bg-neutral-900',
                            'border border-gray-200 dark:border-neutral-800',
                            sizeClasses[size],
                            className
                        )}
                        onClick={(e) => e.stopPropagation()}
                    >
                        {children}
                    </motion.div>
                </div>
            )}
        </AnimatePresence>,
        document.body
    );
}

// Header
interface ModalHeaderProps {
    children: ReactNode;
    className?: string;
    showCloseButton?: boolean;
}

export function ModalHeader({ children, className, showCloseButton = true }: ModalHeaderProps) {
    const { close } = useModal();

    return (
        <div
            className={cn(
                'flex items-center justify-between px-6 py-4',
                'border-b border-gray-200 dark:border-neutral-800',
                className
            )}
        >
            <div className="flex-1">{children}</div>
            {showCloseButton && (
                <button
                    onClick={close}
                    className={cn(
                        'p-1.5 rounded-lg transition-colors',
                        'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200',
                        'hover:bg-gray-100 dark:hover:bg-neutral-800'
                    )}
                    aria-label="Close modal"
                >
                    <X className="w-5 h-5" />
                </button>
            )}
        </div>
    );
}

// Body
interface ModalBodyProps {
    children: ReactNode;
    className?: string;
}

export function ModalBody({ children, className }: ModalBodyProps) {
    return (
        <div className={cn('px-6 py-4', className)}>
            {children}
        </div>
    );
}

// Footer
interface ModalFooterProps {
    children: ReactNode;
    className?: string;
}

export function ModalFooter({ children, className }: ModalFooterProps) {
    return (
        <div
            className={cn(
                'flex items-center justify-end gap-3 px-6 py-4',
                'border-t border-gray-200 dark:border-neutral-800',
                'bg-gray-50 dark:bg-neutral-900/50',
                className
            )}
        >
            {children}
        </div>
    );
}

// Close button (for use inside modal)
interface ModalCloseProps {
    children: ReactNode;
    asChild?: boolean;
    className?: string;
}

export function ModalClose({ children, asChild, className }: ModalCloseProps) {
    const { close } = useModal();

    if (asChild && typeof children === 'object' && 'props' in (children as any)) {
        const child = children as React.ReactElement;
        return (
            <child.type
                {...child.props}
                onClick={(e: React.MouseEvent) => {
                    child.props?.onClick?.(e);
                    close();
                }}
            />
        );
    }

    return (
        <button onClick={close} className={className}>
            {children}
        </button>
    );
}

// Convenience hook for programmatic control
export function useModalControl() {
    return useModal();
}
