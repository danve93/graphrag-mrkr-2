'use client';

import {
    motion,
    useMotionValue,
    useSpring,
    AnimatePresence,
} from 'framer-motion';
import {
    useState,
    useRef,
    useEffect,
    useCallback,
    isValidElement,
    cloneElement,
    ReactNode,
    ReactElement,
} from 'react';
import { createPortal } from 'react-dom';
import { cn } from '../../src/lib/utils';

export interface AnimatedTooltipProps {
    children: ReactNode;
    content: string;
    className?: string;
    delayShow?: number;
    delayHide?: number;
}

export function AnimatedTooltip({
    children,
    content,
    className,
    delayShow = 100,
    delayHide = 0,
}: AnimatedTooltipProps) {
    const [isVisible, setIsVisible] = useState(false);
    const [mounted, setMounted] = useState(false);
    const triggerRef = useRef<HTMLElement | null>(null);
    const tooltipRef = useRef<HTMLDivElement | null>(null);
    const lastPointerRef = useRef<{ x: number; y: number } | null>(null);
    const showTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const hideTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    // Mouse position for the tooltip to follow
    const mouseX = useMotionValue(0);
    const mouseY = useMotionValue(0);

    // Spring animation tuned to be snappy with minimal bounce.
    const springConfig = { damping: 50, stiffness: 700, mass: 0.25 };
    const x = useSpring(mouseX, springConfig);
    const y = useSpring(mouseY, springConfig);

    useEffect(() => {
        setMounted(true);
        return () => {
            if (showTimeoutRef.current) clearTimeout(showTimeoutRef.current);
            if (hideTimeoutRef.current) clearTimeout(hideTimeoutRef.current);
        };
    }, []);

    const clamp = (value: number, min: number, max: number) =>
        Math.min(Math.max(value, min), max);

    const getTooltipSize = useCallback(() => {
        const rect = tooltipRef.current?.getBoundingClientRect();
        if (rect) return { width: rect.width, height: rect.height };
        const viewportWidth = typeof window !== 'undefined' ? window.innerWidth : 0;
        const maxWidth = viewportWidth ? Math.min(320, viewportWidth - 16) : 320;
        const estimatedWidth = Math.min(content.length * 7 + 24, maxWidth);
        return { width: estimatedWidth, height: 32 };
    }, [content]);

    const updateMousePosition = useCallback((clientX: number, clientY: number) => {
        if (!triggerRef.current || typeof window === 'undefined') return;
        lastPointerRef.current = { x: clientX, y: clientY };

        const rect = triggerRef.current.getBoundingClientRect();
        const { width, height } = getTooltipSize();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        const baseLeft = rect.left + rect.width / 2;
        const baseTop = rect.top - 8;

        const rawX = clientX - rect.left - rect.width / 2;
        const rawY = clientY - rect.top - rect.height - 20;

        const desiredLeft = baseLeft + rawX;
        const desiredTop = baseTop + rawY;

        const minLeft = 8;
        const maxLeft = Math.max(minLeft, viewportWidth - width - 8);
        const minTop = 8;
        const maxTop = Math.max(minTop, viewportHeight - height - 8);

        const clampedLeft = clamp(desiredLeft, minLeft, maxLeft);
        const clampedTop = clamp(desiredTop, minTop, maxTop);

        mouseX.set(clampedLeft - baseLeft);
        mouseY.set(clampedTop - baseTop);
    }, [getTooltipSize, mouseX, mouseY]);

    useEffect(() => {
        if (!isVisible || !lastPointerRef.current) return;
        const { x: clientX, y: clientY } = lastPointerRef.current;
        const frame = requestAnimationFrame(() => {
            updateMousePosition(clientX, clientY);
        });
        return () => cancelAnimationFrame(frame);
    }, [isVisible, updateMousePosition]);

    const handleMouseMove = (e: React.MouseEvent) => {
        updateMousePosition(e.clientX, e.clientY);
    };

    const handleMouseEnter = (e: React.MouseEvent) => {
        updateMousePosition(e.clientX, e.clientY);
        if (hideTimeoutRef.current) {
            clearTimeout(hideTimeoutRef.current);
            hideTimeoutRef.current = null;
        }
        showTimeoutRef.current = setTimeout(() => {
            setIsVisible(true);
        }, delayShow);
    };

    const handleMouseLeave = () => {
        if (showTimeoutRef.current) {
            clearTimeout(showTimeoutRef.current);
            showTimeoutRef.current = null;
        }
        hideTimeoutRef.current = setTimeout(() => {
            setIsVisible(false);
        }, delayHide);
    };

    // Calculate tooltip position relative to trigger
    const getTooltipPosition = () => {
        if (!triggerRef.current) return { top: 0, left: 0 };
        const rect = triggerRef.current.getBoundingClientRect();
        return {
            top: rect.top - 8,
            left: rect.left + rect.width / 2,
        };
    };

    const tooltipContent = mounted ? (
        <AnimatePresence>
            {isVisible && (
                <motion.div
                    ref={tooltipRef}
                    initial={{ opacity: 0, y: 4, scale: 0.99 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 2, scale: 0.99 }}
                    transition={{
                        type: 'tween',
                        duration: 0.08,
                        ease: 'easeOut',
                    }}
                    style={{
                        position: 'fixed',
                        top: getTooltipPosition().top,
                        left: getTooltipPosition().left,
                        x,
                        y,
                        zIndex: 9999,
                        pointerEvents: 'none',
                        transformOrigin: 'bottom center',
                        maxWidth: 'min(320px, calc(100vw - 16px))',
                    }}
                    className={cn(
                        'px-3 py-1.5 rounded-lg text-sm font-medium whitespace-normal break-words',
                        'bg-gray-900 dark:bg-white text-white dark:text-gray-900',
                        'shadow-lg shadow-black/20',
                        className
                    )}
                >
                    {content}
                    {/* Arrow pointing down */}
                    <div
                        className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0"
                        style={{
                            borderLeft: '6px solid transparent',
                            borderRight: '6px solid transparent',
                            borderTop: '6px solid rgb(17, 24, 39)',
                        }}
                    />
                </motion.div>
            )}
        </AnimatePresence>
    ) : null;

    // Clone child element with handlers attached
    if (isValidElement(children)) {
        const child = children as ReactElement<any>;
        const existingOnEnter = child.props?.onMouseEnter;
        const existingOnLeave = child.props?.onMouseLeave;
        const existingOnMove = child.props?.onMouseMove;

        const cloned = cloneElement(child, {
            ref: (node: HTMLElement | null) => {
                triggerRef.current = node;
                // Handle original ref
                const origRef = (child as any).ref;
                if (typeof origRef === 'function') origRef(node);
                else if (origRef && typeof origRef === 'object') origRef.current = node;
            },
            onMouseEnter: (e: React.MouseEvent) => {
                handleMouseEnter(e);
                if (existingOnEnter) existingOnEnter(e);
            },
            onMouseLeave: (e: React.MouseEvent) => {
                handleMouseLeave();
                if (existingOnLeave) existingOnLeave(e);
            },
            onMouseMove: (e: React.MouseEvent) => {
                handleMouseMove(e);
                if (existingOnMove) existingOnMove(e);
            },
        });

        return (
            <>
                {cloned}
                {mounted && createPortal(tooltipContent, document.body)}
            </>
        );
    }

    // Fallback: wrap children in a span
    return (
        <>
            <span
                ref={triggerRef as any}
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
                onMouseMove={handleMouseMove}
                style={{ position: 'relative', display: 'inline-block' }}
            >
                {children}
            </span>
            {mounted && createPortal(tooltipContent, document.body)}
        </>
    );
}

export default AnimatedTooltip;
