// Motion primitives - Aceternity UI style components with framer-motion
// These components provide premium micro-interactions and animations

export { AnimatedTooltip } from './animated-tooltip';
export type { AnimatedTooltipProps } from './animated-tooltip';

export {
    AnimatedModal,
    ModalTrigger,
    ModalContent,
    ModalHeader,
    ModalBody,
    ModalFooter,
    ModalClose,
    useModalControl,
} from './animated-modal';

export { StatefulButton, useStatefulButton } from './stateful-button';
export type { ButtonState } from './stateful-button';

export { FileUpload } from './file-upload';

export { MultiStepLoader, InlineLoader } from './multi-step-loader';
export type { LoaderStep } from './multi-step-loader';

// Re-export dock component
export { Dock, DockIcon, DockItem, DockLabel } from './dock';
