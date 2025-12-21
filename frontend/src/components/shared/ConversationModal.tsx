import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { X, User, Bot, AlertTriangle, Calendar, Clock, MessageSquare, Download } from 'lucide-react';

interface ConversationModalProps {
    isOpen: boolean;
    onClose: () => void;
    sessionId: string | null;
}

interface Message {
    role: string;
    content: string;
    timestamp: string;
    sources?: any[];
}

interface ConversationData {
    session_id: string;
    messages: Message[];
    created_at: string;
    updated_at: string;
}

export default function ConversationModal({ isOpen, onClose, sessionId }: ConversationModalProps) {
    const [conversation, setConversation] = useState<ConversationData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (isOpen && sessionId) {
            fetchConversation(sessionId);
        } else {
            setConversation(null);
            setError(null);
        }
    }, [isOpen, sessionId]);

    const fetchConversation = async (id: string) => {
        setLoading(true);
        setError(null);
        try {
            const res = await fetch(`/api/history/${id}`);
            if (!res.ok) {
                if (res.status === 403) throw new Error("You do not have permission to view this conversation.");
                if (res.status === 404) throw new Error("Conversation not found.");
                throw new Error("Failed to load conversation.");
            }
            const data = await res.json();
            setConversation(data);
        } catch (err: any) {
            setError(err.message || 'An error occurred');
        } finally {
            setLoading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
            <div className="bg-white dark:bg-neutral-900 rounded-xl shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col overflow-hidden border border-gray-200 dark:border-gray-800 animate-in fade-in zoom-in duration-200">

                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-800 bg-gray-50/50 dark:bg-neutral-900/50">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-primary-500/10 dark:bg-primary-500/20 rounded-lg text-primary-500">
                            <MessageSquare size={20} />
                        </div>
                        <div>
                            <h3 className="font-semibold text-gray-900 dark:text-gray-100">Conversation Details</h3>
                            <p className="text-xs text-gray-500 font-mono">{sessionId}</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-gray-200 dark:hover:bg-neutral-800 rounded-lg transition-colors text-gray-500"
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-6 bg-white dark:bg-neutral-900">
                    {loading ? (
                        <div className="flex justify-center py-12">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
                        </div>
                    ) : error ? (
                        <div className="flex flex-col items-center justify-center py-12 text-center text-red-500">
                            <AlertTriangle size={32} className="mb-2" />
                            <p>{error}</p>
                        </div>
                    ) : conversation ? (
                        <div className="space-y-6">
                            {/* Metadata */}
                            <div className="flex flex-wrap gap-4 text-xs text-gray-500 pb-4 border-b border-gray-100 dark:border-gray-800">
                                <div className="flex items-center gap-1.5">
                                    <Calendar size={14} />
                                    <span>Created: {new Date(conversation.created_at || Date.now()).toLocaleString()}</span>
                                </div>
                                <div className="flex items-center gap-1.5">
                                    <Clock size={14} />
                                    <span>Updated: {new Date(conversation.updated_at || Date.now()).toLocaleString()}</span>
                                </div>
                                <div className="flex items-center gap-1.5 ml-auto">
                                    <span className="px-2 py-0.5 rounded-full bg-gray-100 dark:bg-neutral-800 text-gray-600 dark:text-gray-400 font-medium">
                                        {conversation.messages.length} messages
                                    </span>
                                </div>
                            </div>

                            {/* Messages */}
                            <div className="space-y-6">
                                {conversation.messages.map((msg, idx) => (
                                    <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>

                                        {/* Avatar (Assistant) */}
                                        {msg.role !== 'user' && (
                                            <div className="w-8 h-8 rounded-full bg-gray-100 dark:bg-neutral-800 flex items-center justify-center text-gray-500 dark:text-gray-400 flex-shrink-0 mt-1">
                                                <Bot size={16} />
                                            </div>
                                        )}

                                        <div className={`max-w-[80%] rounded-2xl px-5 py-3.5 shadow-sm ${msg.role === 'user'
                                            ? 'bg-primary-500 text-white rounded-tr-sm'
                                            : 'bg-gray-100 dark:bg-neutral-800 text-gray-800 dark:text-gray-200 rounded-tl-sm'
                                            }`}>
                                            <div className={`prose prose-sm max-w-none ${msg.role === 'user' ? 'prose-invert' : 'dark:prose-invert'
                                                }`}>
                                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                                    {msg.content}
                                                </ReactMarkdown>
                                            </div>

                                            {/* Timestamp */}
                                            <div className={`text-[10px] mt-2 opacity-70 ${msg.role === 'user' ? 'text-white/70' : 'text-gray-500'
                                                }`}>
                                                {new Date(msg.timestamp).toLocaleTimeString()}
                                            </div>
                                        </div>

                                        {/* Avatar (User) */}
                                        {msg.role === 'user' && (
                                            <div className="w-8 h-8 rounded-full bg-primary-500/10 dark:bg-primary-500/20 flex items-center justify-center text-primary-500 flex-shrink-0 mt-1">
                                                <User size={16} />
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : (
                        <div className="text-center text-gray-500 py-12">No data available</div>
                    )}
                </div>
            </div>
        </div>
    );
}
