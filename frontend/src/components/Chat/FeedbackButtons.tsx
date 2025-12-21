import React, { useState } from "react";
import { ThumbsUp, ThumbsDown } from "lucide-react";
import axios from "axios";
import { API_URL } from "../../lib/api";

interface FeedbackProps {
  messageId: string;
  sessionId: string;
  query: string;
  routingInfo?: any;
  onFeedbackSent?: (rating: number) => void;
}

export const FeedbackButtons: React.FC<FeedbackProps> = ({
  messageId,
  sessionId,
  query,
  routingInfo,
  onFeedbackSent,
}) => {
  const [submitted, setSubmitted] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  const sendFeedback = async (rating: number) => {
    setLoading(true);
    try {
      await axios.post(`${API_URL || ''}/api/feedback`, {
        message_id: messageId,
        session_id: sessionId,
        query,
        rating,
        routing_info: routingInfo || {},
      }, { withCredentials: true });
      setSubmitted(rating);
      if (onFeedbackSent) onFeedbackSent(rating);
    } catch (err) {
      // Optionally show error toast
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "flex", gap: 4 }}>
      <button
        type="button"
        title="Helpful"
        disabled={loading || submitted !== null}
        onClick={() => sendFeedback(1)}
        className="p-1 rounded transition-all hover:scale-110 active:scale-95 disabled:opacity-30"
        style={{
          color: submitted === 1 ? "var(--accent-primary)" : "var(--text-secondary)",
          background: 'transparent',
          border: 'none',
          cursor: loading || submitted !== null ? 'default' : 'pointer',
        }}
      >
        <ThumbsUp className="w-4 h-4" />
      </button>
      <button
        type="button"
        title="Not Helpful"
        disabled={loading || submitted !== null}
        onClick={() => sendFeedback(-1)}
        className="p-1 rounded transition-all hover:scale-110 active:scale-95 disabled:opacity-30"
        style={{
          color: submitted === -1 ? "var(--accent-primary)" : "var(--text-secondary)",
          background: 'transparent',
          border: 'none',
          cursor: loading || submitted !== null ? 'default' : 'pointer',
        }}
      >
        <ThumbsDown className="w-4 h-4" />
      </button>
    </div>
  );
};
