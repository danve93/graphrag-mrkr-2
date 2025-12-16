import React, { useState } from "react";
import { IconButton, Tooltip } from "@mui/material";
import ThumbUpIcon from "@mui/icons-material/ThumbUp";
import ThumbDownIcon from "@mui/icons-material/ThumbDown";
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
      <Tooltip title="Helpful" placement="top">
        <IconButton
          size="small"
          disabled={loading || submitted !== null}
          onClick={() => sendFeedback(1)}
          sx={{ 
            padding: "4px",
            color: submitted === 1 ? "#f27a03" : "rgba(242, 122, 3, 0.7)",
            "&:hover": {
              color: "#f27a03",
              backgroundColor: "rgba(242, 122, 3, 0.1)"
            },
            "&:disabled": {
              color: "rgba(242, 122, 3, 0.3)"
            }
          }}
        >
          <ThumbUpIcon fontSize="small" />
        </IconButton>
      </Tooltip>
      <Tooltip title="Not Helpful" placement="top">
        <IconButton
          size="small"
          disabled={loading || submitted !== null}
          onClick={() => sendFeedback(-1)}
          sx={{ 
            padding: "4px",
            color: submitted === -1 ? "#f27a03" : "rgba(242, 122, 3, 0.7)",
            "&:hover": {
              color: "#f27a03",
              backgroundColor: "rgba(242, 122, 3, 0.1)"
            },
            "&:disabled": {
              color: "rgba(242, 122, 3, 0.3)"
            }
          }}
        >
          <ThumbDownIcon fontSize="small" />
        </IconButton>
      </Tooltip>
    </div>
  );
};
