import os
import logging

logger = logging.getLogger(__name__)

class UserActivityService:
    """
    Stub for the missing UserActivityService.
    Provides access to the user activity log file.
    """
    def __init__(self, log_path: str = "logs/user_activity.jsonl"):
        # Ensure log path is absolute or relative to base
        self.log_path = log_path
        
        # Ensure the logs directory exists
        log_dir = os.path.dirname(self.log_path)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create logs directory: {e}")

def get_user_activity_service():
    """Returns a singleton-like instance of the UserActivityService stub."""
    return UserActivityService()
