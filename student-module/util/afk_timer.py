"""
AFK (Away From Keyboard) detection module.
"""

import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class AfkState:
    """Current state of AFK detection."""
    is_afk: bool
    start_time: Optional[float]
    last_active: float
    total_afk_seconds: float

class AfkTimer:
    """
    Tracks user activity and detects AFK periods.
    """
    
    def __init__(self, threshold_seconds: float = 300):
        """
        Initialize the AFK timer.
        
        Args:
            threshold_seconds: Time in seconds before considering user AFK
        """
        self.threshold = threshold_seconds
        self.state = AfkState(
            is_afk=False,
            start_time=None,
            last_active=time.time(),
            total_afk_seconds=0.0
        )
        
    def update(self, is_active: bool) -> bool:
        """
        Update AFK state based on current activity.
        
        Args:
            is_active: Whether user is currently active
            
        Returns:
            bool: True if AFK state changed
        """
        now = time.time()
        changed = False
        
        if is_active:
            self.state.last_active = now
            if self.state.is_afk:
                # User returned from AFK
                self.state.is_afk = False
                if self.state.start_time:
                    self.state.total_afk_seconds += now - self.state.start_time
                self.state.start_time = None
                changed = True
        else:
            if not self.state.is_afk and (now - self.state.last_active) > self.threshold:
                # User became AFK
                self.state.is_afk = True
                self.state.start_time = now
                changed = True
                
        return changed
        
    def get_stats(self) -> dict:
        """
        Get current AFK statistics.
        
        Returns:
            dict: AFK statistics including current state and total time
        """
        return {
            "is_afk": self.state.is_afk,
            "total_afk_seconds": self.state.total_afk_seconds,
            "current_afk_seconds": time.time() - self.state.start_time if self.state.is_afk else 0
        } 