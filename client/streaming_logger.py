"""
Streaming logger for real-time log display in Streamlit.

This module provides a callback-based logging system that allows capturing
and displaying logs as they are generated during agent execution.
"""

import sys
import io
import threading
from typing import Callable, Optional, List
from datetime import datetime


class StreamingLogHandler:
    """
    Handler that captures print/log output and calls a callback for each line.
    
    This allows real-time processing and display of logs as they are generated.
    """
    
    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the streaming log handler.
        
        Args:
            callback: Optional function to call with each log line as it's captured
        """
        self.callback = callback
        self.logs: List[str] = []
        self._lock = threading.Lock()
        self._original_stdout = None
        self._original_stderr = None
        self._capturing = False
    
    def _process_line(self, line: str):
        """Process a complete line of output."""
        if not line.strip():
            return
        
        with self._lock:
            self.logs.append(line)
            
        # Call callback if provided
        if self.callback:
            try:
                self.callback(line)
            except Exception as e:
                # Don't let callback errors break logging
                print(f"Warning: Log callback error: {e}", file=sys.__stderr__)
    
    def __enter__(self):
        """Start capturing output."""
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        
        # Create tee streams
        sys.stdout = TeeStream(self._original_stdout, self._process_line)
        sys.stderr = TeeStream(self._original_stderr, self._process_line)
        
        self._capturing = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop capturing and restore original streams."""
        if self._capturing:
            # Flush any remaining content
            if hasattr(sys.stdout, 'flush_buffer'):
                sys.stdout.flush_buffer()
            if hasattr(sys.stderr, 'flush_buffer'):
                sys.stderr.flush_buffer()
            
            # Restore original streams
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
            self._capturing = False
    
    def get_logs(self) -> List[str]:
        """Get all captured logs."""
        with self._lock:
            return self.logs.copy()


class TeeStream:
    """Stream that writes to original stream and captures lines for callback."""
    
    def __init__(self, original_stream, line_callback: Callable[[str], None]):
        self.original = original_stream
        self.line_callback = line_callback
        self.buffer = ""
    
    def write(self, text: str):
        """Write text to original stream and capture complete lines."""
        # Write to original
        self.original.write(text)
        self.original.flush()
        
        # Buffer and process lines
        self.buffer += text
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line:  # Only process non-empty lines
                self.line_callback(line)
    
    def flush(self):
        """Flush the stream and any buffered content."""
        self.original.flush()
        self.flush_buffer()
    
    def flush_buffer(self):
        """Flush any remaining buffered content as a line."""
        if self.buffer.strip():
            self.line_callback(self.buffer)
            self.buffer = ""
    
    def isatty(self):
        """Check if stream is a TTY."""
        return self.original.isatty()


class EventEmitter:
    """
    Event emitter for structured agent execution events.
    
    This provides a cleaner API than raw log capture for tracking
    specific agent execution phases and events.
    """
    
    def __init__(self):
        self.events: List[dict] = []
        self.callbacks: List[Callable[[dict], None]] = []
        self._lock = threading.Lock()
    
    def on(self, callback: Callable[[dict], None]):
        """Register an event callback."""
        self.callbacks.append(callback)
    
    def emit(self, event_type: str, data: dict):
        """Emit an event to all registered callbacks."""
        event = {
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        with self._lock:
            self.events.append(event)
        
        # Call all callbacks
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Warning: Event callback error: {e}", file=sys.__stderr__)
    
    def get_events(self) -> List[dict]:
        """Get all emitted events."""
        with self._lock:
            return self.events.copy()


# Event type constants
EVENT_AGENT_START = "agent_start"
EVENT_AGENT_COMPLETE = "agent_complete"
EVENT_CONNECTING = "connecting"
EVENT_CONNECTED = "connected"
EVENT_TOOL_CONFIG = "tool_config"
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_HTTP_REQUEST = "http_request"
EVENT_RAG_SEARCH = "rag_search"
EVENT_THINKING = "thinking"
EVENT_RESPONSE = "response"
EVENT_ERROR = "error"
EVENT_WARNING = "warning"

