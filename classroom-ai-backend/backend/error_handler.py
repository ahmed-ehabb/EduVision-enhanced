"""
Enhanced Error Handler Module
Provides comprehensive error handling, retry logic, and recovery mechanisms
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from functools import wraps
from enum import Enum
import traceback
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ErrorSeverity(Enum):
    """Error severity levels for different handling strategies"""
    LOW = "low"  # Log and continue
    MEDIUM = "medium"  # Retry with backoff
    HIGH = "high"  # Alert and failover
    CRITICAL = "critical"  # System shutdown

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e

class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

def calculate_backoff_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate exponential backoff delay with optional jitter"""
    delay = min(
        config.initial_delay * (config.exponential_base ** (attempt - 1)),
        config.max_delay
    )
    
    if config.jitter:
        import random
        delay *= random.uniform(0.5, 1.5)
    
    return delay

def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    exceptions: tuple = (Exception,),
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
):
    """Decorator for automatic retry with exponential backoff"""
    if config is None:
        config = RetryConfig()
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        logger.error(
                            f"Max retry attempts ({config.max_attempts}) reached for {func.__name__}. "
                            f"Error: {str(e)}"
                        )
                        break
                    
                    delay = calculate_backoff_delay(attempt, config)
                    logger.warning(
                        f"Attempt {attempt}/{config.max_attempts} failed for {func.__name__}. "
                        f"Retrying in {delay:.2f}s. Error: {str(e)}"
                    )
                    
                    await asyncio.sleep(delay)
            
            # Handle final failure based on severity
            handle_error(last_exception, severity, func.__name__)
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        logger.error(
                            f"Max retry attempts ({config.max_attempts}) reached for {func.__name__}. "
                            f"Error: {str(e)}"
                        )
                        break
                    
                    delay = calculate_backoff_delay(attempt, config)
                    logger.warning(
                        f"Attempt {attempt}/{config.max_attempts} failed for {func.__name__}. "
                        f"Retrying in {delay:.2f}s. Error: {str(e)}"
                    )
                    
                    time.sleep(delay)
            
            # Handle final failure based on severity
            handle_error(last_exception, severity, func.__name__)
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def handle_error(
    error: Exception,
    severity: ErrorSeverity,
    context: str = ""
) -> None:
    """Handle error based on severity level"""
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "severity": severity.value,
        "context": context,
        "timestamp": time.time(),
        "traceback": traceback.format_exc()
    }
    
    if severity == ErrorSeverity.LOW:
        logger.info(f"Low severity error in {context}: {error}")
    elif severity == ErrorSeverity.MEDIUM:
        logger.warning(f"Medium severity error in {context}: {error}")
    elif severity == ErrorSeverity.HIGH:
        logger.error(f"High severity error in {context}: {error}")
        # Send alert to monitoring system
        send_error_alert(error_info)
    elif severity == ErrorSeverity.CRITICAL:
        logger.critical(f"CRITICAL error in {context}: {error}")
        # Send urgent alert and consider system shutdown
        send_critical_alert(error_info)

def send_error_alert(error_info: Dict[str, Any]) -> None:
    """Send error alert to monitoring system"""
    # Placeholder for integration with monitoring service
    logger.info(f"Error alert sent: {error_info['context']}")

def send_critical_alert(error_info: Dict[str, Any]) -> None:
    """Send critical alert to monitoring system"""
    # Placeholder for integration with monitoring service
    logger.critical(f"CRITICAL alert sent: {error_info['context']}")

class ErrorRecoveryManager:
    """Manages error recovery strategies across the system"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
    
    def register_recovery_strategy(
        self,
        service_name: str,
        strategy: Callable[[], Any]
    ) -> None:
        """Register a recovery strategy for a service"""
        self.recovery_strategies[service_name] = strategy
    
    def get_circuit_breaker(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60
    ) -> CircuitBreaker:
        """Get or create circuit breaker for a service"""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )
        return self.circuit_breakers[service_name]
    
    def record_error(self, service_name: str) -> None:
        """Record an error for a service"""
        self.error_counts[service_name] = self.error_counts.get(service_name, 0) + 1
    
    def attempt_recovery(self, service_name: str) -> bool:
        """Attempt to recover a failing service"""
        if service_name in self.recovery_strategies:
            try:
                logger.info(f"Attempting recovery for {service_name}")
                self.recovery_strategies[service_name]()
                self.error_counts[service_name] = 0
                return True
            except Exception as e:
                logger.error(f"Recovery failed for {service_name}: {e}")
                return False
        return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        return {
            "circuit_breakers": {
                name: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count
                }
                for name, cb in self.circuit_breakers.items()
            },
            "error_counts": self.error_counts,
            "timestamp": time.time()
        }

# Global error recovery manager instance
error_recovery_manager = ErrorRecoveryManager()

# Specific error handlers for different components

@retry_with_backoff(
    config=RetryConfig(max_attempts=5, initial_delay=2.0),
    exceptions=(ConnectionError, TimeoutError),
    severity=ErrorSeverity.HIGH
)
async def resilient_database_operation(operation: Callable[..., T], *args, **kwargs) -> T:
    """Execute database operation with retry and circuit breaker"""
    cb = error_recovery_manager.get_circuit_breaker("database")
    return await cb.call(operation, *args, **kwargs)

@retry_with_backoff(
    config=RetryConfig(max_attempts=3, initial_delay=1.0),
    exceptions=(RuntimeError,),
    severity=ErrorSeverity.MEDIUM
)
async def resilient_ai_model_operation(operation: Callable[..., T], *args, **kwargs) -> T:
    """Execute AI model operation with retry logic"""
    try:
        return await operation(*args, **kwargs)
    except Exception as e:
        # Attempt to reload model if it fails
        if "model" in str(e).lower():
            logger.warning("AI model error detected, attempting recovery")
            error_recovery_manager.attempt_recovery("ai_models")
        raise

def graceful_degradation(
    primary_func: Callable[..., T],
    fallback_func: Callable[..., T],
    exceptions: tuple = (Exception,)
) -> Callable[..., T]:
    """Decorator for graceful degradation to fallback functionality"""
    def wrapper(*args, **kwargs) -> T:
        try:
            return primary_func(*args, **kwargs)
        except exceptions as e:
            logger.warning(
                f"Primary function {primary_func.__name__} failed, "
                f"falling back to {fallback_func.__name__}. Error: {e}"
            )
            return fallback_func(*args, **kwargs)
    
    return wrapper

# WebSocket connection recovery
class WebSocketReconnector:
    """Handles WebSocket reconnection with exponential backoff"""
    
    def __init__(self, url: str, max_attempts: int = 10):
        self.url = url
        self.max_attempts = max_attempts
        self.attempt = 0
        self.connected = False
    
    async def connect_with_retry(self, on_message: Callable) -> Any:
        """Connect to WebSocket with automatic retry"""
        config = RetryConfig(max_attempts=self.max_attempts, initial_delay=1.0)
        
        while self.attempt < self.max_attempts:
            try:
                # WebSocket connection logic here
                self.connected = True
                self.attempt = 0
                logger.info(f"WebSocket connected to {self.url}")
                return True
            except Exception as e:
                self.attempt += 1
                delay = calculate_backoff_delay(self.attempt, config)
                logger.warning(
                    f"WebSocket connection attempt {self.attempt} failed. "
                    f"Retrying in {delay:.2f}s"
                )
                await asyncio.sleep(delay)
        
        logger.error(f"Failed to connect to WebSocket after {self.max_attempts} attempts")
        return False 

class AppError(Exception):
    def __init__(self, message: str, status_code: int = 500, details: Dict[str, Any] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

async def error_handler(request: Request, exc: Union[AppError, Exception]) -> JSONResponse:
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "message": str(exc.detail),
                "code": exc.status_code
            }
        )
    
    if isinstance(exc, AppError):
        logger.error(f"Application error: {exc.message}", exc_info=True)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "message": exc.message,
                "code": exc.status_code,
                "details": exc.details
            }
        )
    
    # Unexpected errors
    logger.critical(f"Unexpected error: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An unexpected error occurred",
            "code": 500
        }
    )

class ValidationError(AppError):
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, status_code=400, details=details)

class AuthenticationError(AppError):
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)

class AuthorizationError(AppError):
    def __init__(self, message: str = "Not authorized"):
        super().__init__(message, status_code=403)

class NotFoundError(AppError):
    def __init__(self, resource: str):
        super().__init__(f"{resource} not found", status_code=404)

class ConflictError(AppError):
    def __init__(self, message: str):
        super().__init__(message=message, status_code=409)

def handle_model_error(error: Exception, model_name: str, context: str = "") -> None:
    """Handle AI model-specific errors with appropriate logging and recovery."""
    error_info = {
        "model_name": model_name,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
        "timestamp": time.time(),
        "traceback": traceback.format_exc()
    }
    
    if isinstance(error, (ValueError, TypeError)):
        # Input validation errors
        logger.warning(f"Model input error for {model_name}: {error}")
        severity = ErrorSeverity.LOW
    elif isinstance(error, RuntimeError):
        # Runtime errors (e.g., CUDA out of memory)
        logger.error(f"Model runtime error for {model_name}: {error}")
        severity = ErrorSeverity.HIGH
        # Try to free GPU memory
        from .gpu_memory_manager import gpu_memory_manager
        gpu_memory_manager.cleanup_memory()
    elif isinstance(error, ImportError):
        # Missing dependencies
        logger.error(f"Model dependency error for {model_name}: {error}")
        severity = ErrorSeverity.HIGH
    else:
        # Unknown errors
        logger.error(f"Unexpected error for {model_name}: {error}")
        severity = ErrorSeverity.MEDIUM
    
    handle_error(error, severity, f"Model: {model_name} - {context}") 