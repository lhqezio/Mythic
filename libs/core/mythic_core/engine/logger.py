"""
Mythic AI Engine - Advanced Logging System
Provides comprehensive logging with progress bars, structured logging, and performance monitoring
"""

import os
import sys
import time
import logging
import logging.handlers
import threading
from datetime import datetime
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from contextlib import contextmanager

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with 'pip install tqdm' for progress bars.")

try:
    import colorama
    from colorama import Fore, Back, Style
    colorama.init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # Define fallback colors
    class Fore:
        RED = GREEN = BLUE = YELLOW = MAGENTA = CYAN = WHITE = RESET = ""
    class Back:
        RED = GREEN = BLUE = YELLOW = MAGENTA = CYAN = WHITE = RESET = ""
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ""


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and enhanced formatting"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA + Style.BRIGHT,
    }
    
    def format(self, record):
        # Add color to level name
        if COLORS_AVAILABLE and record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{Style.RESET_ALL}"
        
        # Add timestamp with color
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        if COLORS_AVAILABLE:
            timestamp = f"{Fore.BLUE}{timestamp}{Style.RESET_ALL}"
        
        # Format the message
        formatted = super().format(record)
        
        # Add timestamp prefix
        return f"[{timestamp}] {formatted}"


class ProgressLogger:
    """Progress logging with tqdm integration"""
    
    def __init__(self, 
                 total: int = 100,
                 desc: str = "Progress",
                 unit: str = "it",
                 color: str = "green",
                 show_percentage: bool = True,
                 show_eta: bool = True,
                 show_rate: bool = True):
        """
        Initialize progress logger
        
        Args:
            total: Total number of items to process
            desc: Description of the progress
            unit: Unit of measurement
            color: Progress bar color
            show_percentage: Show percentage completion
            show_eta: Show estimated time remaining
            show_rate: Show processing rate
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.color = color
        self.show_percentage = show_percentage
        self.show_eta = show_eta
        self.show_rate = show_rate
        
        self.pbar = None
        self.start_time = None
        self.current = 0
        
        if TQDM_AVAILABLE:
            self._create_progress_bar()
    
    def _create_progress_bar(self):
        """Create the tqdm progress bar"""
        if not TQDM_AVAILABLE:
            return
            
        self.pbar = tqdm(
            total=self.total,
            desc=self.desc,
            unit=self.unit,
            colour=self.color,
            percentage=self.show_percentage,
            eta=self.show_eta,
            rate=self.show_rate,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    
    def update(self, n: int = 1, description: Optional[str] = None):
        """Update progress by n steps"""
        self.current += n
        
        if self.pbar:
            if description:
                self.pbar.set_description(description)
            self.pbar.update(n)
        else:
            # Fallback without tqdm
            percentage = (self.current / self.total) * 100
            print(f"{self.desc}: {self.current}/{self.total} ({percentage:.1f}%)")
    
    def set_description(self, description: str):
        """Update the progress bar description"""
        if self.pbar:
            self.pbar.set_description(description)
        self.desc = description
    
    def close(self):
        """Close the progress bar"""
        if self.pbar:
            self.pbar.close()
    
    def __enter__(self):
        """Context manager entry"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"{self.desc} completed in {elapsed:.2f}s")


class MythicLogger:
    """Main logging class for Mythic AI Engine"""
    
    def __init__(self, 
                 name: str = "mythic",
                 level: str = "INFO",
                 log_file: Optional[str] = None,
                 log_dir: str = "logs",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_colors: bool = True):
        """
        Initialize Mythic logger
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Log file path
            log_dir: Directory for log files
            max_file_size: Maximum log file size before rotation
            backup_count: Number of backup files to keep
            enable_console: Enable console logging
            enable_file: Enable file logging
            enable_colors: Enable colored output
        """
        self.name = name
        self.level = getattr(logging, level.upper())
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_colors = enable_colors and COLORS_AVAILABLE
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup formatters and handlers
        self._setup_formatters()
        self._setup_handlers()
        
        # Performance tracking
        self.performance_metrics = {}
        self.timers = {}
    
    def _setup_formatters(self):
        """Setup log formatters"""
        # Console formatter
        if self.enable_colors:
            self.console_formatter = ColoredFormatter(
                '%(levelname)-8s %(name)s: %(message)s'
            )
        else:
            self.console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)-8s %(name)s: %(message)s',
                datefmt='%H:%M:%S'
            )
        
        # File formatter (detailed)
        self.file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _setup_handlers(self):
        """Setup log handlers"""
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            console_handler.setFormatter(self.console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.enable_file:
            log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.level)
            file_handler.setFormatter(self.file_formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message"""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message"""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message"""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message"""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message"""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, *args, **kwargs)
    
    def progress(self, **kwargs) -> ProgressLogger:
        """Create a progress logger"""
        return ProgressLogger(**kwargs)
    
    @contextmanager
    def timer(self, name: str, log_level: str = "INFO"):
        """Context manager for timing operations"""
        start_time = time.time()
        self.logger.log(getattr(logging, log_level.upper()), f"Starting {name}")
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.logger.log(getattr(logging, log_level.upper()), f"Completed {name} in {elapsed:.2f}s")
            self.performance_metrics[name] = elapsed
    
    def start_timer(self, name: str):
        """Start a named timer"""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str, log_level: str = "INFO") -> float:
        """End a named timer and return elapsed time"""
        if name not in self.timers:
            self.warning(f"Timer '{name}' was not started")
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        
        self.logger.log(getattr(logging, log_level.upper()), f"{name} took {elapsed:.2f}s")
        self.performance_metrics[name] = elapsed
        
        return elapsed
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        self.performance_metrics[operation] = duration
        self.info(f"Performance: {operation} took {duration:.2f}s", extra=kwargs)
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance metrics summary"""
        return self.performance_metrics.copy()
    
    def log_config(self, config: Dict[str, Any], section: str = "Configuration"):
        """Log configuration in a readable format"""
        self.info(f"=== {section} ===")
        for key, value in config.items():
            if isinstance(value, dict):
                self.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    self.info(f"    {sub_key}: {sub_value}")
            else:
                self.info(f"  {key}: {value}")
        self.info("=" * (len(section) + 8))
    
    def log_system_info(self):
        """Log system information"""
        import platform
        import psutil
        
        self.info("=== System Information ===")
        self.info(f"  Platform: {platform.platform()}")
        self.info(f"  Python: {platform.python_version()}")
        self.info(f"  CPU Cores: {psutil.cpu_count()}")
        self.info(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        if hasattr(psutil, 'gpu_count'):
            try:
                gpu_count = psutil.gpu_count()
                self.info(f"  GPU Count: {gpu_count}")
            except:
                self.info("  GPU Count: Unknown")
        
        self.info("=" * 28)


# Global logger instance
_global_logger = None

def get_logger(name: str = "mythic", **kwargs) -> MythicLogger:
    """Get or create a global logger instance"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = MythicLogger(name, **kwargs)
    
    return _global_logger


def setup_logging(config: Dict[str, Any]) -> MythicLogger:
    """Setup logging from configuration"""
    log_config = config.get('logging', {})
    
    logger = MythicLogger(
        name=log_config.get('name', 'mythic'),
        level=log_config.get('level', 'INFO'),
        log_file=log_config.get('log_file'),
        log_dir=log_config.get('log_dir', 'logs'),
        max_file_size=log_config.get('max_file_size', 10 * 1024 * 1024),
        backup_count=log_config.get('backup_count', 5),
        enable_console=log_config.get('enable_console', True),
        enable_file=log_config.get('enable_file', True),
        enable_colors=log_config.get('enable_colors', True)
    )
    
    return logger


# Convenience functions for quick logging
def debug(message: str, *args, **kwargs):
    """Quick debug log"""
    get_logger().debug(message, *args, **kwargs)

def info(message: str, *args, **kwargs):
    """Quick info log"""
    get_logger().info(message, *args, **kwargs)

def warning(message: str, *args, **kwargs):
    """Quick warning log"""
    get_logger().warning(message, *args, **kwargs)

def error(message: str, *args, **kwargs):
    """Quick error log"""
    get_logger().error(message, *args, **kwargs)

def critical(message: str, *args, **kwargs):
    """Quick critical log"""
    get_logger().critical(message, *args, **kwargs)

def exception(message: str, *args, **kwargs):
    """Quick exception log"""
    get_logger().exception(message, *args, **kwargs)


# Progress bar convenience functions
def progress_bar(total: int, desc: str = "Progress", **kwargs) -> ProgressLogger:
    """Create a progress bar"""
    return get_logger().progress(total=total, desc=desc, **kwargs)


@contextmanager
def timer(name: str, log_level: str = "INFO"):
    """Context manager for timing operations"""
    with get_logger().timer(name, log_level):
        yield


def start_timer(name: str):
    """Start a named timer"""
    get_logger().start_timer(name)


def end_timer(name: str, log_level: str = "INFO") -> float:
    """End a named timer"""
    return get_logger().end_timer(name, log_level)
