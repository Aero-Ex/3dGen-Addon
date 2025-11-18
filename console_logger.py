"""
Professional console logging system for TRELLIS addon
Provides consistent, beautiful, and informative console output
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


class Colors:
    """ANSI color codes for terminal output (Windows 10+)"""
    # Status colors
    SUCCESS = '\033[92m'      # Green
    INFO = '\033[96m'         # Cyan
    WARNING = '\033[93m'      # Yellow
    ERROR = '\033[91m'        # Red
    
    # Style colors
    HEADER = '\033[95m'       # Magenta
    BOLD = '\033[1m'          # Bold
    DIM = '\033[2m'           # Dim
    UNDERLINE = '\033[4m'     # Underline
    
    # Reset
    RESET = '\033[0m'
    
    @staticmethod
    def strip_colors(text: str) -> str:
        """Remove ANSI color codes from text"""
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)


class ConsoleLogger:
    """
    Professional logging system for console scripts
    - Dual output: console + file
    - Consistent formatting
    - Progress indicators
    - Timestamp tracking
    """
    
    def __init__(self, log_name: str = "TRELLIS", log_dir: Optional[Path] = None):
        """
        Initialize logger
        
        Args:
            log_name: Name for log file (e.g., "TRELLIS_install")
            log_dir: Directory for log file (default: Documents)
        """
        self.log_name = log_name
        
        # Setup log file
        if log_dir is None:
            log_dir = Path.home() / "Documents"
        
        self.log_file = log_dir / f"{log_name}.log"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file with header
        self._init_log_file()
        
        # Track start time
        self.start_time = datetime.now()
    
    def _init_log_file(self):
        """Initialize log file with header"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("╔" + "═" * 78 + "╗\n")
                f.write(f"║ {self.log_name.center(76)} ║\n")
                f.write(f"║ {datetime.now().strftime('%Y-%m-%d %H:%M:%S').center(76)} ║\n")
                f.write("╚" + "═" * 78 + "╝\n\n")
        except Exception as e:
            print(f"{Colors.WARNING}⚠ Could not create log file: {e}{Colors.RESET}")
    
    def _write_to_log(self, message: str):
        """Write message to log file without colors"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%H:%M:%S")
                clean_msg = Colors.strip_colors(message)
                f.write(f"[{timestamp}] {clean_msg}\n")
                f.flush()
        except Exception:
            pass  # Silently fail to avoid interrupting console output
    
    # ========== Headers and Sections ==========
    
    def header(self, text: str):
        """Print a major header"""
        output = f"\n{Colors.BOLD}{Colors.HEADER}"
        output += "╔" + "═" * 78 + "╗\n"
        output += "║" + text.center(78) + "║\n"
        output += "╚" + "═" * 78 + "╝"
        output += f"{Colors.RESET}\n"
        
        print(output)
        self._write_to_log(text)
    
    def section(self, text: str, icon: str = "▶"):
        """Print a section header"""
        output = f"\n{Colors.BOLD}{Colors.INFO}"
        output += "─" * 80 + "\n"
        output += f"{icon} {text}\n"
        output += "─" * 80
        output += f"{Colors.RESET}\n"
        
        print(output)
        self._write_to_log(f"{icon} {text}")
    
    def subsection(self, text: str):
        """Print a subsection"""
        output = f"\n{Colors.BOLD}• {text}{Colors.RESET}"
        print(output)
        self._write_to_log(f"• {text}")
    
    # ========== Status Messages ==========
    
    def success(self, message: str, indent: int = 0):
        """Print success message"""
        prefix = "  " * indent
        output = f"{prefix}{Colors.SUCCESS}✓{Colors.RESET} {message}"
        print(output)
        self._write_to_log(f"{prefix}✓ {message}")
    
    def info(self, message: str, indent: int = 0):
        """Print info message"""
        prefix = "  " * indent
        output = f"{prefix}{Colors.INFO}ℹ{Colors.RESET} {message}"
        print(output)
        self._write_to_log(f"{prefix}ℹ {message}")
    
    def warning(self, message: str, indent: int = 0):
        """Print warning message"""
        prefix = "  " * indent
        output = f"{prefix}{Colors.WARNING}⚠{Colors.RESET} {message}"
        print(output)
        self._write_to_log(f"{prefix}⚠ {message}")
    
    def error(self, message: str, indent: int = 0):
        """Print error message"""
        prefix = "  " * indent
        output = f"{prefix}{Colors.ERROR}✗{Colors.RESET} {message}"
        print(output)
        self._write_to_log(f"{prefix}✗ {message}")
    
    def plain(self, message: str, indent: int = 0):
        """Print plain message (no icon)"""
        prefix = "  " * indent
        output = f"{prefix}{message}"
        print(output)
        self._write_to_log(f"{prefix}{message}")
    
    # ========== Progress Indicators ==========
    
    def progress(self, current: int, total: int, description: str = ""):
        """Print progress indicator"""
        percentage = (current / total) * 100 if total > 0 else 0
        bar_length = 40
        filled = int(bar_length * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)
        
        output = f"\r{Colors.INFO}[{bar}]{Colors.RESET} "
        output += f"{current}/{total} ({percentage:.1f}%) "
        if description:
            output += f"- {description}"
        
        print(output, end='', flush=True)
        
        if current == total:
            print()  # New line when complete
            self._write_to_log(f"Progress: {current}/{total} ({percentage:.1f}%) - {description}")
    
    def step(self, current: int, total: int, description: str):
        """Print step progress"""
        output = f"{Colors.BOLD}[{current}/{total}]{Colors.RESET} {description}"
        print(output)
        self._write_to_log(f"[{current}/{total}] {description}")
    
    def spinner(self, message: str):
        """Print a message with spinner (call multiple times to animate)"""
        # Simple implementation - just print message
        print(f"\r{Colors.INFO}⟳{Colors.RESET} {message}...", end='', flush=True)
    
    def spinner_stop(self, final_message: str = "Done"):
        """Stop spinner and print final message"""
        self.success(final_message)
    
    # ========== Special Formats ==========
    
    def box(self, lines: list, title: str = None, color: str = None):
        """Print a boxed message"""
        if color is None:
            color = Colors.INFO
        
        max_len = max(len(line) for line in lines) if lines else 0
        if title:
            max_len = max(max_len, len(title))
        
        width = max_len + 4
        
        output = f"\n{color}┌" + "─" * width + "┐\n"
        
        if title:
            output += f"│ {Colors.BOLD}{title}{Colors.RESET}{color}"
            output += " " * (width - len(title) - 1) + "│\n"
            output += "├" + "─" * width + "┤\n"
        
        for line in lines:
            output += f"│  {line}"
            output += " " * (width - len(line) - 2) + "│\n"
        
        output += "└" + "─" * width + "┘" + Colors.RESET + "\n"
        
        print(output)
        self._write_to_log(f"Box: {title}\n" + "\n".join(lines))
    
    def divider(self, char: str = "─", length: int = 80):
        """Print a horizontal divider"""
        output = f"{Colors.DIM}{char * length}{Colors.RESET}"
        print(output)
    
    # ========== Summary and Timing ==========
    
    def elapsed_time(self) -> str:
        """Get elapsed time since logger creation"""
        elapsed = datetime.now() - self.start_time
        total_seconds = int(elapsed.total_seconds())
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def summary(self, success: bool, message: str = None):
        """Print final summary"""
        elapsed = self.elapsed_time()
        
        self.divider("═")
        
        if success:
            self.success(f"Completed successfully in {elapsed}")
            if message:
                self.plain(f"  {message}", indent=1)
        else:
            self.error(f"Failed after {elapsed}")
            if message:
                self.plain(f"  {message}", indent=1)
        
        self.info(f"Log file: {self.log_file}")
        self.divider("═")
    
    # ========== Utility Methods ==========
    
    def env_info(self):
        """Print environment information"""
        self.subsection("Environment Info")
        self.plain(f"Python: {sys.executable}", indent=1)
        self.plain(f"Version: {sys.version.split()[0]}", indent=1)
        self.plain(f"Platform: {sys.platform}", indent=1)
        self.plain(f"Working Dir: {os.getcwd()}", indent=1)
    
    def instructions(self, lines: list):
        """Print important instructions"""
        self.box(lines, title="⚠ IMPORTANT", color=Colors.WARNING)
    
    # ========== Context Manager Support ==========
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type is not None:
            self.error(f"Exception: {exc_val}")
        return False


# ========== Convenience Functions ==========

def create_logger(log_name: str) -> ConsoleLogger:
    """Create a new console logger"""
    return ConsoleLogger(log_name)


# ========== Demo ==========

if __name__ == "__main__":
    """Demo of logger features"""
    logger = ConsoleLogger("DEMO")
    
    logger.header("TRELLIS Installation Demo")
    
    logger.instructions([
        "This process will take 10-30 minutes",
        "Please keep this window open",
        "You can minimize Blender while this runs"
    ])
    
    logger.env_info()
    
    logger.section("Installing Dependencies", icon="📦")
    
    logger.step(1, 5, "Installing PyTorch")
    logger.info("Downloading from PyTorch index...", indent=1)
    logger.success("PyTorch 2.2.2+cu118 installed", indent=1)
    
    logger.step(2, 5, "Installing xformers")
    logger.warning("Using --no-deps flag", indent=1)
    logger.success("xformers 0.0.24 installed", indent=1)
    
    logger.section("Testing Installation", icon="🔍")
    
    for i in range(1, 11):
        logger.progress(i, 10, f"Testing package {i}")
        import time
        time.sleep(0.1)
    
    logger.box([
        "PyTorch: 2.2.2+cu118 ✓",
        "CUDA: Available ✓",
        "TRELLIS: Ready ✓"
    ], title="Installation Status")
    
    logger.summary(True, "All dependencies installed successfully")
