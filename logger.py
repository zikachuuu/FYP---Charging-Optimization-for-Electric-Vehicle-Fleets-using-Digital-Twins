from __future__ import annotations
import logging
from typing import Optional, Union
import os

# Map string levels to logging levels
_LEVELS = {
    "CRITICAL"  : logging.CRITICAL, # highest level, only critical errors
    "ERROR"     : logging.ERROR   ,
    "WARNING"   : logging.WARNING ,
    "INFO"      : logging.INFO    , # default level 
    "DEBUG"     : logging.DEBUG   , # lowest level
}

def _to_level(
        level: Union[str, int]
    ) -> int:
    """
    Convert a string or integer level to a logging level.
    """
    if isinstance(level, int):
        return level
    return _LEVELS.get(str(level).upper(), logging.INFO)

class Logger:
    """
    Simple per-instance logger.
    
    Example usage:
    ```python
    logger = Logger("my_logger", level="INFO", to_console=True)
    logger.info("This is an info message.")
    logger.debug("This message will not appear.")
    logger.set_level("DEBUG")   # Change level to DEBUG
    logger.save("my_log.txt")   # Save future logs to a file
    logger.debug("Now this message will be logged.")
    logger.close()              # Close the file handler when done
    ```
    """

    def __init__(
            self, 
            name:       str, 
            level:      Union[str, int] = "INFO", 
            to_console: bool = True
        ):
        """
        Initialize the logger with a name, level, and whether to log to console.            
        This logger uses a unique internal name to avoid conflicts with other instances.    
        The log format includes a timestamp, level, and the short name of the logger.       

        :param name: Name of the logger instance (for identification)
        :param level: Logging level (string or int), default is "INFO". Choose from:

            - "CRITICAL"
            - "ERROR"
            - "WARNING"
            - "INFO" (default)
            - "DEBUG"

        :param to_console: If True, log messages will also be printed to console.
        """
        self._display_name = name

        # Unique internal logger to avoid handler collisions between instances
        internal_name = f"simplelogger.{name}.{id(self)}"
        self._logger = logging.getLogger(internal_name)
        self._logger.propagate = False  # don't bubble to root
        self.set_level(level)

        self._formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(short_name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        self._file_handler: Optional[logging.Handler] = None

        if to_console:
            sh = logging.StreamHandler()
            sh.setFormatter(self._formatter)
            sh.addFilter(self._name_filter())
            self._logger.addHandler(sh)

    def _name_filter(self) -> logging.Filter:
        # Injects a short_name attribute for pretty output
        display_name = self._display_name
        class _F(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                record.short_name = display_name
                return True
        return _F()

    # Public API
    def set_level(
            self, 
            level: Union[str, int]
        ) -> None:
        """
        Set the logging level for this logger instance.
        """
        self._logger.setLevel(_to_level(level))

    def log(
            self, 
            level:      Union[str, int], 
            message:    str, 
            *args, 
            **kwargs
        ) -> None:
        """
        Log a message at the given level (string or int).
        """
        self._logger.log(_to_level(level), message, *args, **kwargs)

    # Convenience methods (optional to use)
    def debug   (self, msg, *a, **k): self.log("DEBUG"      , msg, *a, **k)
    def info    (self, msg, *a, **k): self.log("INFO"       , msg, *a, **k)
    def warning (self, msg, *a, **k): self.log("WARNING"    , msg, *a, **k)
    def error   (self, msg, *a, **k): self.log("ERROR"      , msg, *a, **k)
    def critical(self, msg, *a, **k): self.log("CRITICAL"   , msg, *a, **k)

    def save(
            self, 
            file_name:  str, 
            mode:       str = "a", 
            encoding:   str = "utf-8"
        ) -> None:
        """
        Save future log messages to a file named `file_name` in the "Logs" directory in the current working directory.
        Ensure the "Logs" directory exists before calling this method.
        If a file handler already exists, it will be closed and removed before creating a new one.

        :param file_name: Name of the log file (will be created in the current working directory under "Logs"). 
                            Do not include the ".log" extension, it will be added automatically.
        :param mode: File mode, default is "a" (append). Use "w" to overwrite.
        :param encoding: File encoding, default is "utf-8".
        """
        if self._file_handler is not None:
            try:
                self._logger.removeHandler(self._file_handler)
                self._file_handler.close()
            except Exception:
                pass
            self._file_handler = None

        file_path = os.path.join(os.getcwd(), "Logs", file_name + ".log")

        fh = logging.FileHandler(file_path, mode=mode, encoding=encoding)
        fh.setFormatter(self._formatter)
        fh.addFilter(self._name_filter())
        fh.setLevel(self._logger.level)
        self._logger.addHandler(fh)
        self._file_handler = fh

    def close(self) -> None:
        """Close any attached file handler."""
        if self._file_handler is not None:
            try:
                self._logger.removeHandler(self._file_handler)
                self._file_handler.close()
            finally:
                self._file_handler = None
    
    def __del__(self):
        """Ensure resources are cleaned up when the logger is deleted."""
        self.close()
        self._logger = None