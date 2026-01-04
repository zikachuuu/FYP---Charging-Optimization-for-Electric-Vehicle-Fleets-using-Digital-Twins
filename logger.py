from __future__ import annotations
import logging
import logging.handlers
import os
import sys
import multiprocessing
from datetime import datetime
from typing import Optional, Union

# Map string levels to logging levels
_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR"   : logging.ERROR,
    "WARNING" : logging.WARNING,
    "INFO"    : logging.INFO,
    "DEBUG"   : logging.DEBUG,
}


def _to_level(level: Union[str, int]) -> int:
    if isinstance(level, int): return level
    return _LEVELS.get(str(level).upper(), logging.INFO)


class LogListener:
    """
    Multiprocessing Log Server.
    Runs in the Main Process to collect logs from all workers and write to a single file.

    To use this:
        1. In main.py, set up the Queue and LogListener.
            manager = multiprocessing.Manager()
            log_queue = manager.Queue()
            listener = LogListener(name, folder_name, file_name, timestamp, log_queue)
            listener.start()

        2. Pass the log_queue to each worker process when creating them.

        3. In each worker, create a local Logger with the same queue.
            local_logger = Logger(name, level, to_console, folder_name, file_name, timestamp, queue=log_queue)
        
        4. After all workers are done, stop the listener in main.py.
            listener.stop()

    """
    def __init__(
            self                                        , 
            name        : str                           ,
            folder_name : str                           ,
            file_name   : str                           , 
            timestamp   : str                           ,
            queue       : multiprocessing.Queue         ,
            to_console  : bool                  = True  ,
        ):
        # Logs will be stored as {current_working_directory} / Logs / {folder_name} / {name}_{file_name}_{timestamp}.log

        self._display_name  = name
        self._folder_name   = folder_name
        self._file_name     = file_name
        self._timestamp     = timestamp
        self.queue          = queue
        
        # Ensure Logs directory exists
        self.log_dir = os.path.join(os.getcwd(), "Logs", self._folder_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.file_path = os.path.join(self.log_dir, f"{self._display_name}_{self._file_name}_{self._timestamp}.log")
        
        # 1. File Handler (The Single Output File)
        # Create a single FileHandler for the multiprocessing run
        self.file_handler = logging.FileHandler(self.file_path, mode='w', encoding='utf-8')
        
        # Format includes Process Name to distinguish workers
        self.formatter = logging.Formatter(
            fmt="%(asctime)s [%(processName)s] %(levelname)-8s %(short_name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.file_handler.setFormatter(self.formatter)

        # 2. Collect Handlers
        handlers = [self.file_handler] # Always write to file

        # 3. Optional Console Handler
        if to_console:
            self.console_handler = logging.StreamHandler()
            self.console_handler.setFormatter(self.formatter)
            handlers.append(self.console_handler)
        else:
            self.console_handler = None

        # The Listener accepts *args for handlers
        self.listener = logging.handlers.QueueListener(
            self.queue, 
            *handlers,  # <--- Unpack the list here
            respect_handler_level=True,
        )

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()
        # QueueListener automatically closes its handlers, so manual closing is optional
    
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class Logger:
    """
    Unified Logger Class.
    
    Modes:
    1. Standalone (queue=None): Behave like a normal logger (console + local file).
    2. Worker (queue=Queue): Send all logs to the main process via Queue.
    """
    def __init__(
            self                                                    , 
            name        : str                                       , 
            level       : Union[str, int]                   = "INFO", 
            to_console  : bool                              = True  ,   # for Standalone mode only
            folder_name : str                               = ""    ,   # for Standalone mode only
            file_name   : str                               = ""    ,   # for Standalone mode only
            timestamp   : str                               = ""    ,   # for Standalone mode only
            queue       : Optional[multiprocessing.Queue]   = None  ,
        ):
        # Logs will be stored as {current_working_directory} / Logs / {folder_name} / {name}_{file_name}_{timestamp}.log

        self._display_name  = name
        self._level         = level
        self._to_console    = to_console
        self._folder_name   = folder_name
        self._file_name     = file_name
        self._timestamp     = timestamp or datetime.now().strftime("%Y%m%d_%H%M")
        self._is_worker     = (queue is not None)

        # Unique internal logger name 
        internal_name = f"logger.{name}.{os.getpid()}.{id(self)}"

        # Create the logger with the internal name
        self._logger = logging.getLogger(internal_name)
        self._logger.setLevel(_to_level(level))
        self._logger.propagate = False
        
        # Helper filter to inject the 'short_name' (aka display name) into log records
        self._name_filter_obj = self._create_name_filter()

        # --- MODE 1: MULTIPROCESSING WORKER ---
        if self._is_worker:
            # Add QueueHandler
            # We do NOT add StreamHandler or FileHandler here.
            # We just ship the packet to the main process.
            qh = logging.handlers.QueueHandler(queue)

            # Attach the filter to add 'short_name' so that the Listener can use it
            qh.addFilter(self._name_filter_obj)
            self._logger.addHandler(qh)

        # --- MODE 2: STANDALONE (OLD FUNCTIONALITY) ---
        else:
            self._formatter = logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(short_name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            self._file_handler: Optional[logging.Handler] = None

            if to_console:
                sh = logging.StreamHandler(sys.stdout)
                sh.setFormatter(self._formatter)
                sh.addFilter(self._name_filter_obj)
                self._logger.addHandler(sh)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _create_name_filter(self):
        # Helper to create a filter that adds 'short_name' to log records
        # short_name is the display name of this logger
        name = self._display_name
        class _F(logging.Filter):
            def filter(self, record):
                record.short_name = name
                return True
        return _F()


    # --- Standard Logging Methods ---
    def log(
            self, 
            level, 
            msg, 
            *args, 
            **kwargs
        ) -> None:
        self._logger.log(_to_level(level), msg, *args, **kwargs)

    def debug   (self, msg, *a, **k): self.log("DEBUG"      , msg, *a, **k)
    def info    (self, msg, *a, **k): self.log("INFO"       , msg, *a, **k)
    def warning (self, msg, *a, **k): self.log("WARNING"    , msg, *a, **k)
    def error   (self, msg, *a, **k): self.log("ERROR"      , msg, *a, **k)
    def critical(self, msg, *a, **k): self.log("CRITICAL"   , msg, *a, **k)

    # --- File Management (Only for Standalone Mode) ---
    def save(
            self                    , 
            mode    : str = "w"     , 
            encoding: str = "utf-8" ,
        ) -> None:
        """
        Enable saving to file.
        NOTE: In Multiprocessing mode, this is ignored (logs go to the Listener's file).
        """
        if self._is_worker:
            # Workers cannot decide where to save. The Listener decides.
            raise RuntimeError("Workers cannot save logs to file. The Listener decides.")

        if self._file_handler is not None:
            self.close()

        log_dir = os.path.join(os.getcwd(), "Logs", self._folder_name)
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, f"{self._display_name}_{self._file_name}_{self._timestamp}.log")

        fh = logging.FileHandler(file_path, mode=mode, encoding=encoding)
        fh.setFormatter (self._formatter)
        fh.addFilter    (self._name_filter_obj)
        fh.setLevel     (self._logger.level)
        
        self._logger.addHandler(fh)
        self._file_handler = fh


    def close(self) -> None:
        if self._is_worker:
            raise RuntimeError("Workers do not manage file handlers.")
            
        if hasattr(self, "_file_handler") and self._file_handler:
            try:
                self._logger.removeHandler(self._file_handler)
                self._file_handler.close()
            except Exception:
                pass
            self._file_handler = None
        else:
            raise RuntimeError("No file handler to close.")
        
