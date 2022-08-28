from math import floor
from typing import Optional
from time import localtime, strftime
from timeit import default_timer as time
import logging
import functools


class Timer:
    """
    Simple Timer class which prints elapsed time since its creation
    """

    DEFAULT_TIME_FORMAT = "%H:%M:%S"

    def __init__(self, name: str, logger: Optional[logging.Logger] = None, level: int = logging.INFO):
        self.name = name
        self.level = level
        self.logger = self.init_logger(logger, level)
        self._start_time = self._start()
        self._end_time = None

    def __enter__(self):
        return self

    def __exit__(self, var_type, value, traceback):
        self.stop()

    def _start(self) -> float:
        self.logger.log(msg=f"Started {self.name}", level=self.level)
        return time()

    def _log_elapsed(self) -> None:
        """
        Internal function which logs a correctly formatted string according to elapsed time units
        """
        elapsed = self._end_time - self._start_time
        unit = "seconds"
        if elapsed >= 3600.:
            unit = "minutes"
            hours = elapsed / 3600.
            minutes = hours % 60.
            hours = floor(hours)
            self.logger.log(msg=f"{self.name} took {hours} hours and {minutes:.2f} {unit} to complete", level=self.level)
        elif elapsed >= 60.:
            minutes = floor(elapsed / 60.)
            seconds = elapsed % 60.
            self.logger.log(msg=f"{self.name} took {minutes} minutes and {seconds:.2f} {unit} to complete", level=self.level)
        elif elapsed < 0.1:
            unit = "ms"
            self.logger.log(msg=f"{self.name} took {elapsed * 1000.:.2f} {unit} to complete", level=self.level)
        else:
            self.logger.log(msg=f"{self.name} took {elapsed:.2f} {unit} to complete", level=self.level)

    def stop(self) -> None:
        self._end_time = time()
        self._log_elapsed()

    @staticmethod
    def init_logger(logger: Optional[logging.Logger] = None, level: int = logging.INFO) -> logging.Logger:
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(level)
            if not logger.hasHandlers():
                formatter = logging.Formatter("{levelname} - {asctime} - {message}", datefmt=Timer.DEFAULT_TIME_FORMAT, style="{")
                handler = logging.StreamHandler()
                handler.setFormatter(formatter)
                logger.addHandler(handler)
        return logger

    @staticmethod
    def get_default_timestamp() -> str:
        return f"{strftime(Timer.DEFAULT_TIME_FORMAT, localtime(time()))} -"


def timer(_args=None, *, name: Optional[str] = None, logger: Optional[logging.Logger] = None, level: int = logging.INFO):
    """
    Timer decorator which utilizes a Timer object for timing a given function's runtime
    """
    def timer_decorator(func):
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            if name is None:
                timer_name = func.__name__
            else:
                timer_name = name
            timer_wrapper = Timer(name=timer_name, logger=logger, level=level)
            func_ret_val = func(*args, **kwargs)
            timer_wrapper.stop()
            return func_ret_val
        return wrapper_timer
    if _args is None:
        return timer_decorator
    else:
        return timer_decorator(_args)
    
    
if __name__ == "__main__":   
    from time import sleep
    
    timer_test = Timer("Function a")
    sleep(2.554)
    timer_test.stop()
