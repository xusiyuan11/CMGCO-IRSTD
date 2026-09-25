"""Logging facilities for CMGCO training and evaluation."""

import logging
import os
import time
import datetime


def get_root_logger(logger_name="cmgco", log_level=logging.INFO, log_file=None):
    """Get the root logger with stream and optional file handler."""
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        return logger

    format_str = "%(asctime)s %(levelname)s: %(message)s"
    logging.basicConfig(format=format_str, level=log_level)

    if log_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, "a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


class MessageLogger:
    """Pretty logger for epoch summaries and metrics."""

    def __init__(self, exp_name, total_epochs, logger=None):
        self.exp_name = exp_name
        self.total_epochs = total_epochs
        self.logger = logger if logger is not None else get_root_logger()
        self.start_time = time.time()

    def log_epoch(self, epoch, lr, train_loss, train_metrics, val_metrics=None, extra_info=None):
        elapsed = time.time() - self.start_time
        avg_per_epoch = elapsed / max(1, epoch)
        eta_sec = avg_per_epoch * max(0, self.total_epochs - epoch)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

        msg = f"[{self.exp_name}] Epoch: [{epoch:3d}/{self.total_epochs:3d}] | LR: {lr:.3e} | ETA: {eta_str}\n"
        msg += f"  Train Loss: {train_loss:.4e}"
        if train_metrics:
            msg += f" | Train IoU: {train_metrics.get('IoU', 0):.2f}% | nIoU: {train_metrics.get('nIoU', 0):.2f}%"

        if val_metrics:
            msg += f"\n  Val Evaluation: IoU: {val_metrics.get('IoU', 0):.2f}% | nIoU: {val_metrics.get('nIoU', 0):.2f}% | Pd: {val_metrics.get('Pd', 0):.2f}% | Fa: {val_metrics.get('Fa', 0):.2f} (10^-6)"

        if extra_info:
            msg += f"\n  Info: {extra_info}"

        self.logger.info(msg)
