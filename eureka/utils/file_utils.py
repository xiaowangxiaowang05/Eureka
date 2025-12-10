import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict
import traceback

def find_files_with_substring(directory, substring):
    matches = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if substring in file:
                matches.append(os.path.join(root, file))
    return matches

def load_tensorboard_logs(path):
    """Load tensorboard logs from a directory path (str or Path)."""
    data = defaultdict(list)
    # Convert Path to string if needed
    path_str = str(path) if path else ""
    if not path_str:
        return data
    event_acc = EventAccumulator(path_str)
    event_acc.Reload()  # Load all data written so far

    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        for event in events:
            data[tag].append(event.value)
    
    return data

import importlib.util

def import_class_from_file(file_path, function_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    function = getattr(module, function_name)
    return function


def filter_traceback(log_content: str) -> str:
    """
    提取 Python 报错堆栈信息；若未找到则返回空字符串。
    简单扫描 Traceback 段落并在捕获到异常行后停止。
    """
    if not log_content:
        return ""
    
    lines = log_content.split("\n")
    traceback_lines = []
    capture = False
    marker = "Traceback (most recent call last):"
    
    for line in lines:
        if marker in line:
            capture = True
        if capture:
            traceback_lines.append(line)
            # 当遇到异常行（通常不以空格或 Traceback 开头）后停止
            if line and not line.startswith((" ", "\t")) and not line.startswith("Traceback"):
                break
    
    return "\n".join(traceback_lines) if traceback_lines else ""