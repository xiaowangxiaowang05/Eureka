import subprocess
import os
import json
import logging
import time

from utils.extract_task_code import file_to_string


def set_freest_gpu():
    freest_gpu = get_freest_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(freest_gpu)


def get_freest_gpu():
    sp = subprocess.Popen(["gpustat", "--json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode("utf-8"))
    # Find GPU with most free memory
    freest_gpu = min(gpustats["gpus"], key=lambda x: x["memory.used"])

    return freest_gpu["index"]


def filter_traceback(s):
    lines = s.split("\n")
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith("Traceback"):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return "\n".join(filtered_lines)
    return ""  # Return an empty string if no Traceback is found


def block_until_training(rl_filepath, log_status: bool = False, iter_num: int = -1, response_id: int = -1, timeout_sec: int = 600):
    """
    Wait until RL training has clearly started or failed.
    
    This function monitors the training log file and returns when:
    - Training loop has started (detected by "fps step:" output)
    - Training environment has initialized (detected by "Tensorboard Directory:" or "Started to train")
    - An error occurred (detected by "Traceback")
    - Timeout reached (default 600 seconds)
    
    Args:
        rl_filepath: Path to the training log file
        log_status: Whether to log status messages
        iter_num: Iteration number for logging
        response_id: Response ID for logging
        timeout_sec: Maximum time to wait in seconds (default 600)
    """
    start_time = time.time()
    
    while True:
        try:
            rl_log = file_to_string(rl_filepath)
        except Exception:
            rl_log = ""
        
        # Check for training loop output (original check - highest priority)
        has_fps = "fps step:" in rl_log
        has_traceback = "Traceback" in rl_log
        
        # Check for training initialization signals (new checks for robustness)
        has_tensorboard_dir = "Tensorboard Directory:" in rl_log
        has_started_train = "Started to train" in rl_log
        
        # Training loop started - this is the most reliable signal
        if has_fps:
            if log_status:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully training!")
            break
        
        # Error detected - exit immediately
        if has_traceback:
            if log_status:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            break
        
        # Training environment initialized but no fps yet - wait a bit more for fps to appear
        # This handles cases where GPU is busy and training loop hasn't started printing yet
        if has_tensorboard_dir or has_started_train:
            # Wait up to 60 seconds for fps to appear after initialization
            elapsed = time.time() - start_time
            if elapsed > 60:
                if log_status:
                    logging.info(
                        f"Iteration {iter_num}: Code Run {response_id} training initialized "
                        f"(Tensorboard/Started detected) but no fps yet after {int(elapsed)}s. "
                        "Proceeding assuming training is active."
                    )
                break
        
        # Timeout protection - prevent infinite waiting
        elapsed = time.time() - start_time
        if elapsed > timeout_sec:
            if log_status:
                logging.warning(
                    f"Iteration {iter_num}: Code Run {response_id} timeout after {timeout_sec}s "
                    f"waiting for training log '{rl_filepath}'. Proceeding anyway."
                )
            break
        
        # Sleep to avoid excessive file I/O
        time.sleep(1.0)


if __name__ == "__main__":
    print(get_freest_gpu())