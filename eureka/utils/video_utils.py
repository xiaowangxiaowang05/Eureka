import logging
import os
import subprocess
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional


def _collect_videos(workspace_dir: Path) -> Dict[str, float]:
    """Return a mapping of video file path -> mtime for all Hydra policy outputs."""
    videos: Dict[str, float] = {}
    for video_path in workspace_dir.glob("**/*.mp4"): 
        try:
            videos[str(video_path)] = video_path.stat().st_mtime
        except FileNotFoundError:
            continue
    return videos


def record_policy_rollout(
    isaac_root_dir: Path,
    workspace_dir: Path,
    task_name: str,
    suffix: str,
    checkpoint_path: Path,
    *,
    wandb_username: str = "",
    wandb_project: str = "",
    env: Optional[Dict[str, str]] = None,
    rollout_steps: int = 200,
    headless: bool = True,
    force_render: bool = False,
    seed: int = 0,
    gpu_id: str = "1",
) -> Optional[Path]:
    """Launch an Isaac Gym evaluation run that records a rollout video."""
    logging.info("Recording rollout video for checkpoint: %s", checkpoint_path)
    before = _collect_videos(workspace_dir)
    start_time = time.time()

    use_xvfb = headless
    gym_headless_arg = False if use_xvfb else headless
    
    if use_xvfb and shutil.which("xvfb-run") is None:
        logging.error("xvfb-run not found! Please install: sudo apt-get install xvfb")
        use_xvfb = False
        gym_headless_arg = True 

    base_cmd = [
        "python",
        "-u",
        str(isaac_root_dir / "train.py"),
        "hydra/output=subprocess",
        f"task={task_name}{suffix}",
        f"checkpoint={checkpoint_path}",
        "test=True",
        "capture_video=True",
        f"capture_video_freq={rollout_steps}",
        f"capture_video_len={rollout_steps}",
        f"headless={gym_headless_arg}",
        "force_render=True",
        "task.env.enableCameraSensors=True",
        "multi_gpu=False",
        "graphics_device_id=0",
        "sim_device=cuda:0",
        "rl_device=cuda:0",
        "pipeline=gpu",
        "num_envs=1",
        "max_iterations=0",
        f"seed={seed}",
        "train.params.config.player.games_num=1",
        f"task.env.episodeLength={rollout_steps}",
    ]

    if wandb_username:
        base_cmd.append(f"wandb_entity={wandb_username}")
    if wandb_project:
        base_cmd.append(f"wandb_project={wandb_project}")
    
    if env is None:
        env = os.environ.copy()
    else:
        env = env.copy()

    if use_xvfb:
        video_cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24"] + base_cmd
    else:
        video_cmd = base_cmd

    log_path = workspace_dir / f"vlm_eval_{int(start_time)}.txt"
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(video_cmd, stdout=log_file, stderr=log_file, env=env)
    
    max_wait_time = max(rollout_steps * 5 + 180, 400) 
    try:
        elapsed = 0
        check_interval = 5
        while elapsed < max_wait_time:
            return_code = process.poll()
            if return_code is not None:
                if return_code != 0:
                    logging.warning(f"Video recording process exited with error code {return_code}. Log: {log_path}")
                break
            time.sleep(check_interval)
            elapsed += check_interval
        else:
            logging.error(f"Video recording timed out after {max_wait_time}s.")
            process.terminate()
            time.sleep(5)
            if process.poll() is None:
                process.kill()
            raise TimeoutError(f"Video recording timed out")
    except Exception as exc:
        logging.error(f"Error during video recording: {exc}")
        if process.poll() is None:
            process.kill()
        raise

    after = _collect_videos(workspace_dir)
    new_videos = {
        Path(path)
        for path, mtime in after.items()
        if (path not in before or mtime > before[path]) and mtime >= start_time
    }
    
    if not new_videos:
        logging.warning(f"No new video file found. Check full log at {log_path}")
        return None

    latest_video = max(new_videos, key=lambda p: p.stat().st_mtime)
    if latest_video.stat().st_size < 1000:
        logging.warning(f"Video file created but empty: {latest_video}")
        return None
        
    logging.info(f"Video successfully recorded: {latest_video}")
    return latest_video