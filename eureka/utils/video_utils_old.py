import logging
import os
import subprocess
import time
import shutil
import select  # for non-blocking log reads
from pathlib import Path
from typing import Dict, Optional


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
    gpu_id: str = "0",
) -> Optional[Path]:
    """Launch an Isaac Gym evaluation run that records a rollout video.

    该实现直接从新版 Eureka 复制而来，已经在同一套 Isaac Gym 环境下验证可以稳定录制视频。
    """
    logging.info("Recording rollout video for checkpoint: %s", checkpoint_path)
    before = _collect_videos(workspace_dir)
    start_time = time.time()

    # 完全对齐新 Eureka：headless 时默认使用 Xvfb + 显式启用渲染与相机传感器
    use_xvfb = headless
    gym_headless_arg = False if use_xvfb else headless

    if use_xvfb and shutil.which("xvfb-run") is None:
        logging.error("xvfb-run not found! Please install: sudo apt-get install xvfb")
        use_xvfb = False
        gym_headless_arg = True

    # Ensure a single short rollout and real-time stdout
    base_cmd = [
        "python",
        "-u",  # unbuffered stdout for real-time monitoring
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
        "max_iterations=1",  # run once and exit
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
    env["PYTHONUNBUFFERED"] = "1"  # flush stdout promptly
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if use_xvfb:
        video_cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24"] + base_cmd
    else:
        video_cmd = base_cmd

    log_path = workspace_dir / f"eureka_best_policy_video_{int(start_time)}.txt"

    # Smart monitoring with live log inspection
    max_wait_time = max(rollout_steps * 5 + 180, 400)

    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            video_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,  # line buffered
        )

        start_monitor = time.time()

        try:
            while True:
                # Timeout guard
                if time.time() - start_monitor > max_wait_time:
                    logging.error(f"Video recording timed out after {max_wait_time}s.")
                    process.terminate()
                    raise TimeoutError("Video recording timed out")

                return_code = process.poll()

                # Non-blocking read of any available output
                while True:
                    reads = [process.stdout.fileno()]
                    ready, _, _ = select.select(reads, [], [], 0.0)
                    if not ready:
                        break

                    line = process.stdout.readline()
                    if not line:
                        break

                    log_file.write(line)
                    log_file.flush()

                    # Isaac Gym 完成一次 rollout 时常见的日志行
                    if "Run finished" in line:
                        logging.info("Detected run completion from logs. Exiting early.")
                        process.terminate()
                        return_code = 0
                        break

                    # 常见崩溃模式
                    if "CUDA error" in line or "RuntimeError" in line:
                        logging.error(f"Simulation crashed: {line.strip()}")

                if return_code is not None:
                    if return_code != 0:
                        logging.warning(
                            f"Video recording process exited with code {return_code}. Log: {log_path}"
                        )
                    break

                time.sleep(0.5)
        except Exception as exc:
            logging.error(f"Error during video recording: {exc}")
            if process.poll() is None:
                process.kill()
            raise
        finally:
            if process.stdout:
                process.stdout.close()

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
    if latest_video.stat().st_size < 500:
        logging.warning(f"Video file created but empty: {latest_video}")
        return None

    logging.info(f"Video successfully recorded: {latest_video}")
    return latest_video



