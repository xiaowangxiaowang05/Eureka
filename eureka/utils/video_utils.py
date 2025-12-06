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
    gpu_id: str = "1",  # 物理GPU ID，可以通过配置传入
) -> Optional[Path]:
    """Launch an Isaac Gym evaluation run that records a rollout video."""
    logging.info("Recording rollout video for checkpoint: %s", checkpoint_path)
    before = _collect_videos(workspace_dir)
    start_time = time.time()

    # ================= 核心逻辑修正 =================
    # 策略：如果系统是 headless (无物理屏幕)，我们使用 xvfb-run。
    # 但是！为了让 Isaac Gym 进行渲染，我们必须告诉它 headless=False。
    # 这样 Gym 就会尝试打开窗口，xvfb 会接管这个窗口，录像机就能录到内容。
    
    use_xvfb = headless # 如果需要在无头模式下录像，必须用 xvfb
    
    # 传给 Isaac Gym 脚本的参数
    # 关键点：如果有 xvfb，gym_headless 必须是 False
    gym_headless_arg = False if use_xvfb else headless
    
    # 强制开启渲染
    gym_force_render = True 

    # 检查 xvfb 是否存在
    if use_xvfb and shutil.which("xvfb-run") is None:
        logging.error("❌ xvfb-run not found! Please install: sudo apt-get install xvfb")
        # 如果没有 xvfb，只能尝试硬跑 (可能会失败)
        use_xvfb = False
        gym_headless_arg = True 

    # 构建基础命令
    base_cmd = [
        "python",
        "-u",
        str(isaac_root_dir / "train.py"),
        "hydra/output=subprocess",
        f"task={task_name}{suffix}",
        f"checkpoint={checkpoint_path}",
        "test=True",
        "capture_video=True",
        # ⚠️ 关键修复：设置 capture_video_freq 为一个很大的值，确保只在开始时录制一次完整视频
        # 如果设置为 1，每一步都会触发录制（step % 1 == 0），导致生成很多短视频片段
        # 设置为 rollout_steps 确保只在 step=0 时触发一次录制
        f"capture_video_freq={rollout_steps}",  # 设置为 rollout_steps，确保只录制一次
        f"capture_video_len={rollout_steps}",  # 视频长度等于整个 rollout 的长度
        
        # ⚠️ 这里是关键修改：
        f"headless={gym_headless_arg}",          # 骗 Isaac Gym 说我们有屏幕
        f"force_render={gym_force_render}",      # 强制渲染每一帧
        "task.env.enableCameraSensors=True",     # 显式开启相机传感器
        # ⚠️ 重要：当设置了 CUDA_VISIBLE_DEVICES=1 后，物理 GPU 1 被映射为逻辑 GPU 0
        # graphics_device_id 需要使用逻辑 GPU ID（0），而不是物理 GPU ID（1）
        # 否则会出现 "invalid device ordinal" 错误导致段错误
        "graphics_device_id=0",                  # 使用逻辑 GPU 0（对应物理 GPU 1，因为 CUDA_VISIBLE_DEVICES=1）
        "sim_device=cuda:0",                     # 显式设置物理设备（CUDA_VISIBLE_DEVICES=1后，物理GPU1映射为cuda:0）
        "rl_device=cuda:0",                      # 显式设置RL设备
        
        "pipeline=gpu",
        "num_envs=1",
        "max_iterations=0",
        f"seed={seed}",
        "train.params.config.player.games_num=1",
        # ⚠️ 重要：限制episode长度，使其与rollout_steps一致
        # 这样环境只会运行rollout_steps步，而不是默认的600步
        f"task.env.episodeLength={rollout_steps}",
    ]

    if wandb_username:
        base_cmd.append(f"wandb_entity={wandb_username}")
    if wandb_project:
        base_cmd.append(f"wandb_project={wandb_project}")

    # 强制让子进程只看得到指定的GPU
    # 从参数中获取GPU ID（如果传入了多个GPU，只使用第一个用于渲染）
    # 设置 CUDA_VISIBLE_DEVICES 后，物理GPU会被重新映射为逻辑 cuda:0
    target_gpu_id = gpu_id.split(",")[0].strip()  # 如果指定了多个GPU，只取第一个用于渲染
    
    # ⚠️ 重要：当设置了 CUDA_VISIBLE_DEVICES 后，Isaac Gym 只能看到被映射的GPU
    # 例如：CUDA_VISIBLE_DEVICES=1 后，物理GPU 1 被映射为逻辑GPU 0
    # 此时 graphics_device_id 必须使用逻辑GPU ID（0），而不是物理GPU ID（1）
    # 否则会出现 "invalid device ordinal" 错误
    # 替换 base_cmd 中的占位符为逻辑GPU ID（0，因为CUDA_VISIBLE_DEVICES会重新映射）
    logical_gpu_id = 0  # 设置CUDA_VISIBLE_DEVICES后，总是映射为逻辑GPU 0
    for i, arg in enumerate(base_cmd):
        if arg.startswith("graphics_device_id="):
            base_cmd[i] = f"graphics_device_id={logical_gpu_id}"
            break

    # 包装命令
    if use_xvfb:
        # -a: 自动寻找空闲 server number
        # -s "-screen 0 1024x768x24": 指定屏幕分辨率，防止默认太小
        video_cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24"] + base_cmd
        logging.info("🚀 Using xvfb-run to simulate display for recording.")
    else:
        video_cmd = base_cmd
        logging.info("🚀 Running with local display (headless=False).")

    log_path = workspace_dir / f"vlm_eval_{int(start_time)}.txt"
    logging.info(f"Video recording command: {' '.join(video_cmd)}")
    
    if env is None:
        env = os.environ.copy()
    else:
        env = env.copy()

    env["CUDA_VISIBLE_DEVICES"] = target_gpu_id
    logging.info(f"🔒 Setting CUDA_VISIBLE_DEVICES={target_gpu_id} (Physical GPU {target_gpu_id} will be mapped to cuda:0)")


    # 启动进程
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(video_cmd, stdout=log_file, stderr=log_file, env=env)
    
    # ================= 监控进程 =================
    # 给予充足的时间：启动环境可能很慢 (特别是编译 CUDA kernel 时)
    max_wait_time = max(rollout_steps * 5 + 180, 400) 
    logging.info(f"Waiting for video recording (timeout: {max_wait_time}s)...")
    
    try:
        elapsed = 0
        check_interval = 5
        while elapsed < max_wait_time:
            return_code = process.poll()
            if return_code is not None:
                if return_code != 0:
                    logging.warning(f"❌ Video recording process exited with error code {return_code}. Log: {log_path}")
                    # 读取最后几行日志方便调试
                    try:
                        with open(log_path, 'r') as f:
                            lines = f.readlines()[-10:]
                            logging.warning("Last 10 lines of log:\n" + "".join(lines))
                    except:
                        pass
                break
            
            time.sleep(check_interval)
            elapsed += check_interval
            if elapsed % 30 == 0:
                logging.info(f"Recording in progress... ({elapsed}s)")
        else:
            logging.error(f"❌ Video recording timed out after {max_wait_time}s.")
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

    # ================= 收集视频 =================
    after = _collect_videos(workspace_dir)
    new_videos = {
        Path(path)
        for path, mtime in after.items()
        if (path not in before or mtime > before[path]) and mtime >= start_time
    }
    
    if not new_videos:
        logging.warning(f"⚠️ No new video file found. Check full log at {log_path}")
        return None

    # 找到最新的视频
    latest_video = max(new_videos, key=lambda p: p.stat().st_mtime)
    
    # 验证视频大小，如果是 0KB 说明生成失败
    if latest_video.stat().st_size < 1000:
        logging.warning(f"⚠️ Video file created but empty: {latest_video}")
        return None
        
    logging.info(f"✅ Video successfully recorded: {latest_video}")
    return latest_video