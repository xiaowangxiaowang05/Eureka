import os
import subprocess
import sys

# ================= 配置区域 =================
ISAAC_GYM_ENVS_PATH = "/home/wangyuhang/python/Eureka/isaacgymenvs/isaacgymenvs"
CHECKPOINT_PATH = "Eureka/eureka/outputs/eureka/2025-12-06_17-20-02/policy-2025-12-06_17-20-22/runs/ShadowHandQwen-2025-12-06_17-20-22/nn/last_ShadowHandQwen_ep_30.pth"
TASK_NAME = "ShadowHand"
# ===========================================

def test_recording():
    # 检查 xvfb (虽然我们要切回 headless=True，但保留 xvfb 作为安全网有时更好，也可以去掉)
    try:
        subprocess.run(["which", "xvfb-run"], check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("❌ 错误: 未检测到 xvfb-run")
        sys.exit(1)

    train_script = os.path.join(ISAAC_GYM_ENVS_PATH, "train.py")
    
    # ---------------- 核心修改 1: 清理环境变量 ----------------
    env = os.environ.copy()
    # ⚠️ 删除了 VK_ICD_FILENAMES 的强制指定，让系统自动寻找 NVIDIA 驱动
    # 只有当你 100% 确定你在用 Windows WSL2 时才需要那行代码
    
    # 尝试解决库冲突 (可选，如果还是崩，取消注释下面这行)
    # env["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

    cmd = [
        "xvfb-run", "-a", 
        "python", train_script,
        f"task={TASK_NAME}",
        
        # ---------------- 核心修改 2: 参数调整 ----------------
        "headless=False",                # 改回 True，这是服务器运行最稳定的模式
        "force_render=Ture",           # 设为 False，避免强制弹窗逻辑
        
        "capture_video=True",           # 开启录像
        "capture_video_freq=1",
        "capture_video_len=100",
        
        # 🌟 关键参数：强制开启相机传感器 🌟
        # 即使 headless=True，这也会告诉物理引擎准备图像数据
        "task.env.enableCameraSensors=True", 
        
        "graphics_device_id=0",         # 必须设为 0 (对应你的 cuda:0)
        # ----------------------------------------------------
        
        "test=True",
        "num_envs=1",
        "hydra/output=subprocess",
    ]

    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print(f"📦 使用模型: {CHECKPOINT_PATH}")
        cmd.append(f"checkpoint={CHECKPOINT_PATH}")
    else:
        print("⚠️ 使用随机策略运行")

    print("\n🚀 开始执行测试命令...")
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=sys.stdout, 
            stderr=sys.stderr,
            env=env,
            text=True
        )
        process.wait()
    except KeyboardInterrupt:
        print("\n🛑 用户停止")

if __name__ == "__main__":
    test_recording()