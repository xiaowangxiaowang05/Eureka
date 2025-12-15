import hydra
import numpy as np
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sysconfig
import sys
import time
import concurrent.futures
from http import HTTPStatus
import matplotlib
matplotlib.use("Agg")  # Ensure plotting works without a display
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
import dashscope

# Import original utility functions (keeping original directory structure)
from utils.create_task import create_task
from utils.extract_task_code import file_to_string, get_function_signature
from utils.file_utils import load_tensorboard_logs, filter_traceback
from utils.misc import block_until_training
from utils.video_utils import record_policy_rollout

# Constants
_EUREKA_PACKAGE_DIR = Path(__file__).parent.resolve()
EUREKA_ROOT_DIR = _EUREKA_PACKAGE_DIR.parent.resolve()
ISAAC_ROOT_DIR = EUREKA_ROOT_DIR / "isaacgymenvs" / "isaacgymenvs"

# ==========================================
# 1. Core Utility Functions: Environment
# ==========================================

def get_env_with_python_lib():
    """
    Constructs the subprocess environment, ensuring Python/LD_LIBRARY paths are correct.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    def _append_path(env_key: str, paths: List[str]):
        existing = env.get(env_key, "")
        existing_list = [p for p in existing.split(os.pathsep) if p]
        for p in paths:
            if p and os.path.exists(p) and p not in existing_list:
                existing_list.append(p)
        if existing_list:
            env[env_key] = os.pathsep.join(existing_list)
    
    # Python lib paths (PYTHONPATH)
    python_lib = sysconfig.get_paths().get("purelib")
    site_packages = sysconfig.get_paths().get("platlib")
    _append_path("PYTHONPATH", [python_lib, site_packages, str(EUREKA_ROOT_DIR)])
    
    # Conda / current interpreter lib paths
    current_prefix = sys.prefix
    conda_lib_path = os.path.join(current_prefix, "lib")
    cuda_lib = "/usr/local/cuda/lib64"
    isaac_lib = str(ISAAC_ROOT_DIR / "bindings" / "python")
    _append_path("LD_LIBRARY_PATH", [conda_lib_path, cuda_lib, isaac_lib])
    
    return env

def extract_env_metadata(env_code: str, client: OpenAI, model: str) -> str:
    """
    Extract high-level metadata (robot type, joints, observables) from environment code.
    """
    prompt_path = EUREKA_ROOT_DIR / "eureka/utils/prompts/env_metadata.txt"
    prompt_template = file_to_string(str(prompt_path))
    prompt = prompt_template.replace("{ENV_CODE}", env_code[:6000])
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "ROBOT: Unknown\nJOINTS: Unknown\nOBSERVABLES: Unknown"
# ==========================================
# 2. Data Structures
# ==========================================

@dataclass
class CodeSample:
    code: str
    raw_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingArtifact:
    env_file: Path
    reward_only_file: Path
    log_file: Path
    metrics_summary: Dict[str, Any]
    stats_text: str
    success_metric: float
    reward_correlation: float
    checkpoint_path: Optional[Path] = None
    tensorboard_dir: Optional[Path] = None
    video_path: Optional[Path] = None

@dataclass
class VLMResult:
    visual_score: float
    qualitative_feedback: str
    analysis_notes: Dict[str, str] = field(default_factory=dict)

@dataclass
class PopulationMember:
    generation: int
    index: int
    code_sample: CodeSample
    artifact: Optional[TrainingArtifact] = None
    vlm_result: Optional[VLMResult] = None
    skip_vlm: bool = False
    visual_score: float = 0.0
    reflection_text: str = ""     # Data-based reflection
    final_reflection: str = ""    # Combined reflection (Data + Visual)
    lineage_history: List[str] = field(default_factory=list)

    @property
    def physical_metric(self) -> float:
        if self.artifact is not None and self.artifact.success_metric != float('-inf'):
            return float(self.artifact.success_metric)
        return float('-inf')

    @property
    def combined_score(self) -> float:
        """
        Multiplicative blend of physical and visual scores:
        final = phy * (1 + vis/100). If phy <= 0, return phy to avoid
        negative * boost flipping sign.
        """
        phy = self.physical_metric if self.physical_metric != float('-inf') else -100.0
        vis = self.visual_score
        if phy <= 0:
            return phy
        return phy * (1.0 + vis / 100.0)

# ==========================================
# 3. LLM/VLM Interaction
# ==========================================

def _clean_response_text(raw_response: str) -> str:
    cleaned = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
    return cleaned.replace("<|im_end|>", "").strip()

def _extract_reward_code(raw_response: str) -> Optional[str]:
    content = _clean_response_text(raw_response)
    patterns = [r'```python(.*?)```', r'```(.*?)```', r'"""(.*?)"""']
    code_string = None
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            code_string = match.group(1).strip()
            break
    
    if not code_string:
        code_string = content # Fallback

    lines = code_string.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            return "\n".join(lines[i:])
    
    return code_string if "def compute_reward" in code_string else None

def generate_visual_rubric(
    task_description: str,
    env_code: str,
    env_metadata: str,
    prompt_template: str,
    client: OpenAI,
    model: str,
    temperature: float,
    top_p: float = 1.0
) -> Dict[str, Any]:
    prompt = (
        prompt_template
        .replace("{TASK_DESCRIPTION}", task_description)
        .replace("{ENV_CODE}", env_code)  # Pass full environment code instead of truncating
        .replace("{ENV_METADATA}", env_metadata)
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p
        )
        content = _clean_response_text(response.choices[0].message.content)
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception:
        pass
    
    return {"overall_instruction": "Evaluate if the robot completes the task naturally."}

def construct_reward_reflection(
    training_results: Optional[Dict],
    error: Optional[str] = None,
    policy_feedback_text: str = "",
    code_feedback_text: str = "",
    execution_error_feedback_text: str = "",
) -> str:
    """
    Constructs feedback string by merging Original Eureka logic (detailed stats) 
    with Eureka_new's structure.
    """
    # 1. Handle Execution Errors
    if error:
        return execution_error_feedback_text.format(traceback_msg=error)
    
    if not training_results:
        return execution_error_feedback_text.format(traceback_msg="No training results found.")

    # 2. Handle Success
    metrics = training_results.get("metrics_summary", {})
    
    # Calculate Epoch Frequency
    epoch_freq = 1
    if metrics:
        try:
            sample_key = next(iter(metrics.keys()))
            max_iterations = np.array(metrics[sample_key]).shape[0]
            epoch_freq = max(int(max_iterations // 10), 1)
        except Exception:
            epoch_freq = 1

    # Start with the policy feedback template
    content = policy_feedback_text.format(epoch_freq=epoch_freq)

    # Add reward components log
    for metric in metrics:
        if "/" not in metric: 
            data = metrics[metric]
            if len(data) == 0: continue
            
            # Format list
            metric_cur = ['{:.2f}'.format(x) for x in data[::epoch_freq]]
            metric_cur_max = max(data)
            metric_cur_mean = sum(data) / len(data)
            metric_cur_min = min(data)
            if metric != "gt_reward" and metric != "gpt_reward":
                if metric != "consecutive_successes":
                    metric_name = metric 
                else:
                    metric_name = "task_score"
                content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
            else:
                if "consecutive_successes" not in metrics:
                    content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"

    # Append the generic code feedback instruction
    content += code_feedback_text
    return content

def _call_vlm_api(
    model: str,
    video_path: str,
    rubric_json: Dict,
    env_metadata: str,
    sample_fps: float = 1.0,  # New parameter for VLM sampling rate
    retries: int = 3,
    prompt_template: Optional[str] = None,
    gen_id: int = 0,
    cand_id: int = 0,
    task_description: str = "",
    robot_morphology: str = ""
) -> VLMResult:
    """
    Uploads the ORIGINAL video but instructs the API to sample at `sample_fps`.
    This prevents the "too many frames" error while using the original file.
    """
    log_prefix = f"[Gen {gen_id} Cand {cand_id} VLM]"
    # Check if model is valid
    if not model or model.lower() in ["null", "none", ""]:
        return VLMResult(0, f"{log_prefix} VLM model not configured", {})
    
    # Load VLM evaluation prompt template
    if prompt_template is None:
        prompt_dir = EUREKA_ROOT_DIR / "eureka/utils/prompts"
        prompt_template = file_to_string(str(prompt_dir / "vlm_evaluation.txt"))
    
    # Extract components from rubric
    critical_failures = rubric_json.get("critical_failures", ["Catastrophic failure visible in video"])
    failures_list_str = "\n".join([f"- {item}" for item in critical_failures])
    rubric_criteria = rubric_json.get("criteria", [])
    rubric_criteria_str = json.dumps(rubric_criteria, indent=2)
    rubric_json_str = json.dumps(rubric_json, indent=2)

    # Format the prompt template with actual values
    prompt_text = (
        prompt_template
        .replace("{TASK_DESCRIPTION}", task_description)
        .replace("{ROBOT_MORPHOLOGY}", robot_morphology)
        .replace("{CRITICAL_FAILURES}", failures_list_str)
        .replace("{EVALUATION_CRITERIA}", rubric_criteria_str)
        .replace("{RUBRIC_JSON}", rubric_json_str)
        .replace("{ENV_METADATA}", env_metadata)
    )

    abs_video_path = os.path.abspath(video_path)
    file_url = f"file://{abs_video_path}"

    # --- KEY FIX: Pass FPS to DashScope SDK ---
    # According to DashScope docs, "fps" parameter controls frame sampling.
    content_item = {
        "video": file_url,
        "fps": sample_fps  # Use the config value (e.g., 24 or 2.0)
    }
    
    messages = [
        {
            "role": "user",
            "content": [
                {"text": prompt_text},
                content_item
            ]
        }
    ]

    for attempt in range(retries):
        try:
            response = dashscope.MultiModalConversation.call(
                model=model,
                messages=messages,
                result_format='message',
            )

            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0].message.content
                if isinstance(content, list):
                    text_content = ""
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            text_content += item['text']
                    content = text_content

                content = _clean_response_text(content)
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    
                    # Extract all fields from VLM response (new schema)
                    functional_score = data.get("functional_score", 0.0)
                    motion_quality_score = data.get("motion_quality_score", 0.0)
                    success_detected = data.get("success_detected", False)
                    ugly_behaviors = data.get("ugly_behaviors", [])
                    reward_suggestion = data.get("reward_engineering_suggestion", "")
                    qualitative_feedback = data.get("qualitative_feedback", "")

                    # Use motion_quality_score as visual_score proxy
                    visual_score = float(motion_quality_score if motion_quality_score is not None else 0.0)
                    
                    # Build comprehensive feedback string
                    feedback_str = f"Visual Feedback: {qualitative_feedback}\n"
                    feedback_str += f"Functional Score: {functional_score:.2f}/100\n"
                    feedback_str += f"Motion Quality Score: {motion_quality_score:.2f}/100\n"
                    if success_detected:
                        feedback_str += f"Success Detected: Yes\n"
                    if ugly_behaviors:
                        feedback_str += f"Detected Motion Issues: {', '.join(ugly_behaviors)}\n"
                    if reward_suggestion:
                        feedback_str += f"Suggestion for Improvement: {reward_suggestion}\n"
                    
                    result = VLMResult(
                        visual_score=visual_score,
                        qualitative_feedback=feedback_str.strip(),
                        analysis_notes={
                            "functional_score": functional_score,
                            "motion_quality_score": motion_quality_score,
                            "success_detected": success_detected,
                            "ugly_behaviors": ugly_behaviors,
                            "reward_suggestion": reward_suggestion
                        }
                    )
                    
                    # Log all VLM evaluation scores and data
                    logging.info("=" * 80)
                    logging.info(f"{log_prefix} Video: {os.path.basename(video_path)}")
                    logging.info(f"{log_prefix} ========== SCORES ==========")
                    logging.info(f"{log_prefix} Visual Score (Weighted): {visual_score:.2f}/100")
                    if functional_score is not None:
                        logging.info(f"{log_prefix} Functional Score: {functional_score:.2f}/100")
                    if motion_quality_score is not None:
                        logging.info(f"{log_prefix} Motion Quality Score: {motion_quality_score:.2f}/100")
                    logging.info(f"{log_prefix} Success Detected: {success_detected}")
                    logging.info(f"{log_prefix} ========== DETAILS ==========")
                    if ugly_behaviors:
                        logging.info(f"{log_prefix} Ugly Behaviors: {', '.join(ugly_behaviors)}")
                    if reward_suggestion:
                        logging.info(f"{log_prefix} Reward Suggestion: {reward_suggestion}")
                    logging.info(f"{log_prefix} Qualitative Feedback: {qualitative_feedback}")
                    logging.info(f"{log_prefix} ========== FULL JSON DATA ==========")
                    logging.info(f"{log_prefix} Complete VLM Response JSON:")
                    logging.info(json.dumps(data, indent=2, ensure_ascii=False))
                    logging.info(f"{log_prefix} ========== RAW RESPONSE ==========")
                    # Print full raw VLM response
                    max_line_length = 2000
                    for i in range(0, len(content), max_line_length):
                        chunk = content[i:i+max_line_length]
                        logging.info(f"{log_prefix} Raw Response [part {i//max_line_length + 1}]: {chunk}")
                    logging.info("=" * 80)
                    
                    return result
            else:
                time.sleep(2)

        except Exception:
            time.sleep(2)

    return VLMResult(0, f"{log_prefix} Evaluation Failed", {})

# ==========================================
# 4. Training Management
# ==========================================

@dataclass
class TrainingJob:
    process: subprocess.Popen
    log_file: Path
    candidate_idx: int
    env_metadata: Dict[str, Path]

def _write_candidate_files(gen: int, idx: int, base_code: str, reward_code: str, workspace: Path) -> Dict:
    if "@torch.jit.script" not in reward_code:
        reward_code = "@torch.jit.script\n" + reward_code
        
    signature, _ = get_function_signature(reward_code)
    signature_block = f"""
        self.rew_buf[:], self.rew_dict = {signature}
        self.extras['gpt_reward'] = self.rew_buf.mean()
        for k, v in self.rew_dict.items(): self.extras[k] = v.mean()
    """
    
    task_code = base_code
    for pattern in ["def compute_reward(self):", "def compute_reward(self, actions):"]:
        if pattern in task_code:
            task_code = task_code.replace(pattern, f"{pattern}\n{signature_block}")
            break
    
    env_filename = f"env_gen{gen}_cand{idx}.py"
    with open(workspace / env_filename, 'w') as f:
        f.write(task_code + "\n")
        f.write("from typing import Tuple, Dict\n")
        f.write("import math\n")
        f.write("import torch\n")
        f.write("from torch import Tensor\n")
        f.write(reward_code)
        
    reward_filename = f"env_gen{gen}_cand{idx}_rewardonly.py"
    with open(workspace / reward_filename, 'w') as f:
        f.write(reward_code)
        
    return {"env_file": workspace / env_filename, "reward_only_file": workspace / reward_filename}

def _launch_training(gen: int, idx: int, code: str, base_code: str, cfg, workspace: Path, isaac_dir: Path) -> TrainingJob:
    metadata = _write_candidate_files(gen, idx, base_code, code, workspace)
    log_file = workspace / f"env_gen{gen}_cand{idx}.txt"
    
    # Copy generated code to Isaac Gym task file
    target_task_filename = f"{cfg.env.env_name.lower()}{cfg.suffix.lower()}.py"
    target_task_path = isaac_dir / "tasks" / target_task_filename
    shutil.copy(metadata["env_file"], target_task_path)
    
    gpus = str(cfg.gpu_id).split(",")
    gpu_id = gpus[idx % len(gpus)]
    
    env_vars = get_env_with_python_lib()
    env_vars["CUDA_VISIBLE_DEVICES"] = gpu_id
    env_vars.setdefault("MASTER_PORT", str(29500 + idx + (gen * 100)))

    cmd = [
        "python", "-u", f"{isaac_dir}/train.py",
        "hydra/output=subprocess",
        f"task={cfg.env.task}{cfg.suffix}",
        f"wandb_activate={cfg.use_wandb}",
        f"seed={idx}",
        "headless=True", "capture_video=False", "force_render=False",
        f"max_iterations={cfg.max_iterations}"
    ]
    
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=f, env=env_vars)
        
    # Wait for environment build to prevent race condition
    logging.info(f"Gen {gen} Cand {idx}: Waiting for environment build...")
    block_until_training(str(log_file), log_status=False, iter_num=gen, response_id=idx)
    time.sleep(2)  # Buffer to ensure file handle release
        
    return TrainingJob(proc, log_file, idx, metadata)

def _process_training_result(job: TrainingJob, cfg) -> TrainingArtifact:
    job.process.wait()
    log_content = job.log_file.read_text() if job.log_file.exists() else ""
    
    traceback_msg = filter_traceback(log_content)
    
    tb_line = next((l for l in log_content.split('\n') if "Tensorboard Directory:" in l), None)
    tb_dir = Path(tb_line.split(":")[-1].strip()) if tb_line else None
    
    metrics = {}
    success = float('-inf')
    correlation = float('-inf')
    
    if tb_dir and tb_dir.exists():
        logs = load_tensorboard_logs(str(tb_dir))
        metrics = logs
        if "consecutive_successes" in logs:
            success = max(logs["consecutive_successes"])
        elif "gt_reward" in logs:
            success = max(logs["gt_reward"])
            
        if "gt_reward" in logs and "gpt_reward" in logs:
            try:
                correlation = np.corrcoef(logs["gt_reward"], logs["gpt_reward"])[0, 1]
            except Exception:
                pass
            
    net_line = next((l for l in log_content.split('\n') if "Network Directory:" in l), None)
    ckpt_path = None
    if net_line:
        net_dir = Path(net_line.split(":")[-1].strip())
        ckpts = sorted(net_dir.glob("*.pth"), key=os.path.getmtime)
        if ckpts: ckpt_path = ckpts[-1]

    if tb_dir and tb_dir.exists():
        stats_text = ""
    elif traceback_msg:
        stats_text = f"Runtime Error Captured:\n{traceback_msg}"
    else:
        stats_text = "Training Failed: No metrics and no traceback found."

    return TrainingArtifact(
        env_file=job.env_metadata["env_file"],
        reward_only_file=job.env_metadata["reward_only_file"],
        log_file=job.log_file,
        metrics_summary=metrics,
        stats_text=stats_text,
        success_metric=success,
        reward_correlation=correlation,
        tensorboard_dir=tb_dir,
        checkpoint_path=ckpt_path
    )

# ==========================================
# 5. Core Evaluation Logic
# ==========================================

def _evaluate_population(
    generation: int,
    code_samples: List[CodeSample],
    base_task_code: str,
    cfg,
    workspace_dir: Path,
    isaac_root_dir: Path,
    global_rubric: Dict,
    policy_feedback_text: str,
    code_feedback_text: str,
    execution_error_feedback_text: str,
    env_metadata: str,
) -> Tuple[List[PopulationMember], float]:
    
    # 1. Launch Training
    logging.info(f"Gen {generation}: Training {len(code_samples)} candidates...")
    jobs = []
    for i, sample in enumerate(code_samples):
        try:
            job = _launch_training(generation, i, sample.code, base_task_code, cfg, workspace_dir, isaac_root_dir)
            jobs.append(job)
        except Exception:
            jobs.append(None)
            
    # 2. Wait and Collect Results
    logging.info("Waiting for training...")
    artifacts = []
    reflection_texts = {}
    
    for i, job in enumerate(jobs):
        if not job:
            artifacts.append(None)
            reflection_texts[i] = construct_reward_reflection(
                None, "Launch Failed",
                policy_feedback_text, code_feedback_text, execution_error_feedback_text
            )
            continue

        artifact = _process_training_result(job, cfg)
        artifacts.append(artifact)

        if "Runtime Error" in artifact.stats_text:
            reflection_texts[i] = construct_reward_reflection(
                None, artifact.stats_text,
                policy_feedback_text, code_feedback_text, execution_error_feedback_text
            )
        else:
            res_dict = {"success_metric": artifact.success_metric, "metrics_summary": artifact.metrics_summary}
            reflection_texts[i] = construct_reward_reflection(
                res_dict, None,
                policy_feedback_text, code_feedback_text, execution_error_feedback_text
            )
        artifact.stats_text = reflection_texts[i]

    # 3. Filtering - keep more candidates in early generations
    keep_ratio = 0.75 if generation < 3 else 0.5

    valid_indices = [i for i, a in enumerate(artifacts) if a]

    def sort_key(idx):
        art = artifacts[idx]
        if art.success_metric != float("-inf"):
            return art.success_metric
        if "gpt_reward" in art.metrics_summary:
            return art.metrics_summary.get("gpt_reward", {}).get("mean", float("-inf"))
        return float("-inf")

    valid_indices.sort(key=sort_key, reverse=True)

    cutoff = max(1, int(len(valid_indices) * keep_ratio))
    survivor_indices = set(valid_indices[:cutoff])

    skip_vlm_map = {i: (i not in survivor_indices) for i in range(len(code_samples))}

    logging.info(f"Filter Complete: Keeping Top {len(survivor_indices)}/{len(valid_indices)} (Ratio: {keep_ratio})")

    # 4. Record Videos
    logging.info("Recording videos for survivors...")
    for idx in survivor_indices:
        artifact = artifacts[idx]
        if artifact and artifact.checkpoint_path:
            try:
                vid_path = record_policy_rollout(
                    isaac_root_dir=isaac_root_dir,
                    workspace_dir=workspace_dir,
                    task_name=cfg.env.task,
                    suffix=cfg.suffix,
                    checkpoint_path=artifact.checkpoint_path,
                    wandb_username=cfg.wandb_username,
                    wandb_project=cfg.wandb_project,
                    env=get_env_with_python_lib(),
                    rollout_steps=cfg.video.rollout_len,
                    headless=True, force_render=False, seed=idx, gpu_id=str(cfg.gpu_id)
                )
                
                if vid_path and vid_path.exists():
                    cand_dir = workspace_dir / f"gen{generation}_cand{idx}"
                    cand_dir.mkdir(exist_ok=True, parents=True)
                    
                    target_vid = cand_dir / vid_path.name
                    if target_vid.exists(): target_vid.unlink()
                    shutil.move(str(vid_path), target_vid)
                    
                    artifact.video_path = target_vid
                    shutil.copy2(artifact.reward_only_file, cand_dir / artifact.reward_only_file.name)
            except Exception:
                pass

    # 5. VLM Evaluation
    # Check if VLM model is configured
    if not cfg.model_vlm or cfg.model_vlm.lower() == "null" or cfg.model_vlm.strip() == "":
        vlm_results = {idx: VLMResult(0, "VLM not configured", {}) for idx in survivor_indices}
    else:
        logging.info(f"Starting VLM Evaluation (DashScope sampling at {cfg.video.target_fps} FPS)...")
        vlm_results = {}

        # Load VLM evaluation prompt template
        prompt_dir = EUREKA_ROOT_DIR / "eureka/utils/prompts"
        vlm_prompt_template = file_to_string(str(prompt_dir / "vlm_evaluation.txt"))

        def process_vlm_task(idx, vid_path):
            try:
                result = _call_vlm_api(
                    model=cfg.model_vlm,
                    video_path=str(vid_path),
                    rubric_json=global_rubric,
                    env_metadata=env_metadata,
                    sample_fps=float(cfg.video.target_fps),
                    prompt_template=vlm_prompt_template,
                    gen_id=generation,
                    cand_id=idx,
                    task_description=cfg.env.description,
                    robot_morphology=cfg.env.env_name
                )
                logging.info(f"[VLM Evaluation #{idx}] Completed - Visual Score: {result.visual_score:.2f}")
                return idx, result
            except Exception:
                return idx, VLMResult(0, "Evaluation Failed", {})

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for idx in survivor_indices:
                artifact = artifacts[idx]
                if artifact and artifact.video_path and artifact.video_path.exists():
                    futures.append(executor.submit(process_vlm_task, idx, artifact.video_path))
                else:
                    vlm_results[idx] = VLMResult(0, "No Video File", {})

            for f in concurrent.futures.as_completed(futures):
                idx, res = f.result()
                vlm_results[idx] = res

        # Summary of all VLM evaluations with complete scores
        logging.info("=" * 80)
        logging.info("VLM Evaluation Summary (All Scores):")
        logging.info("-" * 80)
        for idx in sorted(survivor_indices):
            if idx in vlm_results:
                res = vlm_results[idx]
                func_score = res.analysis_notes.get("functional_score") if isinstance(res.analysis_notes, dict) else None
                qual_score = res.analysis_notes.get("motion_quality_score") if isinstance(res.analysis_notes, dict) else None
                if func_score is not None and qual_score is not None:
                    logging.info(f"Candidate #{idx:2d}: Visual={res.visual_score:6.2f} | Functional={func_score:6.2f} | Quality={qual_score:6.2f}")
                else:
                    logging.info(f"Candidate #{idx:2d}: Visual={res.visual_score:6.2f}")
                logging.info(f"  Feedback: {res.qualitative_feedback[:200] if res.qualitative_feedback else 'N/A'}...")
            else:
                logging.info(f"Candidate #{idx:2d}: No result")
        logging.info("=" * 80)

    # 6. Assemble Population
    population = []
    for i, sample in enumerate(code_samples):
        is_skipped = skip_vlm_map.get(i, True)
        vlm_res = vlm_results.get(i)
        
        base_ref = reflection_texts.get(i, "")
        final_ref = base_ref
        
        if vlm_res and not is_skipped:
            final_ref += f"\n\n[Visual Feedback]\n{vlm_res.qualitative_feedback}"
            reward_suggestion = vlm_res.analysis_notes.get("reward_suggestion") if isinstance(vlm_res.analysis_notes, dict) else None
            if reward_suggestion:
                final_ref += f"\n\n[Reward Function Suggestion]\n{reward_suggestion}"
        elif is_skipped:
            final_ref += "\n\n[Visual Feedback]\nSkipped: Performance too low for visual evaluation."

        member = PopulationMember(
            generation=generation, index=i, code_sample=sample, artifact=artifacts[i],
            vlm_result=vlm_res, skip_vlm=is_skipped,
            visual_score=vlm_res.visual_score if vlm_res else 0.0,
            reflection_text=base_ref,
            final_reflection=final_ref
        )
        population.append(member)

    # Logging
    logging.info("=" * 80)
    logging.info(f"Generation {generation}: Training Results Ranking")
    logging.info("-" * 80)
    ranked = sorted(population, key=lambda m: m.combined_score, reverse=True)
    for r, m in enumerate(ranked, 1):
        phy = m.physical_metric if m.physical_metric != float('-inf') else 0.0
        logging.info(
            f"Rank {r:2d}: Cand {m.index:2d} | Score {m.combined_score:8.4f} "
            f"(Phy {phy:8.4f}) | Vis {m.visual_score:5.1f} | SkipVLM={m.skip_vlm}"
        )
    
    # Calculate execution rate (percentage of candidates that executed successfully)
    successful_count = sum(1 for m in population if m.physical_metric != float('-inf'))
    execute_rate = successful_count / len(population) if population else 0.0
    logging.info(f"Generation {generation}: Execution Rate: {execute_rate:.2%} ({successful_count}/{len(population)})")
    
    return population, execute_rate

def _log_population_ranking(population: List[PopulationMember], generation: int):
    logging.info("=" * 80)
    logging.info(f"Generation {generation}: Population Ranking (all candidates)")
    logging.info("-" * 80)
    ranked = sorted(population, key=lambda m: m.combined_score, reverse=True)
    for r, m in enumerate(ranked, 1):
        phy = m.physical_metric if m.physical_metric != float('-inf') else 0.0
        logging.info(
            f"Rank {r:2d}: Cand {m.index:2d} | Score {m.combined_score:8.4f} "
            f"(Phy {phy:8.4f}) | Vis {m.visual_score:5.1f} | SkipVLM={m.skip_vlm}"
        )
    logging.info("-" * 80)

# ==========================================
# 6. Evolution Operators
# ==========================================

def _select_elites(pop: List[PopulationMember], count: int) -> List[PopulationMember]:
    if not pop: return []
    # Select by combined score
    combined_sorted = sorted(pop, key=lambda m: m.combined_score, reverse=True)
    cutoff = max(1, len(pop)//2)
    candidates = combined_sorted[:cutoff]
    return candidates[:count]

def _tournament_select(pop: List[PopulationMember], k: int = 2) -> PopulationMember:
    combined_sorted = sorted(pop, key=lambda m: m.combined_score, reverse=True)
    cutoff = max(1, len(pop)//2)
    pool = combined_sorted[:cutoff]
    candidates = random.sample(pool, min(k, len(pool)))
    return max(candidates, key=lambda m: m.combined_score)

def _spawn_mutation(
    parent: PopulationMember,
    sys_prompt: str,
    task_desc: str,
    env_code: str,
    client: OpenAI,
    model: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Tuple[CodeSample, str]:
    history_text = "\n".join(f"- {h}" for h in parent.lineage_history) if parent.lineage_history else "(empty)"
    
    user_block = f"""{task_desc}

=== ENVIRONMENT CODE ===
{env_code}
=== EVOLUTION HISTORY ===
{history_text}
=== PARENT PERFORMANCE ===
[Fitness: {parent.physical_metric:.2f} | Visual Score: {parent.visual_score:.1f}]

=== PARENT CODE ===
```python
{parent.code_sample.code}
=== FEEDBACK === {parent.final_reflection} === INSTRUCTION === Based on the history and feedback, improve the reward function.

Provide a one-sentence SUMMARY of what you are changing and why.

Provide the complete new python code.
"""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_block},
    ]

    summary_text = "Modified based on feedback"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p
        )
        content = _clean_response_text(resp.choices[0].message.content)

        m_summary = re.search(r"SUMMARY:\\s*(.+)", content, flags=re.IGNORECASE)
        if m_summary and m_summary.group(1).strip():
            summary_text = m_summary.group(1).strip()

        code = _extract_reward_code(content)
        if not code:
            raise ValueError("No code block found in mutation response")

        return CodeSample(code=code, raw_response=content), summary_text
    except Exception:
        return parent.code_sample, f"{summary_text} (fallback: mutation failed)"


def _generate_initial(
    client: OpenAI,
    model: str,
    messages: List[Dict],
    count: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> List[CodeSample]:
    samples: List[CodeSample] = []
    while len(samples) < count:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                n=min(4, count - len(samples)),
                temperature=temperature,
                top_p=top_p,
            )
            for c in resp.choices:
                code = _extract_reward_code(c.message.content)
                if code:
                    samples.append(CodeSample(code, c.message.content))
        except Exception:
            time.sleep(1)
    return samples


# ==========================================
# 7. Main Loop
# ==========================================
@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace = Path.cwd()
    logging.info(f"Workspace: {workspace}")

    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    llm = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    env_name = cfg.env.env_name.lower()
    task_file = (
        EUREKA_ROOT_DIR
        / "eureka/envs"
        / (
            "isaac"
            if (EUREKA_ROOT_DIR / "eureka/envs/isaac" / f"{env_name}.py").exists()
            else "dexterity"
        )
        / f"{env_name}.py"
    )
    obs_file = task_file.parent / f"{env_name}_obs.py"

    task_code = file_to_string(str(task_file)).replace(cfg.env.task, cfg.env.task + cfg.suffix)
    obs_code = file_to_string(str(obs_file))
    shutil.copy(obs_file, "env_init_obs.py")
    create_task(str(ISAAC_ROOT_DIR), cfg.env.task, cfg.env.env_name, cfg.suffix)
    
    # Create the base task Python file that train.py expects to find
    task_python_file = ISAAC_ROOT_DIR / "tasks" / f"{env_name}{cfg.suffix.lower()}.py"
    task_python_file.parent.mkdir(parents=True, exist_ok=True)
    with open(task_python_file, 'w') as f:
        f.write(task_code)
    logging.info(f"Created base task file: {task_python_file}")

    prompt_dir = EUREKA_ROOT_DIR / "eureka/utils/prompts"
    sys_prompt = file_to_string(str(prompt_dir / "initial_system.txt")).format(
        task_reward_signature_string=file_to_string(str(prompt_dir / "reward_signature.txt"))
    ) + file_to_string(str(prompt_dir / "code_output_tip.txt"))
    user_prompt = file_to_string(str(prompt_dir / "initial_user.txt")).format(
        task_obs_code_string=obs_code, task_description=cfg.env.description
    )
    rubric_tmpl = file_to_string(str(prompt_dir / "visual_rubric.txt"))

    logging.info("Extracting Environment Metadata for Generalization...")
    env_metadata = extract_env_metadata(task_code, llm, cfg.model)
    logging.info(f"Environment Metadata:\n{env_metadata}")

    logging.info("Generating Visual Rubric...")
    rubric = generate_visual_rubric(
        cfg.env.description,
        task_code,
        env_metadata,
        rubric_tmpl,
        llm,
        cfg.model,
        cfg.temperature,
        top_p=getattr(cfg, "top_p", 1.0),
    )
    logging.info(f"Rubric: {json.dumps(rubric, indent=2)}")

    # Load feedback templates
    policy_feedback_text = file_to_string(str(prompt_dir / "policy_feedback.txt"))
    code_feedback_text = file_to_string(str(prompt_dir / "code_feedback.txt"))
    execution_error_feedback_text = file_to_string(str(prompt_dir / "execution_error_feedback.txt"))

    pop_size = int(cfg.evolution.population_size)
    gens = int(cfg.evolution.generations)
    best_phy_history: List[float] = []
    best_vis_history: List[float] = []
    execute_rates: List[float] = []

    logging.info("Generating Gen 0...")
    samples = _generate_initial(
        llm,
        cfg.model,
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        pop_size,
        temperature=cfg.temperature,
        top_p=getattr(cfg, "top_p", 1.0),
    )

    population, execute_rate = _evaluate_population(
        0,
        samples,
        task_code,
        cfg,
        workspace,
        ISAAC_ROOT_DIR,
        rubric,
        policy_feedback_text,
        code_feedback_text,
        execution_error_feedback_text,
        env_metadata,
    )
    _log_population_ranking(population, 0)
    init_best = max(population, key=lambda m: m.combined_score)
    best_phy_history.append(
        init_best.physical_metric if init_best.physical_metric != float("-inf") else 0.0
    )
    best_vis_history.append(init_best.visual_score)
    execute_rates.append(execute_rate)

    best_ever = None

    for g in range(1, gens):
        raw_elite_count = int(pop_size * float(cfg.evolution.elite_fraction))
        elite_count = max(1, min(raw_elite_count, pop_size - 1))

        elites = _select_elites(population, elite_count)

        children_codes = []
        children_histories = []
        while len(children_codes) < pop_size - len(elites):
            parent = _tournament_select(population)
            child_code, change_summary = _spawn_mutation(
                parent,
                sys_prompt,
                cfg.env.description,
                task_code,
                llm,
                cfg.model,
                temperature=cfg.temperature,
                top_p=getattr(cfg, "top_p", 1.0),
            )
            children_codes.append(child_code)
            lineage = list(parent.lineage_history)
            lineage.append(f"Gen {g}: {change_summary}")
            children_histories.append(lineage)

        children_pop, execute_rate = _evaluate_population(
            g,
            children_codes,
            task_code,
            cfg,
            workspace,
            ISAAC_ROOT_DIR,
            rubric,
            policy_feedback_text,
            code_feedback_text,
            execution_error_feedback_text,
            env_metadata,
        )
        for i, child in enumerate(children_pop):
            if i < len(children_histories):
                child.lineage_history = children_histories[i]
        
        execute_rates.append(execute_rate)

        new_pop = []
        for e in elites:
            new_pop.append(
                PopulationMember(
                    generation=g,
                    index=len(new_pop),
                    code_sample=e.code_sample,
                    artifact=e.artifact,
                    vlm_result=e.vlm_result,
                    skip_vlm=e.skip_vlm,
                    visual_score=e.visual_score,
                    reflection_text=e.reflection_text,
                    final_reflection=e.final_reflection,
                    lineage_history=list(e.lineage_history),
                )
            )

        new_pop.extend(children_pop)
        population = new_pop

        _log_population_ranking(population, g)

        curr_best = max(population, key=lambda m: m.combined_score)
        if not best_ever or curr_best.combined_score > best_ever.combined_score:
            best_ever = curr_best

        best_phy_history.append(
            curr_best.physical_metric if curr_best.physical_metric != float("-inf") else 0.0
        )
        best_vis_history.append(curr_best.visual_score)

        # Plot metrics after each generation (similar to old code)
        try:
            fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
            x_axis = list(range(len(best_phy_history)))

            axs[0].plot(x_axis, best_phy_history, marker="o")
            axs[0].set_title("Best Physical Metric per Generation")
            axs[0].set_ylabel("Physical Metric")
            axs[0].grid(True, linestyle="--", alpha=0.4)

            axs[1].plot(x_axis, best_vis_history, marker="o", color="orange")
            axs[1].set_title("Best Visual Score per Generation")
            axs[1].set_ylabel("Visual Score")
            axs[1].grid(True, linestyle="--", alpha=0.4)

            axs[2].plot(x_axis, execute_rates, marker="o", color="green")
            axs[2].set_title("Execution Rate per Generation")
            axs[2].set_xlabel("Generation")
            axs[2].set_ylabel("Execution Rate")
            axs[2].set_ylim([0, 1.1])
            axs[2].grid(True, linestyle="--", alpha=0.4)

            plt.tight_layout()
            plt.savefig("summary.png")
            plt.close(fig)
            np.savez(
                "summary.npz",
                max_successes=best_phy_history,
                execute_rates=execute_rates,
                best_vis_history=best_vis_history,
            )
            logging.info(f"Generation {g}: Saved summary plot (summary.png) and data (summary.npz)")
        except Exception:
            pass

    # Final summary plot (includes execution rate)
    try:
        fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        x_axis = list(range(len(best_phy_history)))
        
        axs[0].plot(x_axis, best_phy_history, marker="o")
        axs[0].set_title("Best Physical Metric per Generation")
        axs[0].set_ylabel("Physical Metric")
        axs[0].grid(True, linestyle="--", alpha=0.4)

        axs[1].plot(x_axis, best_vis_history, marker="o", color="orange")
        axs[1].set_title("Best Visual Score per Generation")
        axs[1].set_ylabel("Visual Score")
        axs[1].grid(True, linestyle="--", alpha=0.4)
        
        axs[2].plot(x_axis, execute_rates, marker="o", color="green")
        axs[2].set_title("Execution Rate per Generation")
        axs[2].set_xlabel("Generation")
        axs[2].set_ylabel("Execution Rate")
        axs[2].set_ylim([0, 1.1])
        axs[2].grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig("gen_best_metrics.png")
        plt.close(fig)
        np.savez("gen_best_metrics.npz", 
                best_phy=best_phy_history, 
                best_vis=best_vis_history,
                execute_rates=execute_rates)
        logging.info("Saved final generation metrics plot (gen_best_metrics.png) and data (gen_best_metrics.npz).")
    except Exception:
        pass

    if best_ever and best_ever.artifact:
        res = {
            "code": best_ever.code_sample.code,
            "phy_score": best_ever.physical_metric,
            "vis_score": best_ever.visual_score,
            "feedback": best_ever.final_reflection,
            "video": str(best_ever.artifact.video_path) if best_ever.artifact.video_path else None,
        }
        with open("champion.json", "w") as f:
            json.dump(res, f, indent=2)
        logging.info("Evolution Done. Champion Saved.")


if __name__ == "__main__":
    main()