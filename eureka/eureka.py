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
from typing import Any, Dict, List, Optional, Tuple, Union
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
    
    # === 关键修改开始 ===
    # Conda / 当前解释器的 lib 路径
    current_prefix = sys.prefix
    conda_lib_path = os.path.join(current_prefix, "lib")

    # Torch/Isaac common dependency paths
    cuda_lib = "/usr/local/cuda/lib64"
    isaac_lib = str(ISAAC_ROOT_DIR / "bindings" / "python")

    # 加入 LD_LIBRARY_PATH，确保能找到 libpython 与 Isaac 依赖
    _append_path("LD_LIBRARY_PATH", [conda_lib_path, cuda_lib, isaac_lib])
    # === 关键修改结束 ===
    
    return env

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
    fitness_score: float
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
    prompt_template: str,
    client: OpenAI,
    model: str,
    temperature: float,
    top_p: float = 1.0
) -> Dict[str, Any]:
    prompt = (
        prompt_template
        .replace("{TASK_DESCRIPTION}", task_description)
        .replace("{ENV_CODE}", env_code[:3000])
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
    except Exception as e:
        logging.warning(f"Rubric generation failed: {e}")
    
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
    try:
        if metrics:
            sample_key = next(iter(metrics.keys()))
            max_iterations = np.array(metrics[sample_key]).shape[0]
            epoch_freq = max(int(max_iterations // 10), 1)
        else:
            epoch_freq = 1
    except:
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
    sample_fps: float = 1.0,  # New parameter for VLM sampling rate
    retries: int = 3
) -> VLMResult:
    """
    Uploads the ORIGINAL video but instructs the API to sample at `sample_fps`.
    This prevents the "too many frames" error while using the original file.
    """
    critical_failures = rubric_json.get("critical_failures", ["Catastrophic failure visible in video"])
    failures_list_str = "\n".join([f"- {item}" for item in critical_failures])
    rubric_criteria = rubric_json.get("criteria", [])
    rubric_str = json.dumps(rubric_criteria, indent=2)

    prompt_text = f"""
You are a strict robotics referee. 
Analyze the video content.

### STEP 1: CRITICAL FAILURE CHECK (The "Death" Check)
Scan the video for ANY of the following specific failures defined for this task. 
If ANY occur, the score is ZERO.

{failures_list_str}

### STEP 2: SCORING (Only if no failures)
If and ONLY IF none of the above failures happened, score the performance (0-100) based on:

{rubric_str}

### OUTPUT FORMAT (JSON)
{{
    "critical_failure_detected": true/false,
    "failure_reason": "Describe the failure if true, else null",
    "fitness_score": float (0-100),
    "qualitative_feedback": "Short summary of behavior",
    "analysis_notes": {{"what_went_wrong": "...", "what_went_right": "..."}}
}}
""".strip()

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
                    if data.get("critical_failure_detected", False):
                        data["fitness_score"] = 0.0
                    return VLMResult(
                        fitness_score=float(data.get("fitness_score", 0)),
                        qualitative_feedback=data.get("qualitative_feedback", ""),
                        analysis_notes=data.get("analysis_notes", {})
                    )
            else:
                logging.warning(f"DashScope Error: {response.code} - {response.message}")
                time.sleep(2)

        except Exception as e:
            logging.warning(f"VLM attempt {attempt+1} failed: {e}")
            time.sleep(2)

    return VLMResult(0, "Evaluation Failed", {})

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
    if "def compute_reward(self):" in task_code:
        task_code = task_code.replace("def compute_reward(self):", f"def compute_reward(self):\n{signature_block}")
    elif "def compute_reward(self, actions):" in task_code:
        task_code = task_code.replace("def compute_reward(self, actions):", f"def compute_reward(self, actions):\n{signature_block}")
    
    env_filename = f"env_gen{gen}_cand{idx}.py"
    with open(workspace / env_filename, 'w') as f:
        f.write(task_code + "\n" + reward_code)
        
    reward_filename = f"env_gen{gen}_cand{idx}_rewardonly.py"
    with open(workspace / reward_filename, 'w') as f:
        f.write(reward_code)
        
    return {"env_file": workspace / env_filename, "reward_only_file": workspace / reward_filename}

def _launch_training(gen: int, idx: int, code: str, base_code: str, cfg, workspace: Path, isaac_dir: Path) -> TrainingJob:
    metadata = _write_candidate_files(gen, idx, base_code, code, workspace)
    log_file = workspace / f"env_gen{gen}_cand{idx}.txt"
    
    gpus = str(cfg.gpu_id).split(",")
    gpu_id = gpus[idx % len(gpus)]
    
    env_vars = get_env_with_python_lib()
    env_vars["CUDA_VISIBLE_DEVICES"] = gpu_id
    env_vars.setdefault("MASTER_PORT", str(29500 + idx))

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
            except: pass
            
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
        logging.warning(f"Candidate {job.candidate_idx} failed with error.")
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
    execution_error_feedback_text: str
) -> List[PopulationMember]:
    
    # 1. Launch Training
    logging.info(f"Gen {generation}: Training {len(code_samples)} candidates...")
    jobs = []
    for i, sample in enumerate(code_samples):
        try:
            job = _launch_training(generation, i, sample.code, base_task_code, cfg, workspace_dir, isaac_root_dir)
            jobs.append(job)
        except Exception as e:
            logging.error(f"Launch failed for cand {i}: {e}")
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

        block_until_training(str(job.log_file), log_status=False, iter_num=generation, response_id=i)
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

    # 3. Filtering
    # 在早期世代保留更多候选，防止过早剪枝
    keep_ratio = 0.75 if generation < 3 else 0.5

    valid_indices = [i for i, a in enumerate(artifacts) if a]

    def sort_key(idx):
        art = artifacts[idx]
        if art.success_metric != float("-inf"):
            return art.success_metric
        if "gpt_reward" in art.metrics_summary:
            try:
                return art.metrics_summary["gpt_reward"]["mean"]
            except Exception:
                return float("-inf")
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
            except Exception as e:
                logging.error(f"Recording error cand {idx}: {e}")

    # 5. VLM Evaluation
    logging.info(f"Starting VLM Evaluation (DashScope sampling at {cfg.video.target_fps} FPS)...")
    vlm_results = {}
    
    def process_vlm_task(idx, vid_path):
        try:
            return idx, _call_vlm_api(
                model=cfg.model_vlm,
                video_path=str(vid_path),
                rubric_json=global_rubric,
                sample_fps=float(cfg.video.target_fps) # PASS CONFIG FPS HERE
            )
        except Exception as e:
            return idx, VLMResult(0, f"Error: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for idx in survivor_indices:
            artifact = artifacts[idx]
            if artifact and artifact.video_path and artifact.video_path.exists():
                futures.append(executor.submit(process_vlm_task, idx, artifact.video_path))
            else:
                vlm_results[idx] = VLMResult(0, "No Video File")
        
        for f in concurrent.futures.as_completed(futures):
            idx, res = f.result()
            vlm_results[idx] = res

    # 6. Assemble Population
    population = []
    for i, sample in enumerate(code_samples):
        is_skipped = skip_vlm_map.get(i, True)
        vlm_res = vlm_results.get(i)
        
        base_ref = reflection_texts.get(i, "")
        final_ref = base_ref
        
        if vlm_res and not is_skipped:
            final_ref += f"\n\n[Visual Feedback]\n{vlm_res.qualitative_feedback}"
            if vlm_res.analysis_notes:
                final_ref += f"\nAnalysis: {vlm_res.analysis_notes}"
        elif is_skipped:
            final_ref += "\n\n[Visual Feedback]\nSkipped: Performance too low for visual evaluation."

        member = PopulationMember(
            generation=generation, index=i, code_sample=sample, artifact=artifacts[i],
            vlm_result=vlm_res, skip_vlm=is_skipped,
            visual_score=vlm_res.fitness_score if vlm_res else 0.0,
            reflection_text=base_ref,
            final_reflection=final_ref
        )
        population.append(member)

    # Logging
    logging.info("=" * 80)
    logging.info(f"Generation {generation}: Training Results Ranking")
    logging.info("-" * 80)
    ranked = sorted(
        population,
        key=lambda m: (m.physical_metric if m.physical_metric != float('-inf') else float('-inf'), m.visual_score),
        reverse=True,
    )
    for r, m in enumerate(ranked, 1):
        phy = m.physical_metric if m.physical_metric != float('-inf') else 0.0
        logging.info(f"Rank {r:2d}: Cand {m.index:2d} | Phy {phy:8.4f} | Vis {m.visual_score:5.1f} | SkipVLM={m.skip_vlm}")
    return population

def _log_population_ranking(population: List[PopulationMember], generation: int):
    logging.info("=" * 80)
    logging.info(f"Generation {generation}: Population Ranking (all candidates)")
    logging.info("-" * 80)
    ranked = sorted(
        population,
        key=lambda m: (m.physical_metric if m.physical_metric != float('-inf') else float('-inf'), m.visual_score),
        reverse=True,
    )
    for r, m in enumerate(ranked, 1):
        phy = m.physical_metric if m.physical_metric != float('-inf') else 0.0
        logging.info(f"Rank {r:2d}: Cand {m.index:2d} | Phy {phy:8.4f} | Vis {m.visual_score:5.1f} | SkipVLM={m.skip_vlm}")
    logging.info("-" * 80)

# ==========================================
# 6. Evolution Operators
# ==========================================

def _select_elites(pop: List[PopulationMember], count: int) -> List[PopulationMember]:
    if not pop: return []
    phy_sorted = sorted(pop, key=lambda m: m.physical_metric, reverse=True)
    cutoff = max(1, len(pop)//2)
    candidates = phy_sorted[:cutoff]
    vis_sorted = sorted(candidates, key=lambda m: m.visual_score, reverse=True)
    return vis_sorted[:count]

def _tournament_select(pop: List[PopulationMember], k: int = 2) -> PopulationMember:
    phy_sorted = sorted(pop, key=lambda m: m.physical_metric, reverse=True)
    cutoff = max(1, len(pop)//2)
    pool = phy_sorted[:cutoff]
    candidates = random.sample(pool, min(k, len(pool)))
    return max(candidates, key=lambda m: m.visual_score)

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
    except Exception as e:
        logging.error(f"Mutation failed: {e}")
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
        except Exception as e:
            logging.error(f"Init gen failed: {e}")
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

    prompt_dir = EUREKA_ROOT_DIR / "eureka/utils/prompts"
    sys_prompt = file_to_string(str(prompt_dir / "initial_system.txt")).format(
        task_reward_signature_string=file_to_string(str(prompt_dir / "reward_signature.txt"))
    ) + file_to_string(str(prompt_dir / "code_output_tip.txt"))
    user_prompt = file_to_string(str(prompt_dir / "initial_user.txt")).format(
        task_obs_code_string=obs_code, task_description=cfg.env.description
    )
    rubric_tmpl = file_to_string(str(prompt_dir / "visual_rubric.txt"))

    logging.info("Generating Visual Rubric...")
    rubric = generate_visual_rubric(
        cfg.env.description,
        task_code,
        rubric_tmpl,
        llm,
        cfg.model,
        cfg.temperature,
        top_p=getattr(cfg, "top_p", 1.0),
    )
    logging.info(f"Rubric: {json.dumps(rubric, indent=2)}")

    # 加载原版 Eureka 的三个反馈模板
    try:
        policy_feedback_text = file_to_string(str(prompt_dir / "policy_feedback.txt"))
        code_feedback_text = file_to_string(str(prompt_dir / "code_feedback.txt"))
        execution_error_feedback_text = file_to_string(str(prompt_dir / "execution_error_feedback.txt"))
    except Exception as e:
        logging.error(f"Failed to load feedback prompts from {prompt_dir}: {e}")
        raise e

    pop_size = int(cfg.evolution.population_size)
    gens = int(cfg.evolution.generations)
    best_phy_history: List[float] = []
    best_vis_history: List[float] = []

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

    population = _evaluate_population(
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
    )
    _log_population_ranking(population, 0)
    init_best = max(population, key=lambda m: (m.physical_metric, m.visual_score))
    best_phy_history.append(
        init_best.physical_metric if init_best.physical_metric != float("-inf") else 0.0
    )
    best_vis_history.append(init_best.visual_score)

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

        children_pop = _evaluate_population(
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
        )
        for i, child in enumerate(children_pop):
            if i < len(children_histories):
                child.lineage_history = children_histories[i]

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

        curr_best = max(population, key=lambda m: (m.physical_metric, m.visual_score))
        if not best_ever or curr_best.physical_metric > best_ever.physical_metric:
            best_ever = curr_best

        best_phy_history.append(
            curr_best.physical_metric if curr_best.physical_metric != float("-inf") else 0.0
        )
        best_vis_history.append(curr_best.visual_score)

    try:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        x_axis = list(range(len(best_phy_history)))
        axs[0].plot(x_axis, best_phy_history, marker="o")
        axs[0].set_title("Best Physical Metric per Generation")
        axs[0].set_ylabel("Physical Metric")
        axs[0].grid(True, linestyle="--", alpha=0.4)

        axs[1].plot(x_axis, best_vis_history, marker="o", color="orange")
        axs[1].set_title("Best Visual Score per Generation")
        axs[1].set_xlabel("Generation")
        axs[1].set_ylabel("Visual Score")
        axs[1].grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig("gen_best_metrics.png")
        plt.close(fig)
        np.savez("gen_best_metrics.npz", best_phy=best_phy_history, best_vis=best_vis_history)
        logging.info("Saved generation best metrics plot/gen_best_metrics.png and npz.")
    except Exception as e:
        logging.warning(f"Failed to save generation metrics plot: {e}")

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