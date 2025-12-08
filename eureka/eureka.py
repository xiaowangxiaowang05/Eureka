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
import time
import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from utils.create_task import create_task
from utils.extract_task_code import *
from utils.file_utils import find_files_with_substring, load_tensorboard_logs
from utils.misc import *
from utils.video_utils import record_policy_rollout
from utils.vlm_utils import VLMClient, VLMResult

_EUREKA_PACKAGE_DIR = Path(__file__).parent.resolve()
EUREKA_ROOT_DIR = _EUREKA_PACKAGE_DIR.parent.resolve()
ISAAC_ROOT_DIR = EUREKA_ROOT_DIR / "isaacgymenvs" / "isaacgymenvs"

def get_env_with_python_lib():
    """Prepare environment variables with Python library path and CUDA paths for subprocess calls."""
    env = os.environ.copy()
    lib_paths = []
    python_lib_dir = None
    lib_dir = sysconfig.get_config_var('LIBDIR')
    if lib_dir and os.path.exists(lib_dir):
        python_lib_dir = lib_dir
    else:
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            conda_lib = os.path.join(conda_prefix, 'lib')
            if os.path.exists(conda_lib):
                python_lib_dir = conda_lib
    if python_lib_dir:
        lib_paths.append(python_lib_dir)
    cuda_paths = []
    possible_cuda_paths = [
        '/usr/lib/wsl/lib',
        '/usr/local/cuda/lib64',
        '/usr/local/cuda/lib',
        '/usr/lib/x86_64-linux-gnu',
        '/usr/lib64',
    ]
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_cuda_lib = os.path.join(conda_prefix, 'lib')
        if os.path.exists(conda_cuda_lib):
            possible_cuda_paths.insert(0, conda_cuda_lib)
    existing_ld_path = env.get('LD_LIBRARY_PATH', '')
    if existing_ld_path:
        for path in existing_ld_path.split(':'):
            if path and os.path.exists(path):
                if 'cuda' in path.lower() or any(lib in os.listdir(path) if os.path.isdir(path) else False 
                                                  for lib in ['libcuda.so', 'libcudart.so']):
                    cuda_paths.append(path)
    for cuda_path in possible_cuda_paths:
        if os.path.exists(cuda_path):
            try:
                files = os.listdir(cuda_path)
                has_libcuda = any('libcuda.so' in f for f in files)
                has_cuda_libs = any('cuda' in f.lower() or 'cudart' in f.lower() for f in files)
                if has_libcuda or has_cuda_libs:
                    if cuda_path not in cuda_paths:
                        if has_libcuda:
                            cuda_paths.insert(0, cuda_path)
                        else:
                            cuda_paths.append(cuda_path)
            except:
                pass
    all_paths = cuda_paths + lib_paths
    if existing_ld_path:
        existing_paths = [p for p in existing_ld_path.split(':') if p and p not in all_paths]
        all_paths.extend(existing_paths)
    if all_paths:
        env['LD_LIBRARY_PATH'] = ':'.join(all_paths)
    return env

MUTATION_PROMPT_TEMPLATE = """
You are executing an Evolutionary Algorithm.

Parent Code (Score: {Fitness_Score}):
{Parent_Reward_Code}

VLM Feedback: "{VLM_Feedback_Text}"
Reward Stats: "{Component_Stats_Analysis}"

Task: The VLM pointed out the defect: "{VLM_Feedback_Text}".
Please modify the parent code to specifically address this defect.
"""

CROSSOVER_PROMPT_TEMPLATE = """
You are executing an Evolutionary Algorithm.

Parent A (Score {Score_A}): "{Feedback_A}"
Code A:
{Code_A}

Parent B (Score {Score_B}): "{Feedback_B}"
Code B:
{Code_B}

Task: Combine the strengths of Code A and Code B into a new child reward function.
"""

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
    tensorboard_dir: Optional[Path]
    network_dir: Optional[Path]
    checkpoint_path: Optional[Path]
    metrics_summary: Dict[str, Any]
    stats_text: str
    success_metric: float
    reward_correlation: float
    video_path: Optional[Path] = None

@dataclass
class PopulationMember:
    generation: int
    index: int
    code_sample: CodeSample
    artifact: Optional[TrainingArtifact] = None
    vlm_result: Optional[VLMResult] = None

    @property
    def fitness(self) -> float:
        if self.vlm_result is not None:
            return float(self.vlm_result.fitness_score)
        if self.artifact is not None and self.artifact.success_metric != float('-inf'):
            return float(self.artifact.success_metric)
        return float('-inf')

def _clean_response_text(raw_response: str) -> str:
    cleaned_text = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
    return cleaned_text.replace("<|im_end|>", "").strip()

def _extract_reward_code(raw_response: str) -> Optional[str]:
    response_cur = _clean_response_text(raw_response)
    patterns = [
        r'```python(.*?)```',
        r'```(.*?)```',
        r'"""(.*?)"""',
        r"'''\s*(.*?)\s*'''",
        r'""(.*?)""',
        r'"(.*?)"',
    ]
    code_string = None
    for pattern in patterns:
        match = re.search(pattern, response_cur, re.DOTALL)
        if match is not None:
            code_string = match.group(1).strip()
            break
    if code_string is None:
        code_string = response_cur
    lines = code_string.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            code_string = "\n".join(lines[i:])
            break
    if not code_string.strip():
        return None
    return code_string

def _inject_reward_signature(task_code_template: str, reward_signature: str) -> str:
    indent = " " * 8
    reward_block = "\n".join([indent + line for line in reward_signature.splitlines()])
    if "def compute_reward(self):" in task_code_template:
        needle = "def compute_reward(self):"
        return task_code_template.replace(needle, f"{needle}\n{reward_block}", 1)
    if "def compute_reward(self, actions):" in task_code_template:
        needle = "def compute_reward(self, actions):"
        return task_code_template.replace(needle, f"{needle}\n{reward_block}", 1)
    if "def compute_reward(" in task_code_template:
         pass
    return task_code_template

def _write_candidate_files(
    *,
    generation: int,
    candidate_idx: int,
    base_task_code: str,
    reward_code: str,
    reward_signature: str,
    output_file: str,
    workspace_dir: Path,
) -> Dict[str, Path]:
    task_code_string_iter = _inject_reward_signature(base_task_code, reward_signature)
    if "@torch.jit.script" not in reward_code:
        reward_code = "@torch.jit.script\n" + reward_code
    task_code_string_iter = task_code_string_iter.replace(
        "def compute_reward(self):", 
        "def compute_reward(self, actions):" 
    )

    with open(output_file, "w") as file:
        file.writelines(task_code_string_iter + "\n")
        file.writelines("from typing import Tuple, Dict\n")
        file.writelines("import math\n")
        file.writelines("import torch\n")
        file.writelines("from torch import Tensor\n")
        file.writelines(reward_code + "\n")

    env_file = workspace_dir / f"env_gen{generation}_cand{candidate_idx}.py"
    reward_only_file = workspace_dir / f"env_gen{generation}_cand{candidate_idx}_rewardonly.py"
    shutil.copy(output_file, env_file)
    with open(reward_only_file, "w") as file:
        file.writelines(reward_code + "\n")

    return {
        "env_file": env_file,
        "reward_only_file": reward_only_file,
    }

def _find_line_value(log_text: str, prefix: str) -> Optional[str]:
    for line in log_text.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[-1].strip()
    return None

def _summarize_tensorboard_logs(
    tensorboard_logs: Dict[str, List[float]],
    policy_feedback_template: str,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "stats_text": "",
        "success_metric": float("-inf"),
        "reward_correlation": float('-inf'),
    }
    if not tensorboard_logs:
        return summary

    metrics_text: List[str] = []
    metric_lengths = [len(values) for values in tensorboard_logs.values() if values]
    total_steps = max(metric_lengths) if metric_lengths else 1
    epoch_freq = max(int(total_steps // 10), 1)
    metrics_text.append(policy_feedback_template.format(epoch_freq=epoch_freq))

    gt_reward = tensorboard_logs.get("gt_reward")
    gpt_reward = tensorboard_logs.get("gpt_reward")
    consecutive_successes = tensorboard_logs.get("consecutive_successes")
    if consecutive_successes:
        summary["success_metric"] = max(consecutive_successes)
    elif gt_reward:
        summary["success_metric"] = max(gt_reward)

    if gt_reward and gpt_reward:
        try:
            summary["reward_correlation"] = float(np.corrcoef(np.array(gt_reward), np.array(gpt_reward))[0, 1])
        except Exception:
            summary["reward_correlation"] = float('-inf')

    for metric, values in tensorboard_logs.items():
        if not values:
            continue
        sampled = values[::epoch_freq]
        metric_cur = [f"{x:.2f}" for x in sampled]
        metric_max = max(values)
        metric_mean = sum(values) / len(values)
        metric_min = min(values)
        metric_name = metric if metric != "consecutive_successes" else "task_score"
        metrics_text.append(
            f"{metric_name}: {metric_cur}, Max: {metric_max:.2f}, Mean: {metric_mean:.2f}, Min: {metric_min:.2f}"
        )

    summary["stats_text"] = "\n".join(metrics_text)
    return summary

def _locate_checkpoint(network_dir: Optional[str]) -> Optional[Path]:
    if not network_dir:
        return None
    network_path = Path(network_dir)
    if not network_path.exists():
        return None
    checkpoints = sorted(network_path.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


@dataclass
class TrainingJob:
    process: subprocess.Popen
    log_file: Path
    candidate_idx: int
    env_metadata: Dict[str, Path]
    start_time: float

def _launch_candidate_training(
    *,
    generation: int,
    candidate_idx: int,
    code_sample: CodeSample,
    base_task_code: str,
    output_file: str,
    workspace_dir: Path,
    isaac_root_dir: str,
    task: str,
    suffix: str,
    cfg,
) -> TrainingJob:
    log_file = workspace_dir / f"env_gen{generation}_cand{candidate_idx}.txt"
    
    try:
        result = get_function_signature(code_sample.code)
        if result is None:
            raise ValueError("No function definition found in reward code")
        reward_signature, _ = result
    except Exception as exc:
        raise ValueError(f"Failed to parse reward signature: {exc}") from exc

    env_metadata = _write_candidate_files(
        generation=generation,
        candidate_idx=candidate_idx,
        base_task_code=base_task_code,
        reward_code=code_sample.code,
        reward_signature="\n".join(
            [
                f"self.rew_buf[:], self.rew_dict = {reward_signature}",
                "self.extras['gpt_reward'] = self.rew_buf.mean()",
                "for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
            ]
        ),
        output_file=output_file,
        workspace_dir=workspace_dir,
    )

    env_vars = get_env_with_python_lib()
    env_vars["PYTHONUNBUFFERED"] = "1"

    gpu_list = [g.strip() for g in str(cfg.gpu_id).split(",")]
    num_gpus = len(gpu_list)
    
    # 为每个candidate分配单个GPU（轮询分配）
    assigned_gpu_idx = candidate_idx % num_gpus
    assigned_gpu_id = gpu_list[assigned_gpu_idx]
    env_vars["CUDA_VISIBLE_DEVICES"] = assigned_gpu_id
    
    # 为每个candidate分配不同的端口，避免冲突
    base_port = 29500
    master_port = base_port + candidate_idx

    base_args = [
        f"{isaac_root_dir}/train.py",
        "hydra/output=subprocess",
        f"task={task}{suffix}",
        f"wandb_activate={cfg.use_wandb}",
        f"wandb_entity={cfg.wandb_username}",
        f"wandb_project={cfg.wandb_project}",
        "headless=True",
        "capture_video=False",
        "force_render=False",
        f"max_iterations={cfg.max_iterations}",
        "pipeline=gpu",
        f"seed={candidate_idx}",
        "graphics_device_id=0",
        "sim_device=cuda:0",
        "rl_device=cuda:0",
    ]

    # 每个candidate使用单GPU训练
    cmd = ["python", "-u"] + base_args
    logging.info(f"Candidate {candidate_idx}: Using single GPU {assigned_gpu_id} (port {master_port})")

    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=f, env=env_vars)

    block_until_training(str(log_file), log_status=False, iter_num=generation, response_id=candidate_idx)
    
    logging.info(f"Launched training for Gen {generation} Candidate {candidate_idx}")
    return TrainingJob(
        process=process,
        log_file=log_file,
        candidate_idx=candidate_idx,
        env_metadata=env_metadata,
        start_time=time.time()
    )

def _harvest_training_artifact(
    *,
    job: TrainingJob,
    policy_feedback_template: str,
    execution_error_feedback: str,
) -> TrainingArtifact:
    job.process.wait()
    stdout_str = job.log_file.read_text() if job.log_file.exists() else ""
    traceback_msg = filter_traceback(stdout_str)

    tensorboard_dir = _find_line_value(stdout_str, "Tensorboard Directory:")
    network_dir = _find_line_value(stdout_str, "Network Directory:")
    tensorboard_logs: Dict[str, List[float]] = {}
    stats_text = ""
    success_metric = float("-inf")
    reward_correlation = float('-inf')

    if traceback_msg:
        stats_text = execution_error_feedback.format(traceback_msg=traceback_msg)
        logging.warning(f"Candidate {job.candidate_idx} execution error.")
    elif tensorboard_dir:
        try:
            tensorboard_logs = load_tensorboard_logs(tensorboard_dir)
            summary = _summarize_tensorboard_logs(tensorboard_logs, policy_feedback_template)
            stats_text = summary["stats_text"]
            success_metric = summary["success_metric"]
            reward_correlation = summary["reward_correlation"]
            logging.info(f"Candidate {job.candidate_idx} training success. Metric: {success_metric}")
        except Exception as exc:
            logging.warning(f"Failed to load tensorboard logs for candidate {job.candidate_idx}: {exc}")
            stats_text = f"Tensorboard parsing failed: {exc}"
    else:
        stats_text = "Training log missing Tensorboard Directory."

    checkpoint_path = _locate_checkpoint(network_dir)

    return TrainingArtifact(
        env_file=job.env_metadata["env_file"],
        reward_only_file=job.env_metadata["reward_only_file"],
        log_file=job.log_file,
        tensorboard_dir=Path(tensorboard_dir) if tensorboard_dir else None,
        network_dir=Path(network_dir) if network_dir else None,
        checkpoint_path=checkpoint_path,
        metrics_summary=tensorboard_logs,
        stats_text=stats_text,
        success_metric=success_metric,
        reward_correlation=reward_correlation,
    )

def _evaluate_population(
    *,
    generation: int,
    code_samples: List[CodeSample],
    base_task_code: str,
    output_file: str,
    workspace_dir: Path,
    isaac_root_dir: str,
    task: str,
    suffix: str,
    cfg,
    policy_feedback_template: str,
    execution_error_feedback: str,
    vlm_client: VLMClient,
    ) -> List[PopulationMember]:
    
    logging.info(f"Starting parallel training for {len(code_samples)} candidates...")
    running_jobs: List[Optional[TrainingJob]] = []
    
    for idx, sample in enumerate(code_samples):
        try:
            job = _launch_candidate_training(
                generation=generation,
                candidate_idx=idx,
                code_sample=sample,
                base_task_code=base_task_code,
                output_file=output_file,
                workspace_dir=workspace_dir,
                isaac_root_dir=isaac_root_dir,
                task=task,
                suffix=suffix,
                cfg=cfg,
            )
            running_jobs.append(job)
        except Exception as exc:
            logging.exception(f"Failed to launch training for candidate {idx}: {exc}")
            running_jobs.append(None)

    logging.info("Waiting for all training jobs to finish...")
    artifacts: List[Optional[TrainingArtifact]] = []
    
    for i, job in enumerate(running_jobs):
        if job is None:
            artifacts.append(None)
            continue
        
        try:
            artifact = _harvest_training_artifact(
                job=job,
                policy_feedback_template=policy_feedback_template,
                execution_error_feedback=execution_error_feedback
            )
            artifacts.append(artifact)
        except Exception as exc:
            logging.exception(f"Error collecting results for candidate {i}: {exc}")
            artifacts.append(None)

    logging.info("Training phase completed. Starting evaluation phase...")
    for idx, artifact in enumerate(artifacts):
        if artifact and artifact.checkpoint_path and cfg.capture_video:
            try:
                logging.info(f"[{idx+1}/{len(artifacts)}] Recording video for Candidate {idx}...")
                video_path = record_policy_rollout(
                    isaac_root_dir=Path(isaac_root_dir),
                    workspace_dir=workspace_dir,
                    task_name=task,
                    suffix=suffix,
                    checkpoint_path=artifact.checkpoint_path,
                    wandb_username=cfg.wandb_username,
                    wandb_project=cfg.wandb_project,
                    env=get_env_with_python_lib(),
                    rollout_steps=cfg.video.rollout_len,
                    headless=cfg.video.headless, 
                    force_render=cfg.video.force_render,
                    seed=idx,
                    gpu_id=str(cfg.gpu_id),
                )
                
                if video_path and video_path.exists():
                    artifact.video_path = video_path
                else:
                    logging.warning(f"Video recording failed for candidate {idx}")
                    artifact.video_path = None
            except Exception as exc:
                logging.warning(f"Exception recording candidate {idx}: {exc}")
                artifact.video_path = None
        else:
            if artifact:
                artifact.video_path = None

    if cfg.capture_video:
        logging.info("Video recording completed. Starting parallel VLM evaluation...")
    
    vlm_results_map: Dict[int, VLMResult] = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(code_samples))) as executor:
        future_to_idx = {}
        for idx, artifact in enumerate(artifacts):
            if artifact and artifact.video_path:
                if not artifact.video_path.exists():
                    logging.warning(f"Video file not found for candidate {idx} at saved path: {artifact.video_path}")
                    continue
                logging.info(f"Submitting video to VLM for candidate {idx}: {artifact.video_path}")
                future = executor.submit(
                    vlm_client.evaluate, 
                    str(artifact.video_path),
                    extra_prompt=artifact.stats_text,
                    max_retries=cfg.vlm.max_retries
                )
                future_to_idx[future] = idx
            else:
                logging.info(f"Skipping VLM for candidate {idx} (no video path saved).")

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                vlm_results_map[idx] = result
                logging.info(f"VLM evaluated candidate {idx}: Score {result.fitness_score}")
            except Exception as exc:
                error_msg = str(exc)
                if len(error_msg) > 500:
                    error_msg = error_msg[:500] + "..."
                logging.error(f"VLM evaluation failed for candidate {idx}: {error_msg}")
                logging.debug(f"Full exception for candidate {idx}:", exc_info=True)

    population: List[PopulationMember] = []
    for idx, sample in enumerate(code_samples):
        member = PopulationMember(
            generation=generation,
            index=idx,
            code_sample=sample,
            artifact=artifacts[idx] if idx < len(artifacts) else None,
            vlm_result=vlm_results_map.get(idx)
        )
        population.append(member)

    return population

def _select_elites(population: List[PopulationMember], elite_count: int) -> List[PopulationMember]:
    if elite_count <= 0:
        return []
    return sorted(population, key=lambda member: member.fitness, reverse=True)[:elite_count]

def _tournament_select(population: List[PopulationMember], tournament_size: int) -> PopulationMember:
    competitors = random.sample(population, k=min(tournament_size, len(population)))
    return max(competitors, key=lambda member: member.fitness)

def _spawn_mutation_child(
    parent: PopulationMember,
    *,
    system_prompt: str,
    llm_client: OpenAI,
    model: str,
    temperature: float,
) -> CodeSample:
    prompt = MUTATION_PROMPT_TEMPLATE.format(
        Fitness_Score=f"{parent.fitness:.2f}",
        Parent_Reward_Code=parent.code_sample.code,
        VLM_Feedback_Text=parent.vlm_result.qualitative_feedback if parent.vlm_result else "No VLM feedback available.",
        Component_Stats_Analysis=parent.artifact.stats_text if parent.artifact else "No stats available.",
    )
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        n=1,
    )
    content = response.choices[0].message.content
    code = _extract_reward_code(content) or ""
    return CodeSample(code=code, raw_response=content, metadata={"prompt": "mutation"})

def _spawn_crossover_child(
    parent_a: PopulationMember,
    parent_b: PopulationMember,
    *,
    system_prompt: str,
    llm_client: OpenAI,
    model: str,
    temperature: float,
) -> CodeSample:
    prompt = CROSSOVER_PROMPT_TEMPLATE.format(
        Score_A=f"{parent_a.fitness:.2f}",
        Feedback_A=parent_a.vlm_result.qualitative_feedback if parent_a.vlm_result else "No feedback.",
        Code_A=parent_a.code_sample.code,
        Score_B=f"{parent_b.fitness:.2f}",
        Feedback_B=parent_b.vlm_result.qualitative_feedback if parent_b.vlm_result else "No feedback.",
        Code_B=parent_b.code_sample.code,
    )
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        n=1,
    )
    content = response.choices[0].message.content
    code = _extract_reward_code(content) or ""
    return CodeSample(code=code, raw_response=content, metadata={"prompt": "crossover"})

def _generate_llm_samples(
    *,
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    sample_count: int,
    temperature: float,
) -> List[CodeSample]:
    responses = []
    total_samples = 0
   
    chunk_size = sample_count if "gpt-3.5" in model else min(4, sample_count)

    while total_samples < sample_count:
        batch_size = min(chunk_size, sample_count - total_samples)
        for attempt in range(100):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    n=batch_size,
                )
                responses.extend(response.choices)
                total_samples += batch_size
                break
            except Exception as exc:
                logging.warning("LLM sampling attempt %s failed: %s", attempt + 1, exc)
                time.sleep(1)
        else:
            raise RuntimeError("Exceeded maximum retries while sampling from the LLM.")

    code_samples: List[CodeSample] = []
    for choice in responses[:sample_count]:
        content = choice.message.content
        code = _extract_reward_code(content) or ""
        code_samples.append(CodeSample(code=code, raw_response=content, metadata={"prompt": "initial"}))
    return code_samples

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")

    llm_client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=300.0,
    )

    vlm_api_client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    env_name = cfg.env.env_name.lower()
    envs_isaac_dir = EUREKA_ROOT_DIR / "eureka" / "envs" / "isaac"
    if envs_isaac_dir.exists() and f'{env_name}.py' in [f.name for f in envs_isaac_dir.iterdir() if f.is_file()]:
        env_parent = 'isaac'
    else:
        env_parent = 'dexterity'
    task_file = EUREKA_ROOT_DIR / "eureka" / "envs" / env_parent / f"{env_name}.py"
    task_obs_file = EUREKA_ROOT_DIR / "eureka" / "envs" / env_parent / f"{env_name}_obs.py"
    
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    if not task_obs_file.exists():
        raise FileNotFoundError(f"Task observation file not found: {task_obs_file}")
    
    shutil.copy(str(task_obs_file), "env_init_obs.py")
    task_code_string  = file_to_string(str(task_file))
    task_obs_code_string  = file_to_string(str(task_obs_file))
    output_file = str(ISAAC_ROOT_DIR / "tasks" / f"{env_name}{suffix.lower()}.py")

    prompt_dir = EUREKA_ROOT_DIR / "eureka" / "utils" / "prompts"
    if not prompt_dir.exists():
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")
    
    initial_system = file_to_string(str(prompt_dir / "initial_system.txt"))
    code_output_tip = file_to_string(str(prompt_dir / "code_output_tip.txt"))
    initial_user = file_to_string(str(prompt_dir / "initial_user.txt"))
    reward_signature = file_to_string(str(prompt_dir / "reward_signature.txt"))
    policy_feedback = file_to_string(str(prompt_dir / "policy_feedback.txt"))
    execution_error_feedback = file_to_string(str(prompt_dir / "execution_error_feedback.txt"))

    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    task_code_template = task_code_string.replace(task, task + suffix)
    create_task(str(ISAAC_ROOT_DIR), cfg.env.task, cfg.env.env_name, suffix)

    evo_cfg = cfg.evolution
    generations = max(1, int(evo_cfg.generations))
    population_size = max(1, int(evo_cfg.population_size))
    tournament_size = max(2, int(evo_cfg.tournament_size))
    elite_fraction = float(evo_cfg.elite_fraction)
    elite_count = max(1, int(round(population_size * elite_fraction)))

    model_vlm = cfg.model_vlm if cfg.model_vlm is not None else "mock"
    vlm_client = VLMClient(
        model_name=model_vlm,
        task_description=task_description,
        prompt_dir=prompt_dir,
        openai_client=None if model_vlm and str(model_vlm).lower() == "mock" else vlm_api_client,
    )

    logging.info("Generation 0: sampling %d candidates", population_size)
    initial_samples = _generate_llm_samples(
        client=llm_client,
        model=model,
        messages=messages,
        sample_count=population_size,
        temperature=cfg.temperature,
    )
    population = _evaluate_population(
        generation=0,
        code_samples=initial_samples,
        base_task_code=task_code_template,
        output_file=output_file,
        workspace_dir=workspace_dir,
        isaac_root_dir=str(ISAAC_ROOT_DIR),
        task=task,
        suffix=suffix,
        cfg=cfg,
        policy_feedback_template=policy_feedback,
        execution_error_feedback=execution_error_feedback,
        vlm_client=vlm_client,
    )

    best_member = max(population, key=lambda member: member.fitness, default=None)
    if best_member:
        logging.info(
            "Generation 0 best fitness %.2f (score=%s)",
            best_member.fitness,
            best_member.vlm_result.fitness_score if best_member.vlm_result else "n/a",
        )

    for gen in range(1, generations):
        if not population:
            break

        elites = _select_elites(population, min(elite_count, population_size))
        children_needed = max(0, population_size - len(elites))
        mutation_quota = min(children_needed, int(round(children_needed * evo_cfg.mutation_ratio)))
        crossover_quota = max(0, children_needed - mutation_quota)

        children_samples: List[CodeSample] = []
        for _ in range(mutation_quota):
            parent = _tournament_select(population, tournament_size)
            children_samples.append(
                _spawn_mutation_child(
                    parent,
                    system_prompt=initial_system,
                    llm_client=llm_client,
                    model=model,
                    temperature=cfg.temperature,
                )
            )
        for _ in range(crossover_quota):
            parent_a = _tournament_select(population, tournament_size)
            parent_b = parent_a
            attempts = 0
            while parent_b is parent_a and len(population) > 1 and attempts < 5:
                parent_b = _tournament_select(population, tournament_size)
                attempts += 1
            children_samples.append(
                _spawn_crossover_child(
                    parent_a,
                    parent_b,
                    system_prompt=initial_system,
                    llm_client=llm_client,
                    model=model,
                    temperature=cfg.temperature,
                )
            )

        new_population: List[PopulationMember] = []
        for elite in elites:
            new_population.append(
                PopulationMember(
                    generation=gen,
                    index=len(new_population),
                    code_sample=elite.code_sample,
                    artifact=elite.artifact,
                    vlm_result=elite.vlm_result,
                )
            )

        if children_samples:
            evaluated_children = _evaluate_population(
                generation=gen,
                code_samples=children_samples,
                base_task_code=task_code_template,
                output_file=output_file,
                workspace_dir=workspace_dir,
                isaac_root_dir=str(ISAAC_ROOT_DIR),
                task=task,
                suffix=suffix,
                cfg=cfg,
                policy_feedback_template=policy_feedback,
                execution_error_feedback=execution_error_feedback,
                vlm_client=vlm_client,
            )
            new_population.extend(evaluated_children)

        for idx, member in enumerate(new_population[:population_size]):
            member.index = idx
            member.generation = gen
        population = new_population[:population_size]

        generation_best = max(population, key=lambda member: member.fitness, default=None)
        if generation_best:
            if best_member is None or generation_best.fitness > best_member.fitness:
                best_member = generation_best

    if best_member is None or best_member.artifact is None:
        logging.error("Evolution finished without a valid champion.")
        return

    champion_report = {
        "task": task,
        "generation": best_member.generation,
        "index": best_member.index,
        "fitness": best_member.fitness,
        "vlm_score": best_member.vlm_result.fitness_score if best_member.vlm_result else None,
        "vlm_feedback": best_member.vlm_result.qualitative_feedback if best_member.vlm_result else "",
        "stats_text": best_member.artifact.stats_text,
        "video_path": str(best_member.artifact.video_path) if best_member.artifact.video_path else None,
        "checkpoint": str(best_member.artifact.checkpoint_path) if best_member.artifact.checkpoint_path else None,
    }
    with open("champion.json", "w") as file:
        json.dump(champion_report, file, indent=2)
    logging.info("Champion summary: %s", champion_report)

if __name__ == "__main__":
    main()
