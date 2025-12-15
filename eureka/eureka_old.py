import hydra
import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
import re
import subprocess
from pathlib import Path
import shutil
import time
import ast
import sys
import sysconfig
from typing import List

from openai import OpenAI
import dashscope

# Prefer absolute imports to avoid ModuleNotFoundError when subprocesses do not
# inherit the expected PYTHONPATH.
try:
    from utils.misc import set_freest_gpu, block_until_training, filter_traceback
    from utils.file_utils import find_files_with_substring, load_tensorboard_logs
    from utils.create_task import create_task
    from utils.extract_task_code import file_to_string, get_function_signature
    from utils.video_utils_old import record_policy_rollout  # 使用与新版 Eureka 一致的录视频实现
except ImportError:
    from eureka.utils.misc import set_freest_gpu, block_until_training, filter_traceback
    from eureka.utils.file_utils import find_files_with_substring, load_tensorboard_logs
    from eureka.utils.create_task import create_task
    from eureka.utils.extract_task_code import file_to_string, get_function_signature
    from eureka.utils.video_utils_old import record_policy_rollout  # 使用与新版 Eureka 一致的录视频实现

EUREKA_ROOT_DIR = Path(__file__).resolve().parent  # /.../Eureka/eureka
# 确保本地 utils 可被导入（修复 ModuleNotFoundError: utils.create_task 等）
if str(EUREKA_ROOT_DIR) not in sys.path:
    sys.path.append(str(EUREKA_ROOT_DIR))
ISAAC_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"

def get_env_with_python_lib():
    """
    Constructs the subprocess environment, ensuring Python/LD_LIBRARY paths are correct.
    This is critical for subprocesses to find libpython3.8.so.1.0 and other shared libraries.
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
    # Critical: When rlgames_utils.py imports "from eureka.utils.file_utils", it triggers eureka.py import
    # eureka.py needs EUREKA_ROOT_DIR (eureka directory) in sys.path to import utils modules
    # Also need eureka_parent_dir so "from eureka.utils.file_utils" can find the eureka package
    python_lib = sysconfig.get_paths().get("purelib")
    site_packages = sysconfig.get_paths().get("platlib")
    eureka_parent_dir = str(EUREKA_ROOT_DIR.parent)  # /.../Eureka (contains eureka package)
    # Put eureka directory FIRST so eureka.py can find utils when imported
    _append_path("PYTHONPATH", [str(EUREKA_ROOT_DIR), eureka_parent_dir, python_lib, site_packages])
    
    # Conda / current interpreter lib paths (for libpython3.8.so.1.0)
    current_prefix = sys.prefix
    conda_lib_path = os.path.join(current_prefix, "lib")
    cuda_lib = "/usr/local/cuda/lib64"
    isaac_lib = str(Path(ISAAC_ROOT_DIR) / "bindings" / "python")
    _append_path("LD_LIBRARY_PATH", [conda_lib_path, cuda_lib, isaac_lib])
    
    return env

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    # 使用 DashScope 兼容 OpenAI SDK 的新写法（与新版 Eureka 一致）
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    client = OpenAI(
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
    env_parent = 'isaac' if f'{env_name}.py' in os.listdir(f'{EUREKA_ROOT_DIR}/envs/isaac') else 'dexterity'
    task_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}.py'
    task_obs_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}_obs.py'
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_code_string  = file_to_string(task_file)
    task_obs_code_string  = file_to_string(task_obs_file)
    output_file = f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"

    # Loading all text prompts (use prompts_old folder for eureka_old.py)
    prompt_dir = f'{EUREKA_ROOT_DIR}/utils/prompts_old'
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    task_code_string = task_code_string.replace(task, task+suffix)
    # Create Task YAML files
    create_task(ISAAC_ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None 
    
    # Eureka generation loop
    for iter in range(cfg.iteration):
        # Get Eureka response
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = cfg.sample if "gpt-3.5" in model else 4

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

        while True:
            if total_samples >= cfg.sample:
                break
            for attempt in range(1000):
                try:
                    # 使用新版 OpenAI 客户端的 chat.completions 接口
                    response_cur = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=cfg.temperature,
                        n=chunk_size,
                    )
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        print("Current Chunk Size", chunk_size)
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)
            if response_cur is None:
                logging.info("Code terminated due to too many failed attempts!")
                exit()

            # 新版 SDK 的字段访问方式
            responses.extend(response_cur.choices)
            prompt_tokens = response_cur.usage.prompt_tokens
            total_completion_token += response_cur.usage.completion_tokens
            total_token += response_cur.usage.total_tokens

        if cfg.sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0].message.content + "\n")

        # Logging Token Information
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        
        code_runs = [] 
        rl_runs = []
        for response_id in range(cfg.sample):
            # 新版 SDK 返回的是对象而不是 dict
            response_cur = responses[response_id].message.content
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string

            # Remove unnecessary imports
            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])
                    
            # Add the Eureka Reward Signature to the environment code
            try:
                gpt_reward_signature, input_lst = get_function_signature(code_string)
            except Exception as e:
                logging.info(f"Iteration {iter}: Code Run {response_id} cannot parse function signature!")
                continue

            # Validate that the function is actually a reward function
            # Check if function returns two values (rew_buf, rew_dict) or has "reward" in name
            try:
                module = ast.parse(code_string)
                function_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]
                if function_defs:
                    func = function_defs[0]
                    # Check return statement - should return two values
                    has_tuple_return = False
                    func_name_lower = func.name.lower()
                    is_reward_like = "reward" in func_name_lower or "compute" in func_name_lower
                    
                    for node in ast.walk(func):
                        if isinstance(node, ast.Return) and node.value:
                            if isinstance(node.value, ast.Tuple) and len(node.value.elts) >= 2:
                                has_tuple_return = True
                                break
                    
                    # If function doesn't return tuple and doesn't look like reward function, skip it
                    if not has_tuple_return and not is_reward_like:
                        logging.warning(
                            f"Iteration {iter}: Code Run {response_id} function '{func.name}' "
                            "does not return two values (rew_buf, rew_dict) and doesn't appear to be a reward function. "
                            "Skipping this code sample."
                        )
                        continue
            except Exception as e:
                logging.warning(f"Iteration {iter}: Code Run {response_id} failed to validate reward function: {e}")
                # Continue anyway - let it try to run and fail if it's wrong

            code_runs.append(code_string)
            
            # Ensure reward code has @torch.jit.script decorator
            reward_code = code_string
            if "@torch.jit.script" not in reward_code:
                reward_code = "@torch.jit.script\n" + reward_code
            
            # Build reward signature block (same as new eureka.py)
            reward_signature = [
                f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
                f"self.extras['gpt_reward'] = self.rew_buf.mean()",
                f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
            ]
            indent = " " * 8
            signature_block = "\n".join([indent + line for line in reward_signature])
            
            # Insert reward signature into compute_reward method (same as new eureka.py)
            task_code_iter = task_code_string
            for pattern in ["def compute_reward(self):", "def compute_reward(self, actions):"]:
                if pattern in task_code_iter:
                    task_code_iter = task_code_iter.replace(pattern, f"{pattern}\n{signature_block}")
                    break
            
            # Write environment file with correct order: task code, imports, reward function
            # This matches the structure used in new eureka.py (_write_candidate_files)
            with open(output_file, 'w') as file:
                file.write(task_code_iter + "\n")
                file.write("from typing import Tuple, Dict\n")
                file.write("import math\n")
                file.write("import torch\n")
                file.write("from torch import Tensor\n")
                file.write(reward_code)

            with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                file.write(reward_code)

            # Copy the generated environment code to hydra output directory for bookkeeping
            shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")

            # Find the freest GPU to run GPU-accelerated RL
            set_freest_gpu()
            
            # Get environment variables with proper library paths
            env_vars = get_env_with_python_lib()
            # Ensure CUDA_VISIBLE_DEVICES is set (from set_freest_gpu)
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                env_vars["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
            
            # Execute the python file with flags
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            with open(rl_filepath, 'w') as f:
                process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                            'hydra/output=subprocess',
                                            f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                                            f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                                            f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False',
                                            f'max_iterations={cfg.max_iterations}'],
                                            stdout=f, stderr=f, env=env_vars)
            block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
            rl_runs.append(process)
        
        # Gather RL training results and construct reward reflection
        code_feedbacks = []
        contents = []
        successes = []
        reward_correlations = []
        code_paths = []
        checkpoint_paths = []
        
        exec_success = False 
        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            rl_run.communicate()
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            code_paths.append(f"env_iter{iter}_response{response_id}.py")
            try:
                with open(rl_filepath, 'r') as f:
                    stdout_str = f.read() 
            except: 
                content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                content += code_output_tip
                contents.append(content) 
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                checkpoint_paths.append(None)
                continue

            content = ''
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == '':
                # If RL execution has no error, provide policy statistics feedback
                exec_success = True
                lines = stdout_str.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Tensorboard Directory:'):
                        break 
                tensorboard_logdir = line.split(':')[-1].strip() 
                tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)

                # Locate the latest policy checkpoint for this run
                ckpt_path = None
                try:
                    net_line = ''
                    for l in lines:
                        if l.startswith('Network Directory:'):
                            net_line = l
                            break
                    if net_line != '':
                        net_dir_str = net_line.split(':')[-1].strip()
                        net_dir = Path(net_dir_str)
                        if net_dir.exists():
                            ckpts = sorted(list(net_dir.glob("*.pth")), key=os.path.getmtime)
                            if len(ckpts) > 0:
                                ckpt_path = ckpts[-1]
                except Exception as e:
                    logging.info(f"Iteration {iter}: Code Run {response_id} failed to locate checkpoint with error: {e}")
                checkpoint_paths.append(ckpt_path)

                max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
                epoch_freq = max(int(max_iterations // 10), 1)
                
                content += policy_feedback.format(epoch_freq=epoch_freq)
                
                # Compute Correlation between Human-Engineered and GPT Rewards
                if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
                    gt_reward = np.array(tensorboard_logs["gt_reward"])
                    gpt_reward = np.array(tensorboard_logs["gpt_reward"])
                    reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                    reward_correlations.append(reward_correlation)

                # Add reward components log to the feedback
                for metric in tensorboard_logs:
                    if "/" not in metric:
                        metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                        metric_cur_max = max(tensorboard_logs[metric])
                        metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                        if "consecutive_successes" == metric:
                            successes.append(metric_cur_max)
                        metric_cur_min = min(tensorboard_logs[metric])
                        if metric != "gt_reward" and metric != "gpt_reward":
                            if metric != "consecutive_successes":
                                metric_name = metric 
                            else:
                                metric_name = "task_score"
                            content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                        else:
                            # Provide ground-truth score when success rate not applicable
                            if "consecutive_successes" not in tensorboard_logs:
                                content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                code_feedbacks.append(code_feedback)
                content += code_feedback  
            else:
                # Otherwise, provide execution traceback error feedback
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)
                checkpoint_paths.append(None)

            content += code_output_tip
            contents.append(content) 
        
        # Repeat the iteration if all code generation failed
        if not exec_success and cfg.sample != 1:
            execute_rates.append(0.)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
            continue

        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(np.array(successes))
        best_content = contents[best_sample_idx]
            
        max_success = successes[best_sample_idx]
        max_success_reward_correlation = reward_correlations[best_sample_idx]
        execute_rate = np.sum(np.array(successes) >= 0.) / cfg.sample

        # Update the best Eureka Output
        if max_success > max_success_overall:
            max_success_overall = max_success
            max_success_reward_correlation_overall = max_success_reward_correlation
            max_reward_code_path = code_paths[best_sample_idx]

        execute_rates.append(execute_rate)
        max_successes.append(max_success)
        max_successes_reward_correlation.append(max_success_reward_correlation)
        best_code_paths.append(code_paths[best_sample_idx])

        logging.info(f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation: {max_success_reward_correlation}")
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        # 使用新版 OpenAI Choice 对象的属性访问方式
        logging.info(f"Iteration {iter}: GPT Output Content:\n" +  responses[best_sample_idx].message.content + "\n")
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")

        # After each iteration, run the best policy once and record a rollout video
        try:
            best_checkpoint = None
            if len(checkpoint_paths) > 0:
                best_checkpoint = checkpoint_paths[best_sample_idx]
            if best_checkpoint is not None:
                logging.info(f"Iteration {iter}: Recording rollout video for best checkpoint: {best_checkpoint}")
                try:
                    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
                except Exception:
                    gpu_id = "0"
                _ = record_policy_rollout(
                    isaac_root_dir=Path(ISAAC_ROOT_DIR),
                    workspace_dir=workspace_dir,
                    task_name=task,
                    suffix=suffix,
                    checkpoint_path=best_checkpoint,
                    wandb_username=cfg.wandb_username,
                    wandb_project=cfg.wandb_project,
                    env=None,
                    rollout_steps=500,
                    headless=True,
                    force_render=False,
                    seed=iter,
                    gpu_id=gpu_id,
                )
            else:
                logging.info(f"Iteration {iter}: No valid checkpoint found for best sample, skip video recording.")
        except Exception as e:
            logging.info(f"Iteration {iter}: Failed to record rollout video for best policy with error: {e}")
            
        # Plot the success rate
        fig, axs = plt.subplots(2, figsize=(6, 6))
        fig.suptitle(f'{cfg.env.task}')

        x_axis = np.arange(len(max_successes))

        axs[0].plot(x_axis, np.array(max_successes))
        axs[0].set_title("Max Success")
        axs[0].set_xlabel("Iteration")

        axs[1].plot(x_axis, np.array(execute_rates))
        axs[1].set_title("Execute Rate")
        axs[1].set_xlabel("Iteration")

        fig.tight_layout(pad=3.0)
        plt.savefig('summary.png')
        np.savez('summary.npz', max_successes=max_successes, execute_rates=execute_rates, best_code_paths=best_code_paths, max_successes_reward_correlation=max_successes_reward_correlation)

        if len(messages) == 2:
            messages += [{"role": "assistant", "content": responses[best_sample_idx].message.content}]
            messages += [{"role": "user", "content": best_content}]
        else:
            assert len(messages) == 4
            messages[-2] = {"role": "assistant", "content": responses[best_sample_idx].message.content}
            messages[-1] = {"role": "user", "content": best_content}

        # Save dictionary as JSON file
        with open('messages.json', 'w') as file:
            json.dump(messages, file, indent=4)
    
    # Evaluate the best reward code many times
    if max_reward_code_path is None: 
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()
    logging.info(f"Task: {task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}")
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")
    shutil.copy(max_reward_code_path, output_file)
    
    eval_runs = []
    for i in range(cfg.num_eval):
        set_freest_gpu()
        
        # Get environment variables with proper library paths
        env_vars = get_env_with_python_lib()
        # Ensure CUDA_VISIBLE_DEVICES is set (from set_freest_gpu)
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_vars["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        
        # Execute the python file with flags
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                        'hydra/output=subprocess',
                                        f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                                        f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                                        f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False', f'seed={i}',
                                        ],
                                        stdout=f, stderr=f, env=env_vars)

        block_until_training(rl_filepath)
        eval_runs.append(process)

    reward_code_final_successes = []
    reward_code_correlations_final = []
    for i, rl_run in enumerate(eval_runs):
        rl_run.communicate()
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, 'r') as f:
            stdout_str = f.read() 
        lines = stdout_str.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('Tensorboard Directory:'):
                break 
        tensorboard_logdir = line.split(':')[-1].strip() 
        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        max_success = max(tensorboard_logs['consecutive_successes'])
        reward_code_final_successes.append(max_success)

        if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
            gt_reward = np.array(tensorboard_logs["gt_reward"])
            gpt_reward = np.array(tensorboard_logs["gpt_reward"])
            reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
            reward_code_correlations_final.append(reward_correlation)

    logging.info(f"Final Success Mean: {np.mean(reward_code_final_successes)}, Std: {np.std(reward_code_final_successes)}, Raw: {reward_code_final_successes}")
    logging.info(f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, Std: {np.std(reward_code_correlations_final)}, Raw: {reward_code_correlations_final}")
    np.savez('final_eval.npz', reward_code_final_successes=reward_code_final_successes, reward_code_correlations_final=reward_code_correlations_final)


if __name__ == "__main__":
    main()