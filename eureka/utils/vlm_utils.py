import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

def load_vlm_prompt_template(prompt_dir: Path) -> str:
    """Load VLM prompt template from prompts directory."""
    prompt_file = prompt_dir / "vlm_evaluation.txt"
    if not prompt_file.exists():
        logging.warning(f"VLM prompt file not found at {prompt_file}, using default template")
        return """You are an expert in Robotics and Reinforcement Learning.

Task Goal: {TASK_DESCRIPTION}

Your job is to watch the provided policy execution video and evaluate its "fitness" on a scale of 0 to 100 based on the task goal.

Please output your evaluation strictly in the following JSON format:
{{
    "fitness_score": <integer 0-100. 0 means complete failure (falling/static), 100 means perfect olympic-level performance>,
    "qualitative_feedback": "<A short, descriptive sentence summarizing the agent's core behavior or obvious flaws. E.g., 'The agent hops on one leg instead of running.'>",
    "analysis_notes": {{
        "what_it_did_well": "<relevant positives>",
        "what_it_did_wrong": "<core defects>"
    }}
}}"""
    return prompt_file.read_text()


@dataclass
class VLMResult:
    fitness_score: int = 0
    qualitative_feedback: str = ""
    analysis_notes: Dict[str, Any] = field(default_factory=lambda: {"what_it_did_well": "", "what_it_did_wrong": ""})
    raw_response: Dict[str, Any] = field(default_factory=dict)
    video_path: Optional[str] = None


class VLMClient:
    """Utility wrapper for VLM-based fitness evaluation."""

    def __init__(
        self,
        model_name: str,
        task_description: str,
        prompt_template: Optional[str] = None,
        prompt_dir: Optional[Path] = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_client: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name
        self.task_description = task_description
        # Load prompt template from file if not provided
        if prompt_template is None:
            if prompt_dir is None:
                # Try to find prompts directory relative to this file
                _vlm_utils_dir = Path(__file__).parent
                prompt_dir = _vlm_utils_dir / "prompts"
            self.prompt_template = load_vlm_prompt_template(prompt_dir)
        else:
            self.prompt_template = prompt_template
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.client = openai_client

    def evaluate(
        self,
        video_path: str,
        extra_prompt: Optional[str] = None,
        rubric_json: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> "VLMResult":
        """Evaluate a rollout video using a VLM or mock scorer.
        
        Args:
            video_path: Path to the video file
            extra_prompt: Additional context for evaluation
            rubric_json: Optional visual rubric JSON to inject into prompt
            max_retries: Maximum number of retry attempts
        """
        formatted_prompt = self.prompt_template.format(TASK_DESCRIPTION=self.task_description)
        
        # Inject rubric if provided
        if rubric_json:
            rubric_text = json.dumps(rubric_json, indent=2)
            formatted_prompt += f"\n\nVisual Evaluation Rubric:\n{rubric_text}\n"
            formatted_prompt += "\nUse the above rubric criteria to guide your evaluation. Consider each criterion's weight when scoring.\n"
        
        if extra_prompt:
            formatted_prompt += f"\n\nAdditional Context:\n{extra_prompt}\n"

        if self.model_name.lower() == "mock":
            return self._mock_response(video_path)

        return self._call_vlm_api(video_path, formatted_prompt, max_retries)

    def _mock_response(self, video_path: str) -> "VLMResult":
        """Return a mocked VLM response for debugging without API keys."""
        score = random.randint(15, 95)
        positives = random.choice(
            [
                "Maintains balance for most of the rollout.",
                "Shows progress toward the goal pose.",
                "Demonstrates stable locomotion.",
            ]
        )
        negatives = random.choice(
            [
                "Fails to reach the target object.",
                "Falls after initial acceleration.",
                "Remains mostly static without purposeful motion.",
            ]
        )
        response = {
            "fitness_score": score,
            "qualitative_feedback": f"Mock evaluation: score {score}. {positives}",
            "analysis_notes": {
                "what_it_did_well": positives,
                "what_it_did_wrong": negatives,
            },
        }
        return VLMResult(
            fitness_score=score,
            qualitative_feedback=response["qualitative_feedback"],
            analysis_notes=response["analysis_notes"],
            raw_response=response,
            video_path=video_path,
        )

    def _call_vlm_api(self, video_path: str, prompt: str, max_retries: int) -> "VLMResult":
        """Call VLM API to evaluate video using OpenAI-compatible interface."""
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            logging.warning(
                "No API key found in %s. Falling back to zero score for model %s.",
                self.api_key_env,
                self.model_name,
            )
            return VLMResult(
                fitness_score=0,
                qualitative_feedback="VLM evaluation skipped due to missing API key.",
                analysis_notes={"what_it_did_well": "", "what_it_did_wrong": "API key missing"},
                raw_response={"error": "missing_api_key"},
                video_path=video_path,
            )

        if self.client is None:
            logging.warning("No OpenAI-compatible client provided; returning default score 0.")
            return VLMResult(
                fitness_score=0,
                qualitative_feedback="VLM client not configured.",
                analysis_notes={"what_it_did_well": "", "what_it_did_wrong": "VLM client missing"},
                raw_response={"error": "missing_client"},
                video_path=video_path,
            )

        # Convert video path to absolute path and check if it exists
        abs_video_path = Path(video_path).resolve()
        if not abs_video_path.exists():
            logging.warning(f"Video file not found: {video_path}")
            return VLMResult(
                fitness_score=0,
                qualitative_feedback="Video file not found.",
                analysis_notes={"what_it_did_well": "", "what_it_did_wrong": "Video file missing"},
                raw_response={"error": "video_not_found"},
                video_path=video_path,
            )

        # Prepare video URI (file:// for local files)
        video_uri = abs_video_path.as_uri()

        last_error: Optional[str] = None
        for attempt in range(max_retries):
            try:
                # Use OpenAI-compatible API for DashScope qwen-vl models
                # DashScope supports video input via content array with video type
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": video_uri,
                                },
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                )
                
                response_text = response.choices[0].message.content
                if not response_text:
                    raise ValueError("Empty response from VLM API")
                
                # Parse JSON from response
                parsed_json = parse_vlm_json(response_text)
                if parsed_json:
                    return VLMResult(
                        fitness_score=parsed_json.get("fitness_score", 0),
                        qualitative_feedback=parsed_json.get("qualitative_feedback", ""),
                        analysis_notes=parsed_json.get("analysis_notes", {"what_it_did_well": "", "what_it_did_wrong": ""}),
                        raw_response=parsed_json,
                        video_path=video_path,
                    )
                else:
                    # 记录原始响应以便调试
                    logging.warning("Failed to parse JSON from VLM response. Raw response (first 500 chars):\n%s", response_text[:500])
                    raise ValueError(f"Failed to parse JSON from VLM response. Response preview: {response_text[:200]}...")
                    
            except Exception as exc:
                last_error = str(exc)
                # 记录更详细的错误信息
                error_msg = str(exc)
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                logging.warning("VLM call attempt %d failed: %s", attempt + 1, error_msg)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        logging.error("VLM evaluation failed after %d attempts. Error: %s", max_retries, last_error)
        return VLMResult(
            fitness_score=0,
            qualitative_feedback="VLM evaluation failed.",
            analysis_notes={"what_it_did_well": "", "what_it_did_wrong": last_error or "unknown error"},
            raw_response={"error": last_error or "unknown"},
            video_path=video_path,
        )


def parse_vlm_json(response_text: str) -> Optional[Dict[str, Any]]:
    """Try to parse VLM output into JSON, handling typical formatting noise."""
    if not response_text:
        return None
    
    response_text = response_text.strip()
    
    # 尝试 1: 直接解析
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # 尝试 2: 移除 markdown 代码块标记 (```json ... ```)
    # 移除 ```json 和 ``` 标记
    cleaned = re.sub(r'```json\s*', '', response_text)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    
    if cleaned != response_text:
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    
    # 尝试 3: 提取第一个完整的 JSON 对象
    try:
        start_idx = response_text.index("{")
        end_idx = response_text.rindex("}") + 1
        json_str = response_text[start_idx:end_idx]
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        pass
    
    # 尝试 4: 查找所有可能的 JSON 对象并尝试解析
    try:
        # 使用正则表达式找到所有可能的 JSON 对象
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    
    # 所有尝试都失败了
    logging.warning("Failed to parse JSON from VLM response. Response text (first 500 chars):\n%s", response_text[:500])
    return None

