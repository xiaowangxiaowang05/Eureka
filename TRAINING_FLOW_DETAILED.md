# Eureka.py 完整训练流程详细说明

## 代码检查结果

### ✅ 代码状态
- **语法检查**: 通过
- **Linter警告**: 1个（openai导入警告，这是IDE环境问题，不影响运行）
- **代码结构**: 完整，无语法错误

### ⚠️ 运行时依赖
- 需要安装: `hydra-core`, `openai`, `numpy` 等依赖包
- 导入测试失败是因为缺少运行时依赖，不是代码问题

---

## 整体训练流程架构

Eureka采用**4阶段进化算法（EUREKA-V2）**，包含：
- **Phase 0**: 语义系统设置（Semantic System Setup）
- **Phase 1**: 初始种群生成与评估（Generation 0）
- **Phase 2**: 自适应过滤（Adaptive Filtering）
- **Phase 3**: 选择性VLM评估（Selective VLM Evaluation）
- **Phase 4**: Pareto漏斗选择（Pareto Funnel Selection）
- **进化循环**: Generation 1 到 N

---

## 详细流程步骤

### 📋 阶段 0: 初始化和语义系统设置

#### 步骤 0.1: 初始化配置和客户端
**位置**: `main()` 函数开始（Line 978-1000）

1. **加载配置**
   - 使用 Hydra 加载配置文件 (`cfg/config.yaml`)
   - 获取工作空间目录
   - 读取任务名称 (`task`) 和任务描述 (`task_description`)

2. **初始化API客户端**
   ```python
   llm_client = OpenAI(...)  # 用于LLM调用（生成奖励函数）
   vlm_api_client = OpenAI(...)  # 用于VLM调用（视频评估）
   ```

3. **加载任务文件**
   - 定位环境文件: `eureka/envs/{env_parent}/{env_name}.py`
   - 定位观察文件: `eureka/envs/{env_parent}/{env_name}_obs.py`
   - 读取任务代码字符串 (`task_code_string`)
   - 复制观察文件到工作空间 (`env_init_obs.py`)

4. **加载提示词模板**
   - 从 `eureka/utils/prompts/` 目录加载所有提示词：
     - `initial_system.txt` - 初始系统提示
     - `initial_user.txt` - 初始用户提示
     - `reward_signature.txt` - 奖励函数签名
     - `policy_feedback.txt` - 策略反馈模板
     - `execution_error_feedback.txt` - 执行错误反馈
     - `mutation.txt` - 变异提示词
     - `crossover.txt` - 交叉提示词
     - `visual_rubric.txt` - 视觉量表提示词
     - `sanity_check.txt` - 健康检查提示词

5. **创建Isaac Gym任务**
   - 调用 `create_task()` 创建任务文件
   - 输出文件: `isaacgymenvs/isaacgymenvs/tasks/{env_name}{suffix}.py`

6. **配置进化参数**
   - `generations`: 进化代数
   - `population_size` (K): 每代种群大小
   - `tournament_size`: 锦标赛选择大小
   - `elite_fraction`: 精英比例
   - `elite_count`: 精英数量 = `population_size * elite_fraction`

#### 步骤 0.2: 生成视觉量表（Visual Rubric）
**位置**: `generate_visual_rubric()` (Line 788-835) 和 `main()` (Line 1061-1072)

1. **调用LLM生成视觉评估标准**
   - 使用 `visual_rubric_prompt_template` 提示词
   - 输入: 任务描述 (`task_description`)
   - 输出: JSON格式的视觉评估标准，包含：
     - `criteria`: 评估标准列表（每个标准有名称、描述、权重）
     - `notes`: 备注信息

2. **解析和验证**
   - 从LLM响应中提取JSON
   - 如果解析失败，使用默认量表

3. **日志输出**
   ```
   ================================================================================
   LLM Generated Visual Rubric:
   {
     "criteria": [...],
     "notes": "..."
   }
   ================================================================================
   ```

#### 步骤 0.3: 生成健康检查逻辑（Sanity Check Logic）
**位置**: `generate_sanity_check_logic()` (Line 837-881) 和 `main()` (Line 1074-1088)

1. **调用LLM生成健康检查表达式**
   - 使用 `sanity_check_prompt_template` 提示词
   - 输入: 环境代码 (`task_code_string`)
   - 输出: Python布尔表达式字符串（用于检测失败情况）

2. **验证表达式**
   - 尝试编译表达式
   - 如果无效，返回 `None`

3. **日志输出**
   ```
   ================================================================================
   LLM Generated Sanity Check Logic:
   Sanity Check Expression: {表达式或None}
   ================================================================================
   ```

---

### 🎯 阶段 1: 初始种群生成（Generation 0）

#### 步骤 1.1: 生成初始奖励函数代码
**位置**: `_generate_llm_samples()` (Line 945-976) 和 `main()` (Line 1090-1097)

1. **调用LLM生成K个初始奖励函数**
   - 使用 `initial_system` 和 `initial_user` 提示词
   - 输入: 任务观察代码、任务描述
   - 输出: K个奖励函数代码样本 (`CodeSample` 列表)

2. **提取奖励代码**
   - 从LLM响应中提取Python函数代码
   - 每个样本包含: `code`, `raw_response`, `metadata`

#### 步骤 1.2: 评估初始种群
**位置**: `_evaluate_population()` (Line 420-691) 和 `main()` (Line 1098-1113)

**详细子流程见下方"种群评估详细流程"**

#### 步骤 1.3: 选择初始最佳个体
**位置**: `main()` (Line 1115-1129)

1. **计算最佳个体**
   ```python
   best_member = max(population, key=lambda member: member.fitness)
   ```

2. **计算统计信息**
   - `max_success`: 最大成功指标
   - `execute_rate`: 执行成功率
   - `max_success_reward_correlation`: 最大成功奖励相关性

3. **日志输出**
   ```
   Iteration 0: Max Success: {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation: {correlation}
   Iteration 0: Best Generation ID: {best_member.index}
   ```

---

### 🔄 阶段 2-N: 进化循环（Generation 1 到 N）

#### 步骤 2.1: 生成子代（K个新个体）
**位置**: `main()` (Line 1131-1170)

1. **计算变异和交叉配额**
   ```python
   mutation_quota = int(round(population_size * evo_cfg.mutation_ratio))
   crossover_quota = population_size - mutation_quota
   ```
   - 确保 `mutation_quota + crossover_quota = population_size` (K)

2. **生成变异子代** (`mutation_quota` 个)
   - 对每个变异配额：
     - 使用锦标赛选择选择父代 (`_tournament_select()`)
     - 调用 `_spawn_mutation_child()` 生成变异子代
     - 使用 `mutation_prompt_template` 提示词
     - 输入: 父代代码、适应度分数、VLM反馈、组件统计

3. **生成交叉子代** (`crossover_quota` 个)
   - 对每个交叉配额：
     - 使用锦标赛选择选择两个不同的父代
     - 调用 `_spawn_crossover_child()` 生成交叉子代
     - 使用 `crossover_prompt_template` 提示词
     - 输入: 两个父代的代码、适应度分数、VLM反馈

#### 步骤 2.2: 评估子代
**位置**: `main()` (Line 1172-1189)

- 调用 `_evaluate_population()` 评估所有K个子代
- **详细流程见下方"种群评估详细流程"**

#### 步骤 2.3: 精英选择和种群更新
**位置**: `main()` (Line 1191-1257)

1. **合并种群池**
   ```python
   combined_pool = population + evaluated_children  # 2K个个体
   ```

2. **选择精英** (`_select_elites()`)
   - 使用Pareto漏斗选择：
     - 步骤1: 按物理指标排序，保留前 `elite_count * 2` 个
     - 步骤2: 在功能候选者中按视觉分数排序
     - 步骤3: 返回前 `elite_count` 个精英

3. **填充剩余位置**
   - 从子代中选择适应度最高的个体
   - 如果还不够，从合并池中选择

4. **更新种群**
   - 更新每个成员的 `generation` 和 `index`
   - 限制种群大小为 `population_size` (K)

#### 步骤 2.4: 记录最佳个体
**位置**: `main()` (Line 1259-1276)

1. **计算当前代最佳个体**
   ```python
   generation_best = max(population, key=lambda member: member.fitness)
   ```

2. **更新全局最佳**
   ```python
   if best_member is None or generation_best.fitness > best_member.fitness:
       best_member = generation_best
   ```

3. **日志输出**
   ```
   Iteration {gen}: Max Success: {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation: {correlation}
   Iteration {gen}: Best Generation ID: {generation_best.index}
   ```

---

### 📊 种群评估详细流程 (`_evaluate_population()`)

这是整个系统的核心评估流程，包含Phase 1-3的所有步骤。

#### Phase 1: 并行训练（Parallel Training）

**位置**: `_evaluate_population()` (Line 438-478)

##### 步骤 1.1: 启动训练任务
**位置**: `_launch_candidate_training()` (Line 285-370)

对每个候选（K个）:

1. **解析奖励函数签名**
   - 从代码中提取函数定义
   - 验证函数签名有效性

2. **写入候选文件**
   - 创建环境文件: `env_gen{generation}_cand{candidate_idx}.py`
   - 创建奖励文件: `reward_gen{generation}_cand{candidate_idx}.py`
   - 将奖励代码嵌入到任务代码中

3. **GPU分配**
   - 解析 `cfg.gpu_id` (例如 "1,2,3")
   - 使用轮询分配: `assigned_gpu_idx = candidate_idx % num_gpus`
   - 设置 `CUDA_VISIBLE_DEVICES=assigned_gpu_id`
   - 分配唯一端口: `master_port = 29500 + candidate_idx`

4. **启动训练进程**
   ```python
   cmd = ["python", "-u", "train.py", ...]
   process = subprocess.Popen(cmd, stdout=f, stderr=f, env=env_vars)
   ```
   - 训练参数:
     - `task={task}{suffix}`
     - `max_iterations={cfg.max_iterations}`
     - `headless=True`
     - `capture_video=False` (训练时不录制视频)
     - `seed={candidate_idx}`

5. **等待训练完成**
   - 调用 `block_until_training()` 阻塞直到训练完成

##### 步骤 1.2: 收集训练结果
**位置**: `_harvest_training_artifact()` (Line 372-418)

对每个训练任务:

1. **等待进程完成**
   ```python
   job.process.wait()
   ```

2. **解析训练日志**
   - 读取日志文件内容
   - 提取 Tensorboard 目录路径
   - 提取 Network 目录路径
   - 检测执行错误（traceback）

3. **加载Tensorboard日志**
   - 如果存在Tensorboard目录:
     - 调用 `load_tensorboard_logs()` 加载所有指标
     - 调用 `_summarize_tensorboard_logs()` 生成统计摘要
     - 提取关键指标:
       - `success_metric`: 成功指标（通常是 `consecutive_successes` 的最大值）
       - `reward_correlation`: 奖励相关性
       - `stats_text`: 格式化的统计文本

4. **定位检查点**
   - 在Network目录中查找最新的 `.pth` 文件
   - 用于后续视频录制

5. **创建训练工件**
   ```python
   TrainingArtifact(
       env_file, reward_only_file, log_file,
       tensorboard_dir, network_dir, checkpoint_path,
       metrics_summary, stats_text,
       success_metric, reward_correlation
   )
   ```

**日志输出**:
```
Starting parallel training for {K} candidates...
Waiting for all training jobs to finish...
Training phase completed. Starting Phase 2: Adaptive Filtering...
```

---

#### Phase 2: 自适应过滤（Adaptive Filtering）

**位置**: `_evaluate_population()` (Line 479-525)

##### 步骤 2.1: 健康检查过滤（Sanity Check Filter）

1. **对每个候选应用健康检查**
   - 如果 `sanity_check_fn` 存在:
     - 调用 `_evaluate_sanity_check()` 评估表达式
     - 表达式格式: `artifact.metrics_summary[...]` 或 `artifact.success_metric ...`
     - 如果评估为 `True`（表示失败），标记为 `skip_vlm=True`

2. **统计过滤结果**
   - 记录被健康检查过滤的候选数量

##### 步骤 2.2: 排名截断过滤（Rank-Based Truncation）

1. **筛选有效候选**
   - 排除 `None` 工件
   - 排除已被健康检查过滤的候选

2. **按物理指标排序**
   ```python
   valid_candidates.sort(key=lambda x: x[1].success_metric, reverse=True)
   ```

3. **截断底部50%**
   - 保留前 `len(valid_candidates) // 2` 个
   - 标记剩余候选为 `skip_vlm=True`

**日志输出**:
```
Phase 2 Complete: {sanity_check_pruned} pruned by sanity check, {rank_pruned} pruned by ranking. {remaining} candidates proceed to VLM.
```

---

#### 步骤 2.3: 视频录制（Video Recording）

**位置**: `_evaluate_population()` (Line 527-558)

对每个候选:

1. **检查条件**
   - 工件存在
   - 检查点路径存在
   - `cfg.capture_video=True`

2. **录制策略视频**
   - 调用 `record_policy_rollout()`:
     - 使用训练好的检查点
     - 录制 `cfg.video.rollout_len` 步
     - 使用 `headless=cfg.video.headless`
     - 使用默认GPU 0进行渲染
     - 保存视频到工作空间

3. **保存视频路径**
   - 如果录制成功，将视频路径保存到 `artifact.video_path`

**日志输出**:
```
Recording videos for all candidates...
```

---

#### Phase 3: 选择性VLM评估（Selective VLM Evaluation）

**位置**: `_evaluate_population()` (Line 560-632)

##### 步骤 3.1: 准备VLM评估

1. **创建线程池**
   ```python
   executor = ThreadPoolExecutor(max_workers=min(16, len(code_samples)))
   ```

2. **遍历所有候选**

##### 步骤 3.2: 跳过过滤的候选

对每个被标记为 `skip_vlm=True` 的候选:
- 创建零分VLM结果
- 增加 `skipped_count`

##### 步骤 3.3: 提交VLM评估任务

对每个未被过滤的候选:

1. **检查视频文件**
   - 如果视频路径不存在，标记为缺失，增加 `missing_video_count`

2. **提交异步任务**
   ```python
   future = executor.submit(
       vlm_client.evaluate,
       video_path,
       extra_prompt=artifact.stats_text,  # 奖励组件统计
       rubric_json=global_rubric_json,     # 视觉量表
       max_retries=cfg.vlm.max_retries
   )
   ```

##### 步骤 3.4: 收集VLM结果

1. **等待所有任务完成**
   ```python
   for future in concurrent.futures.as_completed(future_to_idx):
       result = future.result()
       vlm_results_map[idx] = result
   ```

2. **处理错误**
   - 如果VLM评估失败，创建零分结果并记录错误

3. **汇总统计**
   ```
   Skipped VLM evaluation for {skipped_count} candidate(s) (filtered or no video)
   Video file not found for {missing_video_count} candidate(s)
   ```

**VLM评估内容**:
- 输入: 策略视频、奖励组件统计、视觉量表JSON
- 输出: `VLMResult` 包含:
  - `fitness_score`: 0-100的适应度分数
  - `qualitative_feedback`: 定性反馈文本
  - `analysis_notes`: 包含 `what_it_did_well` 和 `what_it_did_wrong`

---

#### 步骤 3.5: 构建种群成员

**位置**: `_evaluate_population()` (Line 634-650)

对每个候选:

1. **创建种群成员**
   ```python
   PopulationMember(
       generation=generation,
       index=idx,
       code_sample=sample,
       artifact=artifact,
       vlm_result=vlm_result,
       skip_vlm=skip_vlm,
       visual_score=visual_score  # VLM分数
   )
   ```

2. **计算适应度**
   - `physical_metric`: `artifact.success_metric` (物理成功指标)
   - `visual_score`: `vlm_result.fitness_score` (VLM视觉分数)
   - `fitness`: 综合适应度（当前实现中等于 `visual_score`）

---

#### 步骤 3.6: 输出评估结果

**位置**: `_evaluate_population()` (Line 652-689)

##### 输出 1: 训练结果排名

```
================================================================================
Generation {generation}: Training Results Ranking (K={K} candidates)
--------------------------------------------------------------------------------
Rank  1: Candidate  0 | Physical Metric:  5.4533 | Visual Score:  85 | Fitness:  85.00
Rank  2: Candidate  1 | Physical Metric:  4.1234 | Visual Score:  78 | Fitness:  78.00
...
================================================================================
```

- 按物理指标排序
- 显示每个候选的物理指标、视觉分数、适应度

##### 输出 2: VLM反馈详情

```
================================================================================
Generation {generation}: VLM Feedback for All Candidates
--------------------------------------------------------------------------------
Candidate  0:
  VLM Score: 85
  Qualitative Feedback: The agent demonstrates stable locomotion...
  What it did well: Maintains balance throughout the rollout
  What it did wrong: Slight deviation from optimal path
--------------------------------------------------------------------------------
Candidate  1:
  ...
================================================================================
```

- 按候选ID排序
- 显示每个候选的完整VLM反馈

---

### 🏆 阶段 3: 最终输出（Termination）

**位置**: `main()` (Line 1278-1295)

#### 步骤 3.1: 验证冠军

1. **检查有效性**
   ```python
   if best_member is None or best_member.artifact is None:
       logging.error("Evolution finished without a valid champion.")
       return
   ```

#### 步骤 3.2: 生成冠军报告

1. **创建报告字典**
   ```python
   champion_report = {
       "task": task,
       "generation": best_member.generation,
       "index": best_member.index,
       "fitness": best_member.fitness,
       "vlm_score": best_member.vlm_result.fitness_score,
       "vlm_feedback": best_member.vlm_result.qualitative_feedback,
       "stats_text": best_member.artifact.stats_text,
       "video_path": str(best_member.artifact.video_path),
       "checkpoint": str(best_member.artifact.checkpoint_path),
   }
   ```

2. **保存到文件**
   - 写入 `champion.json`

3. **日志输出**
   ```
   Champion summary: {champion_report}
   ```

---

## 关键函数说明

### `_tournament_select()`
- **功能**: 锦标赛选择父代
- **方法**: 随机选择 `tournament_size` 个候选，返回适应度最高的
- **适应度计算**: `(physical_metric, visual_score)` 元组

### `_spawn_mutation_child()`
- **功能**: 生成变异子代
- **输入**: 父代成员、系统提示、变异提示模板
- **输出**: 新的 `CodeSample`

### `_spawn_crossover_child()`
- **功能**: 生成交叉子代
- **输入**: 两个父代成员、系统提示、交叉提示模板
- **输出**: 新的 `CodeSample`

### `_select_elites()`
- **功能**: Pareto漏斗选择精英
- **步骤**:
  1. 按物理指标排序，保留前 `elite_count * 2` 个
  2. 按视觉分数排序
  3. 返回前 `elite_count` 个

### `_evaluate_sanity_check()`
- **功能**: 评估健康检查表达式
- **输入**: 表达式字符串、训练工件
- **输出**: 布尔值（True表示失败）

---

## 数据流图

```
Phase 0: 语义系统设置
  ├─> 生成视觉量表 (JSON)
  └─> 生成健康检查逻辑 (Python表达式)

Generation 0: 初始种群
  ├─> LLM生成K个奖励函数
  └─> 评估种群
      ├─> Phase 1: 并行训练 (K个训练任务)
      ├─> Phase 2: 自适应过滤
      │   ├─> 健康检查过滤
      │   └─> 排名截断过滤
      ├─> 视频录制 (所有候选)
      └─> Phase 3: 选择性VLM评估
          └─> 构建种群成员

Generation 1-N: 进化循环
  ├─> 生成K个子代
  │   ├─> 变异子代 (mutation_quota个)
  │   └─> 交叉子代 (crossover_quota个)
  ├─> 评估子代 (同Generation 0流程)
  ├─> Phase 4: Pareto漏斗选择
  │   ├─> 合并种群池 (2K个)
  │   ├─> 选择精英 (elite_count个)
  │   └─> 填充剩余位置
  └─> 更新种群和最佳个体

最终输出
  └─> 生成冠军报告 (champion.json)
```

---

## 关键配置参数

### 进化参数 (`cfg.evolution`)
- `generations`: 进化代数
- `population_size` (K): 每代种群大小
- `tournament_size`: 锦标赛选择大小
- `elite_fraction`: 精英比例
- `mutation_ratio`: 变异比例

### 训练参数 (`cfg`)
- `max_iterations`: 最大训练迭代次数
- `gpu_id`: GPU ID列表 (例如 "1,2,3")
- `use_wandb`: 是否使用WandB
- `capture_video`: 是否录制视频

### VLM参数 (`cfg.vlm`)
- `max_retries`: VLM评估最大重试次数

### 视频参数 (`cfg.video`)
- `rollout_len`: 视频录制步数
- `headless`: 是否无头模式
- `force_render`: 是否强制渲染

---

## 日志输出总结

### Phase 0 输出
1. ✅ LLM生成的视觉量表（完整JSON）
2. ✅ LLM生成的健康检查逻辑（表达式）

### 每代评估输出
3. ✅ 训练结果排名（K个候选的物理指标、视觉分数、适应度）
4. ✅ VLM反馈详情（每个候选的完整反馈）

### 每代总结输出
5. ✅ 最佳个体统计（Max Success, Execute Rate, Correlation）
6. ✅ 最佳个体ID

### 最终输出
7. ✅ 冠军总结（包含所有关键信息）

---

## 潜在问题和注意事项

### 1. GPU分配
- 每个候选使用单个GPU（轮询分配）
- 视频录制使用默认GPU 0
- 确保GPU数量足够支持并行训练

### 2. 端口冲突
- 每个候选使用唯一端口: `29500 + candidate_idx`
- 如果候选数 > 100，可能需要调整端口范围

### 3. 内存管理
- 并行训练K个任务可能消耗大量GPU内存
- 建议根据GPU内存调整 `population_size`

### 4. VLM评估成本
- Phase 2过滤可以减少VLM调用次数
- 但所有候选的视频都会被录制（即使被过滤）

### 5. 文件管理
- 每代生成大量文件（训练日志、检查点、视频）
- 确保工作空间有足够磁盘空间

---

## 总结

Eureka-V2实现了完整的4阶段进化算法流程：

1. **Phase 0**: 语义系统设置，生成评估标准和健康检查
2. **Phase 1**: 并行训练所有候选
3. **Phase 2**: 自适应过滤，减少VLM评估成本
4. **Phase 3**: 选择性VLM评估，获取视觉反馈
5. **Phase 4**: Pareto漏斗选择，平衡物理和视觉指标
6. **进化循环**: 每代生成K个子代，评估并选择精英

整个流程确保了：
- ✅ 每代生成K个子代（`mutation_quota + crossover_quota = K`）
- ✅ 成本高效的VLM评估（通过过滤）
- ✅ 平衡的适应度评估（物理+视觉）
- ✅ 完整的日志输出（用户要求的所有信息）

