import google.generativeai as genai
from google.genai import types
from google.genai import Client as google_client
import re
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Tuple
from dotenv import load_dotenv
import logging
import datetime
import time
import traceback
from dp_for_tsp import DP4TSP
import openai
from openai import OpenAI
from config import framework_config

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft JhengHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 這是 STEP 1 的 Prompt，用於問題分類與初步設計。第一個prompt就是這樣
STEP_1_PROMPT_TEMPLATE = """
Task Description: {task_description}

Analyze the computational complexity of the described task. Based on your analysis, classify the problem into one of two categories:
1.  **Definite Algorithm Feasible (DP-like)**: A category for problems where an exact, optimal solution can be found in a reasonable, predictable amount of time (e.g., polynomial time).
2.  **Indefinite Algorithm Required (Heuristic/NP-Hard)**: A category for problems where finding a guaranteed optimal solution is computationally infeasible, thus requiring heuristic or approximation strategies.

Start your response with the classification in the format: "Classification: [Your Classification]".

Then, provide your reasoning. If you classify it as 'Definite Algorithm Feasible', describe a suitable exact algorithm. If you classify it as 'Indefinite Algorithm Required', propose an effective heuristic algorithm to start with.
"""

# 這是 STEP 2 的 Prompt，用於首次實作
STEP_2_PROMPT_TEMPLATE = """
Based on your previous analysis where you classified the problem as **{classification}**:

-   **If 'Definite Algorithm Feasible'**: Write the Python code for the exact algorithm you described. Execute it to find the optimal solution for the given data.
-   **If 'Indefinite Algorithm Required'**: Write the Python code for a first version of the heuristic algorithm you proposed. Execute the code on the given data and provide the resulting score.

After providing the code and its output, outline a brief, actionable plan for how this initial attempt could be improved in future iterations.
Please output the scores or inference results in the last line without any other values.
"""

# 這是 STEP 3A 的 Prompt (DP/Exact 路線)，用於精煉與驗證
STEP_3A_PROMPT_TEMPLATE = """
You are in the finalization phase for a problem solved with a definite algorithm.

**Previous Work:**
-   Algorithm: {algorithm_description}
-   Code:
```python
{code}
```
-   Result: {result}

**Your Task:**
1.  **Code Review and Refinement**: Review the code. Refactor it for clarity, efficiency, and best practices without changing the core logic. Add comments to explain key parts.
2.  **Verification Run**: Execute the refined code again to ensure the output is identical and correct.
3.  **Final Output**: Present the final, verified optimal solution and its score.

If you are confident that the solution is optimal and the code is finalized, conclude your entire response with the single word "FINISHED".
Please output the scores or inference results in the last line without any other values.
"""

# 這是 STEP 3B 的 Prompt (Heuristic 路線)，用於迭代優化
STEP_3B_PROMPT_TEMPLATE = """
You are in an iterative optimization cycle for a heuristic-based problem.

**Best Solution Found So Far**: {best_score}

**History of Previous Attempts**:
{history_log}

**Your Task for This New Iteration**:
1.  **Analyze and Strategize**: Review your previous attempt(s). Propose a concrete modification to improve the result. This could be adjusting parameters, modifying the algorithm (e.g., 2-Opt to 3-Opt), or even trying a new heuristic. Justify your new strategy.
2.  **Implement and Execute**: Write and run the modified Python code reflecting your new strategy.
3.  **Report and Compare**: State the new result. Compare it to the best result from all previous attempts.

If you believe further significant improvement is unlikely, you can conclude your entire response with the single word "FINISHED".
Please output the scores or inference results in the last line without any other values.
"""

# 這是 LLM-as-a-Judge 的評分 Prompt
EVALUATION_PROMPT_TEMPLATE = """
You are an expert evaluator assessing the reasoning quality of an AI model's response to a complex problem-solving task.
Please evaluate the following response based on five criteria. For each criterion, provide a score from 0 to 20 and a brief justification.

**The AI's Response to Evaluate:**
---
{reasoning_text}
---

**Evaluation Criteria:**

1.  **Problem Understanding & Analysis (0-20)**: How well did the model comprehend the problem's constraints, goals, and underlying computational complexity?
2.  **Strategic Planning (0-20)**: Was the proposed algorithm or plan of action logical, well-justified, and appropriate for the problem? For iterative steps, does the plan build intelligently on previous results?
3.  **Implementation Quality (0-20)**: If code was generated, assess its correctness, efficiency, and clarity. Does it accurately reflect the proposed strategy?
4.  **Self-Correction & Iteration (0-20)**: How effectively did the model analyze its previous results to propose specific, actionable improvements? (This is crucial for heuristic optimization). If this is the first step, score based on the quality of the proposed future directions.
5.  **Clarity and Communication (0-20)**: Was the explanation clear, concise, and easy for a human to understand?

Please provide your evaluation in a JSON format like this:
{{
    "scores": {{
        "problem_understanding": 20,
        "strategic_planning": 20,
        "implementation_quality": 20,
        "self_correction": 20,
        "clarity": 20
    }},
    "justifications": {{
        "problem_understanding": "...",
        "strategic_planning": "...",
        "implementation_quality": "...",
        "self_correction": "...",
        "clarity": "..."
    }},
    "total_score": 100
}}
"""

class SelfOptimizingFramework:
    """
    一個框架，讓 LLM 能夠自我優化解決複雜問題，並透過評估機制追蹤其表現。
    """

    def __init__(self):

        #gemini的initialization
        self.client = google_client()
        self.gpt_client = OpenAI()


        # 初始化歷史紀錄
        self.history = []
        self.scores = []
        self.reasoning_evals = []
        self.best_score = float('inf')
        self.best_solution_details = ""
        self.iteration_count = 0
        self.max_history = 0
        self.temperature = 0.7
        self.reasoning_times = []
        self.evaluation_times = []


        # 設定日誌記錄器，同時輸出到檔案和控制台
        self.logger = logging.getLogger("SelfOptimizingFramework")
        self.logger.setLevel(logging.INFO)

        # 防止重複添加 handler
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 建立帶有時間戳的日誌檔名
        log_filename = f"logs/framework_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # 檔案 Handler
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # 控制台 Handlesr
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter('%(message)s') # 控制台輸出可以簡潔一些
        stream_handler.setFormatter(stream_formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def _call_llm(self, prompt: str, temp: float, model_name:str, ) -> Tuple[genai.types.GenerateContentResponse, float]:
        
        """一個通用的 LLM API 呼叫函式，加上 500 retry 補救"""
        self.logger.info(f"\n--- [呼叫 LLM] Iteration {self.iteration_count} ---")
        self.logger.info(f"--- [傳送的 Prompt] ---\n{prompt}\n--------------------")
        start_time = time.perf_counter()
        max_retries = 3
        
        #嘗試三次，因為有時候會跑出500 error
        for attempt in range(max_retries):
            try:
                if str.upper(model_name[0:6]) == "GEMINI":
                    #使用gemini進行推理
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=[prompt],
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=-1), 
                            tools=[types.Tool(code_execution=types.ToolCodeExecution())],
                            temperature=temp
                        )
                    )
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    if response.text is not None:
                        self.logger.info(f"--- [LLM 回應] (耗時: {duration:.2f} 秒) ---\n{response.text}\n--------------------")

                elif str.upper(model_name[0:3]) == "GPT":
                    #使用gpt進行推理
                    response = openai.chat.completions.create(
                        model=model_name,
                        messages=[prompt],
                        tools=[{"type": "code_interpreter"}],
                        temperature=temp
                    )
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    if response.choices[0].message.content is not None:
                        self.logger.info(f"--- [LLM 回應] (耗時: {duration:.2f} 秒) ---\n{response.choices[0].message.content}\n--------------------")
                else:
                    raise NotImplementedError
                return response, duration
            except Exception as e:
                self.logger.warning(f"[警告] 第 {attempt+1}/{max_retries} 次遇到  500 Internal Error: {e}")
                time.sleep(2 + attempt * 2)  # 指數後退，第一次等 2s，第二次等 4s...

        # 如果全部都失敗
        self.logger.error(f"[錯誤] 連續 {max_retries} 次都遇到伺服器錯誤，跳過此輪。")
        return genai.types.GenerateContentResponse(candidates=[]), 0.0


    def _parse_classification(self, text: str) -> str:
        """從 Step 1 的回應中解析出問題分類"""
        match = re.search(r"Classification:\s*(.*)", text, re.IGNORECASE)
        if match:
            classification = match.group(1).strip()
            if re.search(r'indefinite|heuristic|np-hard', classification, re.I):
                return "Indefinite Algorithm Required"
            if re.search(r'definite|dp|exact', classification, re.I):
                return "Definite Algorithm Feasible"
            
        # 如果沒有明確匹配，預設為更複雜的路徑
        self.logger.warning("在回應中找不到明確的分類，預設為 'Indefinite Algorithm Required'。")

        return "Indefinite Algorithm Required"

    def _parse_full_response(self, response: genai.types.GenerateContentResponse) -> dict:
            """
            從模型回應中安全地解析出所有部分，並確保回傳值的型別正確。
            """
            text_parts = []
            code = ""
            output = ""
            score = None
            # 防禦性檢查：確保 response 和 candidates 存在
            if not response or not response.candidates:
                self.logger.warning("收到了空的或無效的回應物件。")
                return {
                    "reasoning": "", "code": "", "output": "", "score": None
                }

            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
                if hasattr(part, 'executable_code') and part.executable_code:
                    code = part.executable_code.code or ""
                if hasattr(part, 'code_execution_result') and part.code_execution_result:
                    output = part.code_execution_result.output or ""

            full_reasoning = "\n".join(text_parts).strip()

            # --- 優化後的分數解析邏輯 ---    
            def get_last_line(text):
                """輔助函式：取得文字中最後一個非空的行"""
                if not text:
                    return ""
                lines = text.strip().split('\n')
                for line in reversed(lines):
                    if line.strip():
                        return line.strip()
                return ""

            last_output_line = get_last_line(output)
            last_reasoning_line = get_last_line(full_reasoning)

            # 策略 1: 優先嘗試將最後一行直接轉換為數字 (最高信賴度)
            if last_output_line:
                try:
                    score = float(last_output_line)
                except ValueError:
                    pass  # 如果失敗，代表它不是純數字，繼續

            if score is None and last_reasoning_line:
                try:
                    score = float(last_reasoning_line)
                except ValueError:
                    pass  # 如果失敗，繼續

            # 策略 2 (備用方案): 如果最高信賴度的方法失敗，則在最後一行中搜尋數字
            if score is None:
                self.logger.info("在最後一行找不到純數字，嘗試在行內搜尋數字作為備案。")
                if last_output_line:
                    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", last_output_line)
                    if numbers:
                        score = float(numbers[-1]) # 取最後一個找到的數字
                
                if score is None and last_reasoning_line:
                    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", last_reasoning_line)
                    if numbers:
                        score = float(numbers[-1]) # 取最後一個找到的數字

            return {
                "reasoning": full_reasoning,
                "code": code,
                "output": output,
                "score": score
            }
    
    def _evaluate_reasoning(self, reasoning_text: str, model_name:str,) -> dict:
        
        """使用 LLM-as-a-Judge 來評估推理品質"""
        self.logger.info("\n--- [啟動 Evaluator] 正在評估推理品質 ---")
        prompt = EVALUATION_PROMPT_TEMPLATE.format(reasoning_text=reasoning_text)
        try:
            time.sleep(2)
            start_time = time.perf_counter()
            if str.upper(model_name[0:6]) == "GEMINI":
                #使用gemini當作evaluator
                evaluator_model = genai.GenerativeModel(model_name)
                response = evaluator_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        response_mime_type="application/json"
                    )
                )
                answer = response.text
            else:
                #使用gpt當作evaluator
                response = self.gpt_client.chat.completions.create(
                    model = model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "type": "object",
                            "properties": {
                                "scores": {
                                    "type": "object",
                                    "properties": {
                                        "problem_understanding": {"type": "integer"},
                                        "strategic_planning": {"type": "integer"},
                                        "implementation_quality": {"type": "integer"},
                                        "self_correction": {"type": "integer"},
                                        "clarity": {"type": "integer"},
                                    },
                                    "required": [
                                        "problem_understanding",
                                        "strategic_planning",
                                        "implementation_quality",
                                        "self_correction",
                                        "clarity"
                                    ]
                                },
                                "justifications": {
                                    "type": "object",
                                    "properties": {
                                        "problem_understanding": {"type": "string"},
                                        "strategic_planning": {"type": "string"},
                                        "implementation_quality": {"type": "string"},
                                        "self_correction": {"type": "string"},
                                        "clarity": {"type": "string"},
                                    },
                                    "required": [
                                        "problem_understanding",
                                        "strategic_planning",
                                        "implementation_quality",
                                        "self_correction",
                                        "clarity"
                                    ]
                                },
                                "total_score": {"type": "integer"}
                            },
                            "required": ["scores", "justifications", "total_score"]
                        }
                    }
                )
                answer = response.choices[0].message.content
            end_time = time.perf_counter()
            duration = end_time - start_time
            if answer is not None:
                eval_result = json.loads(answer)
                self.logger.info(f"評估完成。總分: {eval_result.get('total_score')}/100 (耗時: {duration:.2f} 秒)")
                self.logger.info(f"詳細評分: {json.dumps(eval_result.get('scores'), indent=2)}")
                return eval_result, duration
            else:
                return "ERROR", duration
        except Exception as e:
            self.logger.error(f"評估推理時發生錯誤: {e}")
            return {"total_score": 0, "error": str(e)}

    def _plot_progress(self):
        """將數值分數和推理品質分數視覺化"""
        if not self.scores or not self.reasoning_evals:
            self.logger.info("沒有足夠的數據來生成圖表。")
            return IndexError

        # 計算迭代次數並建立軸線
        num_iterations = len(self.scores)
        num_reasoning_evals = len(self.reasoning_evals)
        
        # 驗證數據一致性
        self.logger.info(f"數值分數記錄: {num_iterations} 次")
        self.logger.info(f"推理品質評估: {num_reasoning_evals} 次")
        
        # 建立 2x1 的子圖
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 12))
        fig.suptitle('自我優化進程追蹤 (Self-Optimization Progress Tracking)', fontsize=16)

        # --- 圖 1: 分數 vs. 推理品質 ---
        ax1.set_title('分數與推理品質演進 (Score vs. Reasoning Quality)')
        
        # 左側 Y 軸：數值分數 (越低越好)
        color = 'tab:red'
        ax1.set_xlabel('迭代次數 (Iteration)')
        ax1.set_ylabel('問題解數值 (Numerical Score)', color=color)
        
        # 數值分數從第2次迭代開始（因為第1次只有推理品質）
        numerical_iterations = range(2, num_iterations + 2)  # 第2次到第(num_iterations+1)次
        ax1.plot(numerical_iterations, self.scores, 'o-', color=color, label='Numerical Score')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # 標示出最佳分數（最低分數）
        if self.scores:
            min_score_val = min(self.scores)
            min_score_idx = self.scores.index(min_score_val)
            # 調整索引以對應正確的迭代次數
            ax1.scatter(min_score_idx + 2, min_score_val, s=150, facecolors='none', 
                    edgecolors='gold', linewidth=2, label=f'Best Score: {min_score_val:.2f}')

        # 右側 Y 軸：推理品質分數 (越高越好)
        ax2 = ax1.twinx()
        color = 'tab:blue'
        reasoning_scores = [e.get('total_score', 0) for e in self.reasoning_evals]
        ax2.set_ylabel('推理品質分數 (Reasoning Quality Score)', color=color)
        
        # 推理品質分數從第1次迭代開始
        reasoning_iterations = range(1, num_reasoning_evals + 1)
        ax2.plot(reasoning_iterations, reasoning_scores, 's--', color=color, label='Reasoning Score')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 110)  # 分數範圍 0-100，稍微放寬到 110 以便顯示
        
        # 整合兩個軸的圖例
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
        
        # 添加網格線
        ax1.grid(True)
        
        # 設定 X 軸刻度 - 涵蓋所有迭代次數
        max_iterations = max(num_iterations + 1, num_reasoning_evals)  # +1 因為數值分數從第2次開始
        all_iterations = range(1, max_iterations + 1)
        ax1.set_xticks(all_iterations)
        ax1.set_xlim(0.5, max_iterations + 0.5)  # 設定 X 軸範圍

        # --- 圖 2: 時間成本分析 ---
        ax3.set_title('每輪迭代時間成本分析 (Time Cost Analysis per Iteration)')
        ax3.set_xlabel('迭代次數 (Iteration)')
        ax3.set_ylabel('耗時 (秒) (Time in Seconds)')
        
        # 確保時間數據與迭代次數一致
        max_time_iterations = max(len(self.reasoning_times), len(self.evaluation_times))
        
        # 準備堆疊長條圖數據 - 確保數據長度與最大迭代次數一致
        reasoning_times_padded = self.reasoning_times + [0] * (max_time_iterations - len(self.reasoning_times))
        evaluation_times_padded = self.evaluation_times + [0] * (max_time_iterations - len(self.evaluation_times))

        # 繪製堆疊長條圖
        time_iterations = range(1, max_time_iterations + 1)
        ax3.bar(time_iterations, reasoning_times_padded, label='推理耗時 (Reasoning Time)', color='coral')
        ax3.bar(time_iterations, evaluation_times_padded, bottom=reasoning_times_padded, 
            label='評估耗時 (Evaluation Time)', color='skyblue')
        
        # 在長條圖上標示總時間
        totals = [i + j for i, j in zip(reasoning_times_padded, evaluation_times_padded)]
        for i, total in enumerate(totals):
            if total > 0:
                ax3.text(i + 1, total + max(totals) * 0.01, f'{total:.1f}s', ha='center', fontsize=9)

        # 設定 X 軸刻度和圖例
        ax3.set_xticks(time_iterations)
        ax3.set_xlim(0.5, max_time_iterations + 0.5)
        ax3.legend()
        ax3.grid(True, axis='y')

        # 調整布局並儲存圖表
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plot_filename = f"progress_chart_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"進度圖表已儲存至 {plot_filename}")
        plt.show()

    def _validate_data_consistency(self):
        """驗證各種數據的一致性"""
        scores_count = len(self.scores) if self.scores else 0
        reasoning_count = len(self.reasoning_evals) if self.reasoning_evals else 0
        reasoning_time_count = len(self.reasoning_times) if self.reasoning_times else 0
        eval_time_count = len(self.evaluation_times) if self.evaluation_times else 0
        
        self.logger.info("=== 數據一致性檢查 ===")
        self.logger.info(f"數值分數記錄: {scores_count}")
        self.logger.info(f"推理品質評估: {reasoning_count}")
        self.logger.info(f"推理時間記錄: {reasoning_time_count}")
        self.logger.info(f"評估時間記錄: {eval_time_count}")
        
        # 檢查預期的關係
        if reasoning_count > 0 and scores_count > 0:
            if reasoning_count != scores_count + 1:
                self.logger.warning(f"數據不一致: 推理評估({reasoning_count}) 應該比數值分數({scores_count})多1次")
        
        return {
            'scores': scores_count,
            'reasoning': reasoning_count,
            'reasoning_time': reasoning_time_count,
            'eval_time': eval_time_count
        }


    def run(self, model_name:str,task_description: str, points: np.array, max_iterations: int = 5, no_improvement_threshold: int = 3, max_history_length: int = 5,temp=0.7):
        run_start_time = time.perf_counter()
        """
        執行完整的自我優化流程。
        Args:
            task_description: 任務的自然語言描述。
            initial_data: 任務所需的初始數據 (例如 TSP 的座標點)。
            max_iterations: 最大迭代次數。
            no_improvement_threshold: 連續多少次沒有進步就提早停止。
        """
        # --- STEP 1: 推理分類與初步設計 ---
        initial_data = "data = " + np.array2string(points, separator=', ').replace('\n', '')
        self.iteration_count = 1
        self.logger.info("="*20 + " 開始新的自我優化流程 " + "="*20)
        self.logger.info(f"任務: {task_description.strip()}")
        self.logger.info(f"最大迭代次數: {max_iterations}, 無進步停止閾值: {no_improvement_threshold}")
        
        #建立prompt內容
        prompt_step1 = STEP_1_PROMPT_TEMPLATE.format(task_description=task_description)
        
        #使用llm進行推理
        response_step1, r_time1 = self._call_llm(prompt_step1, temp, model_name)
        reasoning_step1 = response_step1.text or "ERROR"
        
        #分析推理結果和類別
        classification = self._parse_classification(reasoning_step1)
        self.logger.info(f"STEP 1 分析完成。 問題類型被分類為: {classification}")
        
        #將此次的推理過程進行評估
        eval_step1, e_time1 = self._evaluate_reasoning(reasoning_step1, model_name=model_name)
        
        #最後將處理結果進行儲存
        self.reasoning_evals.append(eval_step1)
        self.history.append({"iteration": 1, "type": "Analysis", "reasoning": reasoning_step1, "eval": eval_step1, "r_time": r_time1, "e_time": e_time1})
        self.reasoning_times.append(r_time1)
        self.evaluation_times.append(e_time1)

        # --- STEP 2: 演算法實作與首次結果 ---
        self.iteration_count = 2
        
        #建立prompt內容
        task_with_data = f"{reasoning_step1}\n\nHere is the data to use:\n{initial_data}"
        prompt_step2 = STEP_2_PROMPT_TEMPLATE.format(classification=classification)
        full_prompt_step2 = f"Based on your previous analysis:\n{task_with_data}\n\nNow, follow these instructions:\n{prompt_step2}"

        #呼叫LLM進行推理
        response_step2, r_time2 = self._call_llm(full_prompt_step2, temp, model_name)
        
        #將結果進行解析
        parsed_data2 = self._parse_full_response(response_step2)
        code = parsed_data2["code"]
        output = parsed_data2["output"]
        score = parsed_data2["score"]
        
        # 即使模型沒給文字，reasoning 也會是空字串，而不是 None
        reasoning_step2 = parsed_data2["reasoning"] or "ERROR"
        
        #如果有正常回答則處理
        if score is not None and score >0:
            self.scores.append(score)
            self.best_score = score
            self.best_solution_details = f"Iteration 1: Score={score}, Code:\n{code}\nOutput:\n{output}"
            self.logger.info(f"STEP 2 首次執行完成。分數: {score}")
        else:
            self.logger.warning("STEP 2 警告：首次執行未能獲取有效分數。")
            # 添加一個懲罰性分數以保持陣列長度一致
            self.scores.append(self.best_score * 1.2 if self.best_score != float('inf') else 10000)

        #將推理過程進行評估
        eval_step2, e_time2 = self._evaluate_reasoning(reasoning_step2, model_name=model_name)
        self.reasoning_evals.append(eval_step2)
        self.history.append({"iteration": 2, "type": "Initial Implementation", "reasoning": reasoning_step2, "code": code, "output": output, "score": score, "eval": eval_step2, "r_time": r_time2, "e_time": e_time2})
        self.reasoning_times.append(r_time2)
        self.evaluation_times.append(e_time2)

        # --- STEP 3: 迭代優化循環 ---
        no_improvement_count = 0
        #開始迭代處理並期待能夠優化
        for i in range(3, max_iterations + 2):
            
            #如果連續N次的輸出數值結果沒有進步則停止優化
            self.iteration_count = i
            if no_improvement_count >= no_improvement_threshold:
                self.logger.info(f"\n連續 {no_improvement_threshold} 次沒有進步，提前停止迭代。")
                break
            
            # 增加歷史紀錄
            history_log = "\n".join([f"- Iteration {h['iteration']}: Score={h.get('score', 'N/A')}, Strategy: {h['reasoning']}" for h in self.history if 'score' in h])
            
            # 根據分類選擇不同的 Prompt
            if classification == "Definite Algorithm Feasible":
                # Definite Algorithm 演算法迴圈
                prompt_template = STEP_3A_PROMPT_TEMPLATE
                last_attempt = self.history[-1]
                prompt_step3 = prompt_template.format(
                    algorithm_description=reasoning_step1,
                    code=last_attempt.get('code', ''),
                    result=last_attempt.get('output', '')
                )
            else: 
                # Indefinite Algorithm Required
                prompt_template = STEP_3B_PROMPT_TEMPLATE
                prompt_step3 = prompt_template.format(
                    best_score=self.best_score,
                    history_log=history_log
                )

            # 組合完整的上下文prompt
            full_prompt_step3 = f"This is iteration {i}. Your task is to improve upon previous results.\n\n{prompt_step3}\n\nRemember to use the same initial data:\n{initial_data}"
            
            # call llm獲得解答
            response_step3, r_time_i = self._call_llm(full_prompt_step3,temp, model_name)
            
            #將輸出的結果進行parsing
            parsed_data3 = self._parse_full_response(response_step3)
            code = parsed_data3["code"]
            output = parsed_data3["output"]
            score = parsed_data3["score"]
            # 即使模型沒給文字，reasoning 也會是空字串，而不是 None
            reasoning_step3 = parsed_data3["reasoning"] or "ERROR"
            
            #進行推理內容的評論
            eval_step3, e_time_i = self._evaluate_reasoning(reasoning_step3, model_name=model_name)
            self.reasoning_times.append(r_time_i)
            self.evaluation_times.append(e_time_i)
            self.reasoning_evals.append(eval_step3)
            
            # 如果模型自己覺得OK就OK
            if "FINISHED" in reasoning_step3:
                self.logger.info("\n模型回傳 'FINISHED'，結束優化流程。")
                break
            
            # 紀錄分數，如果破紀錄則記錄此紀錄，阿連續N次都沒有進步的話也停止優化
            if score is not None and score >0:
                self.scores.append(score)
                self.logger.info(f"Iteration {i} 完成。分數: {score} (歷史最佳: {self.best_score})")
                if score < self.best_score:
                    self.best_score = score
                    self.best_solution_details = f"Iteration {i}: Score={score}, Code:\n{code}\nOutput:\n{output}"
                    no_improvement_count = 0
                    self.logger.info(f"*** 新的最佳解! ***")
                else:
                    no_improvement_count += 1
                    self.logger.info(f"未找到更優解。連續未進步次數: {no_improvement_count}")

            else:
                self.logger.warning(f"Iteration {i} 警告：未能獲取有效分數。")
                self.scores.append(self.scores[-1] if self.scores else self.best_score) # 重複上一個分數
                no_improvement_count += 1
                self.logger.info(f"計為一次未進步。連續未進步次數: {no_improvement_count}")


            self.history.append({"iteration": i, "type": "Optimization", "reasoning": reasoning_step3, "code": code, "output": output, "score": score, "eval": eval_step3, "r_time": r_time_i, "e_time": e_time_i})

            if len(self.history) > max_history_length:
                self.history.pop(0)

        # --- 最終結果 ---
        run_end_time = time.perf_counter()
        total_run_time = run_end_time - run_start_time
        self.logger.info("\n\n" + "="*20 + " 優化流程結束 " + "="*20)
        self.logger.info(f"總執行時間: {total_run_time:.2f} 秒")
        self.logger.info(f"總共執行了 {len(self.scores)} 次有效的迭代。")
        self.logger.info(f"找到的最佳分數為: {self.best_score}")
        self.logger.info("\n--- [最佳解的詳細資訊] ---\n" + self.best_solution_details)
        self.logger.info("\n---------------------\n")
        if task_description =="""
            Solve the Traveling Salesman Problem (TSP) for a given set of 2D points.
            The goal is to find the shortest possible tour that visits each point exactly once and returns to the origin point.
            The distance between points is the Euclidean distance.
            """:
            self.logger.info("額外加碼:與最佳解之間的距離")
            DP_start_time = time.perf_counter()
            dp_calculation = DP4TSP(points)
            dp_calculation.run(points)
            total_DP_run_time = time.perf_counter() - DP_start_time
            self.logger.info(f"DP執行時間: {total_DP_run_time:.2f} 秒")

        # 在繪製圖表前可以先驗證數據
        data_stats = self._validate_data_consistency()
        if data_stats['reasoning'] != data_stats['scores'] + 1:
            self.logger.warning("數據關係可能不正確")

        self._plot_progress()


if __name__ == '__main__':
    try:
        # ===== Initialization =====
        task_input = str(input("請輸入你的問題, 如果想看預設問題，請輸入'TSP':"))
        
        # 輸入問題:預設是TSP版本的問題，並且配上最優解答來參考其表現，但是也可以輸入各式各樣可以用code優化的問題
        
        if task_input == "TSP":
            TASK_DESCRIPTION = """
            Solve the Traveling Salesman Problem (TSP) for a given set of 2D points.
            The goal is to find the shortest possible tour that visits each point exactly once and returns to the origin point.
            The distance between points is the Euclidean distance.
            """
            data_points = np.random.rand(20, 2)
        # 如果不是預設問題的話，那麼要輸入問題敘述和資料
        else:
            TASK_DESCRIPTION = task_input
            data_points = str(input("請輸入相關的資料，用逗號分割(可略)"))
        print(data_points)

        # configure
        load_dotenv()
        model_name = str(input("請輸入你想使用的model名稱(gemini-2.5-pro, gpt-4o...)"))
        if str.upper(model_name[0:6]) == "GEMINI":
            GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY")
            if GOOGLE_API_KEY == "YOUR_GEMINI_API_KEY":
                print("請設定您的 GOOGLE_API_KEY 環境變數或在程式碼中直接替換 'YOUR_GEMINI_API_KEY'")
                raise IndexError
            genai.configure(api_key=GOOGLE_API_KEY)
        elif str.upper(model_name[0:3]) == "GPT":
            OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
            if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
                print("請設定您的 YOUR_OPENAI_API_KEY 環境變數或在程式碼中直接替換 'YOUR_OPENAI_API_KEY'")
                raise IndexError
            openai.api_key = OPENAI_API_KEY
        else:
            print("請輸入正確的模型名稱。")
            raise NotImplementedError

        print("============================")
        # 創建並運行框架
        framework = SelfOptimizingFramework()
        framework.run(
            model_name=model_name,
            task_description=TASK_DESCRIPTION,
            points=data_points,
            **framework_config  # ← 把 config.py 裡的參數展開塞進來
        )
    except Exception as e:
        print(f"發生錯誤: {e}")
        print(f"錯誤詳情:\n{traceback.format_exc()}")