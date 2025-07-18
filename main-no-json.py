import google.generativeai as genai
from google.genai import types
from google.genai import Client as google_client
import re
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv
import logging
import datetime
import time
import traceback
from dp_for_tsp import DP4TSP
import openai
from openai import OpenAI
from config import framework_config
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft JhengHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# === Prompt Templates ===
matplotlib.use("Agg")  # 強制用非 GUI backend


AGENT_A_PROMPT_TEMPLATE = """
You are an algorithmic expert who prefers definite, optimal strategies such as Dynamic Programming or Divide-and-Conquer.

Given the following task:

{task_description}

Argue why a **definite algorithm** is more suitable for this task. Explain the benefits, provide potential algorithmic outlines, and prepare a rebuttal for typical heuristic claims.

Return your response in **pure JSON format**, with the following structure, and nothing else:

{
  "explanation": "<your explanation here>",
  "confidence": <your confidence score as a float between 0 and 1>
}

Do not include any markdown formatting, headings, or extra commentary. Only return the JSON object.
"""

AGENT_B_PROMPT_TEMPLATE = """
You are a heuristic strategy expert who uses Genetic Algorithms, Simulated Annealing, etc.

Given the following task:

{task_description}

Argue why a **heuristic algorithm** is more suitable for this task. Highlight scalability, flexibility, and robustness. Prepare to counter common critiques from the definite algorithm camp.

Return your response in **pure JSON format**, with the following structure, and nothing else:

{
  "explanation": "<your explanation here>",
  "confidence": <your confidence score as a float between 0 and 1>
}

Do not include any markdown formatting, headings, or extra commentary. Only return the JSON object.

"""

CRITIQUE_PROMPT = """
You are AgentA. You have made the following argument:
{self_argument}
and here is the confidence of your argument:
{confidence}
Here is the argument made by AgentB:
{opponent_argument}
Critique it from your definite algorithm perspective.
Assess if your confidence remains the same, increases, or decreases.
Also, give a persuasion score (0 to 1) on how convincing AgentB's points are.
AT THE BOTTOM OF THE text, format your response as:
Output:
- Your critique
- Updated Confidence: [0~1]
- Persuasion Score: [0~1]
"""

ARBITER_PROMPT_TEMPLATE = """
You are an impartial judge reviewing a debate between two algorithm experts regarding the best strategy for solving the following task:
{task_description}

Agent A Critique of B:
{agent_a_critique}

Agent B Critique of A:
{agent_b_critique}

Final confidence scores:
- Agent A: {agent_a_confidence} (How confident Agent A is in their argument)
- Agent B: {agent_b_confidence} (How confident Agent B is in their argument)

Persuasion scores:
- A convinced by B: {agent_a_persuasion} (How much Agent A is convinced by Agent B's argument)
- B convinced by A: {agent_b_persuasion} (How much Agent B is convinced by Agent A's argument)

Decide whether the task should be solved using a:
- definite algorithm (output: "definite")
- heuristic algorithm (output: "heuristic")

Explain your choice and try to provide a rationale for your decision.
Then, give a final strategy based on the arguments presented. Optimal and Specific algorithm or method can be mentioned.
AT THE BOTTOM OF THE text, format your response as:
FINAL STRATEGY SHOULD BE AT the BOTTOM OF THE text.
Output:
- Your Explanation and Rationale, with algorithmic details if applicable.
- Final Strategy: [definite|heuristic]
"""
#- hybrid or uncertain approach (output: "hybrid")
# Agent B (Heuristic) says:
# {agent_b_argument}
# 刪除A跟B本來的立場
STEP_ITERATIVE_PROMPT_DEFINITE = """
You are an algorithmic optimization expert using a **definite strategy**.

Your task is to **refine and finalize** an existing exact algorithm for the following problem:
{task_description}

With the final strategy:
{final_strategy} and {final_explanation}.

Then, implement the algorithm in Python code.

You previously attempted this solution:
```python
{prev_code}
```
It produced:
```python
{prev_result}
```
Result Score: {prev_score}
Your Responsibilities:

- Review & Reflect: Is the algorithm optimal and final? If not, how can it be improved?
- Enhance & Explain: Refactor or restructure it, improve clarity and efficiency (but preserve correctness).
- Verify: Run the new version and compare it with the previous result. Is the result identical, or improved?

You may stop refining if:
- You believe this is the best possible solution.
- You detect diminishing returns.
If you believe further significant improvement is unlikely, you can conclude your entire response with the single word "FINISHED".
Keep your output quite brief, concise and insightful.

Please output the scores or inference results in the last line without any other values.
"""
# Here is the history of your previous attempts:
# {history_log}
#先不放
STEP_ITERATIVE_PROMPT_HEURISTIC = """
You are optimizing a problem using a **heuristic strategy**.

The task is:
{task_description}
With the final strategy:
{final_strategy} and {final_explanation}.

Then, implement the algorithm in Python code.
Your current best solution is:
```python
{prev_code}
It produced:
```python
{prev_result}
```
Result Score: {prev_score}
Now please:
- Critique the last approach and identify weaknesses.
- Propose a concrete change (e.g. parameter tuning, switching to a different heuristic, using hybrid metaheuristics).
- Implement the new code.
- Run the solution and report the new result.

If you believe further significant improvement is unlikely, you can conclude your entire response with the single word "FINISHED".
Keep your output quite brief, concise and insightful.

Please output the scores or inference results in the last line without any other values.
"""

# STEP_ITERATIVE_PROMPT_HYBRID = """
# You are solving a problem that requires a **hybrid approach**, combining both definite and heuristic methods.

# The task is:
# {task_description}
# With the final strategy:
# {final_strategy} and {final_explanation}.

# Then, implement the algorithm in Python code.
# Here is the history of your previous attempts:
# {history_log}

# Your current best solution is:
# ```python
# {prev_code}
# It produced:
# ```python
# {prev_result}
# ```
# Result Score: {prev_score}
# Now please:
# - Analyze which component (definite or heuristic) can be improved further.

# - Explain your plan (e.g., replace heuristic inner loop with DP refinement).

# - Update the implementation and execute it.

# - Report whether the result improves.

# If you believe further significant improvement is unlikely, you can conclude your entire response with the single word "FINISHED".
# Keep your output quite brief, concise and insightful.

# Please output the scores or inference results in the last line without any other values.
# """

# 這是 LLM-as-a-Judge 的評分 Prompt
EVALUATION_PROMPT_TEMPLATE = """
You are an expert evaluator assessing the reasoning quality of an AI model's response to a complex problem-solving task.
Please evaluate the following response based on five criteria. For each criterion, provide a score from 0 to 20 and a brief justification.

**HISTORY reasoning context - score pairs**: you can refer to the previous content to determine whether this reasoning context is progressing or not.
{history_pairs}
**The AI's Response to Evaluate:**
---
{reasoning_text}
---

**Evaluation Criteria:**

1.  **Problem Understanding & Analysis (0-20)**: How well did the model comprehend the problem's constraints, goals, and underlying computational complexity?
2.  **Strategic Planning (0-20)**: Was the proposed algorithm or plan of action logical, well-justified, and appropriate for the problem? For iterative steps, does the plan build intelligently on previous results?
3.  **Self-Correction & Iteration (0-20)**: How effectively did the model analyze its previous results to propose specific, actionable improvements? (This is crucial for heuristic optimization). If this is the first step, score based on the quality of the proposed future directions.
4.  **Clarity and Communication (0-20)**: Was the explanation clear, concise, and easy for a human to understand?
5.  **Implementation Quality (0-20)**: 
    - If code is offered, assess its correctness, efficiency, and clarity. Does it accurately reflect the proposed strategy?
    - If code is not offered, consider the effectiveness of the plan. Is it feasible and possible for LLM to generate suitable answer? Will it be hard for it to implement?

Please provide your evaluation in a JSON format like this:
{{
    "scores": {{
        "problem_understanding": 20,
        "strategic_planning": 20,
        "self_correction": 20,
        "clarity": 20
        "implementation_quality": 20,
    }},
    "justifications": {{
        "problem_understanding": "...",
        "strategic_planning": "...",
        "self_correction": "...",
        "clarity": "..."
        "implementation_quality": "...",
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
        self.history_log = []
        self.scores = []
        self.reasoning_evals = []
        self.best_score = float('inf')
        self.best_solution_details = ""
        self.iteration_count = 0
        self.max_history = 0
        self.temperature = 0.7
        self.reasoning_times = []
        self.evaluation_times = []
        
        self.plt_dict = {}
        

        # 確保 logs 資料夾存在
        if not os.path.exists("logs"):
            os.makedirs("logs")

        # # 防止重複添加 handler
        # if self.logger.hasHandlers():
        #     self.logger.handlers.clear()
        self.logger = logging.getLogger("SelfOptimizingFramework")
        self.logger.setLevel(logging.INFO)
        # 只加新的 handler，如果沒有的話再加（不要清除 log_stream handler）
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            log_filename = f"logs/framework_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_filename, encoding='utf-8')
            LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
            file_formatter = logging.Formatter(LOG_FORMAT)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            stream_handler = logging.StreamHandler()
            stream_formatter = logging.Formatter('%(message)s')
            stream_handler.setFormatter(stream_formatter)
            self.logger.addHandler(stream_handler)
        for h in self.logger.handlers:
            print(f"[Handler] {type(h)} → {h}")
        # 建立帶有時間戳的日誌檔名
        log_filename = f"logs/framework_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # 檔案 Handler
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        
        # 控制台 Handlesr
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter('%(message)s') # 控制台輸出可以簡潔一些
        stream_handler.setFormatter(stream_formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def _call_llm_api(self, prompt: str, model_name: str, temp: float, generate_code: bool) -> str:
        """
        一個通用的 LLM API 呼叫函式，整合了重試邏輯。
        成功時返回 LLM 的文字回應，失敗時返回錯誤訊息。
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if str.upper(model_name).startswith("GEMINI"):
                    if generate_code:
                        response = self.client.models.generate_content(
                            model=model_name,
                            contents=[prompt],
                            config=types.GenerateContentConfig(
                                thinking_config=types.ThinkingConfig(thinking_budget=-1),
                                temperature=temp,
                                tools=[types.Tool(code_execution=types.ToolCodeExecution())],
                            )
                        )
                    else:
                        response = self.client.models.generate_content(
                            model=model_name,
                            contents=[prompt],
                            config=types.GenerateContentConfig(
                                thinking_config=types.ThinkingConfig(thinking_budget=-1),
                                temperature=temp,
                            )
                        )
                    self.logger.info(f"--- [Gemini API 回應] ---\n{response.text}\n--------------------")
                    return response

                elif str.upper(model_name).startswith("GPT"):
                    if not generate_code:
                        response = openai.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=temp
                        )
                    else:
                        response = self.gpt_client.responses.create(
                            model=model_name,
                            input=prompt,
                            tools=[
                                {"type": "code_interpreter", "container": {"type": "auto"}}
                            ],
                            tool_choice="required",  # ⬅️ 關鍵！要求一定要用 tool
                            temperature=temp,
                        )
                    self.logger.info(f"--- [GPT API 回應] ---\n{response.choices[0].message.content}\n--------------------")
                    return response
                else:
                    raise NotImplementedError(f"Model '{model_name}' is not supported.")

            except Exception as e:
                self.logger.warning(f"[警告] API 呼叫失敗，第 {attempt + 1}/{max_retries} 次嘗試。錯誤: {e}")
                if attempt + 1 < max_retries:
                    time.sleep(2 + attempt * 2) # 指數後退

        self.logger.error(f"[錯誤] API 呼叫在 {max_retries} 次嘗試後徹底失敗。")
        return "ERROR: API call failed after multiple retries."


    def _run_agent(self, prompt_template: str, task_description: str, temp: float, model_name: str) -> dict:
            """產生論點並評估信心度"""
            self.iteration_count += 1
            self.logger.info(f"\n--- Iteration {self.iteration_count} : 正反方發表意見 ---")
            prompt = prompt_template.format(task_description=task_description)
            self.logger.info(f"--- [傳送的 Prompt] ---\n{prompt}\n--------------------")

            start_time = time.perf_counter()
            answer = self._call_llm_api(prompt, model_name, temp, False)
            if model_name.startswith("gemini"):
                answer = answer.text
            elif model_name.startswith("gpt"):
                answer = answer.choices[0].message.content or "ERROR: Empty response from GPT."
            if answer.startswith("ERROR:"):
                self.logger.error(f"--- [以上的LLM 回應失敗] (耗時: {time.perf_counter() - start_time:.2f} 秒) ----")
                return {"argument": answer, "confidence": 0.0}
            duration = time.perf_counter() - start_time

            if answer.startswith("ERROR:"):
                self.logger.error(f"--- [以上的LLM 回應失敗] (耗時: {duration:.2f} 秒) ----")
                return {"argument": answer, "confidence": 0.0}

            self.logger.info(f"\n--- [以上的LLM 回應] (耗時: {duration:.2f} 秒) ----\n")
            
            try:
                confidence = self.__find_confidence_score(answer)
                return {"argument": answer, "confidence": confidence}
            except (ValueError, IndexError) as e:
                self.logger.error(f"解析 'Confidence' 失敗: {e}")
                return {"argument": answer, "confidence": 0.0}


    def _run_critique(self, prompt_template: str, agent_self, agent_opponent, temp: float, model_name: str) -> dict:
        """產生批判性分析並更新信心度"""
        self.iteration_count += 1
        self.logger.info(f"\n--- Iteration {self.iteration_count} : 進行批判性分析 ---")
        prompt = prompt_template.format(
            self_argument=agent_self["argument"],
            opponent_argument=agent_opponent["argument"],
            confidence=agent_self["confidence"]
        )
        self.logger.info(f"--- [傳送的 Prompt] ---\n{prompt}\n--------------------")

        start_time = time.perf_counter()
        answer = self._call_llm_api(prompt, model_name, temp, False)
        if model_name.startswith("gemini"):
            answer = answer.text
        elif model_name.startswith("gpt"):
            answer = answer.choices[0].message.content or "ERROR: Empty response from GPT."
        if answer.startswith("ERROR:"):
            self.logger.error(f"--- [以上的LLM 回應失敗] (耗時: {time.perf_counter() - start_time:.2f} 秒) ----")
            return {"argument": answer, "confidence": 0.0}
        duration = time.perf_counter() - start_time

        if answer.startswith("ERROR:"):
            self.logger.error(f"--- [以上的LLM 回應失敗] (耗時: {duration:.2f} 秒) ----------")
            return {"argument": answer, "confidence": agent_self.get("confidence", 0.0), "persuasion": 0.0}

        self.logger.info(f"--- [以上的LLM 回應] (耗時: {duration:.2f} 秒) ----------")

        try:
            
            # 1. 初始化變數為預設值
            updated_conf = None
            persuasion = None
            critique_lines = []

            lines = answer.strip().splitlines()

            # 2. 逐行檢查與處理
            for line in lines:
                # 檢查是否為 "Updated Confidence" 行
                if "Updated Confidence" in line:
                    try:
                        # 使用 split(':', 1) 只切分第一個冒號，更安全
                        # 再用 re.sub 清理出數字
                        value_str = re.sub(r'[^\d.]', '', line.split(':', 1)[1])
                        if value_str:  # 確保清理後不是空字串
                            updated_conf = float(value_str)
                    except (IndexError, ValueError):
                        # 如果這行格式有誤 (如沒有冒號)，就忽略錯誤並繼續
                        pass
                        
                # 檢查是否為 "Persuasion Score" 行
                elif "Persuasion Score" in line:
                    try:
                        value_str = re.sub(r'[^\d.]', '', line.split(':', 1)[1])
                        if value_str:
                            persuasion = float(value_str)
                    except (IndexError, ValueError):
                        pass
                        
                # 如果都不是分數行，就當作是評論內容
                else:
                    critique_lines.append(line)

            # 3. 組合評論文字
            critique_text = "\n".join(critique_lines).strip()
            return {"argument": critique_text, "confidence": updated_conf, "persuasion": persuasion}
        except (ValueError, IndexError) as e:
            self.logger.error(f"解析 'Critique' 結果失敗: {e}")
            return {"argument": answer, "confidence": agent_self.get("confidence", 0.0), "persuasion": 0.0}
    def __find_confidence_score(self, answer: str) -> float | None:
        """
        在整個 answer 字串中搜尋第一筆 Confidence 分數。
        
        這個函式會：
        1. 逐行搜尋。
        2. 使用不分大小寫的方式尋找關鍵字 'confidence'。
        3. 清理整行文字，只留下數字和小數點。
        4. 找到第一個有效的數字後就回傳結果。
        5. 如果找不到，回傳 None。
        """
        # 1. 設定預設回傳值
        confidence = None
        
        # 2. 取得所有文字行
        lines = answer.strip().splitlines()
        
        # 3. 逐行遍歷
        for line in lines:
            # 4. 檢查該行是否包含關鍵字 (使用 .lower() 來忽略大小寫)
            if 'confidence' in line.lower():
                # 5. 清理整行文字，而不只是特定部分
                numeric_string = re.sub(r'[^\d.]', '', line)
                
                # 6. 確保清理後有內容，避免對空字串做 float() 轉換
                if numeric_string:
                    try:
                        # 7. 嘗試轉換成 float
                        confidence = float(numeric_string)
                        # 8. 成功找到第一個符合的目標，就停止迴圈
                        break
                    except ValueError:
                        # 如果清理後字串仍無法轉成 float (例如 "1.2.3")，就忽略並繼續尋找下一行
                        continue
                        
        return confidence

    def _parse_heuristic_byllm(self, response: str, model_name: str, temp: float) -> Dict[str, Any]:
        """
        解析 LLM 回應，提取推理、程式碼、輸出和分數。
        這個函式會根據不同的模型類型（Gemini 或 GPT）來處理回應。
        """
        classification_prompt = """base on the context provided, please classify the strategy into one of the following categories: definite, heuristic, or uncertain. Do not give any other description except the classification. Do not use any markdown format, just return the classification as a single word."""
        response += classification_prompt
        if str.upper(model_name).startswith("GEMINI"):
            strategy = str.lower(self._call_llm_api(response, model_name, temp, False).text.strip())
        elif str.upper(model_name).startswith("GPT"):
            strategy = str.lower(self._call_llm_api(response, model_name, temp, False).choices[0].message.content.strip())
        else:
            raise NotImplementedError(f"Model '{model_name}' is not supported.")
        has_heuristic = "heuristic" in strategy
        has_definite = "definite" in strategy

        if has_heuristic and not has_definite:
            return "heuristic"
        elif has_definite and not has_heuristic:
            return "definite"
        else: # 包含了兩種情況：兩個都有，或兩個都沒有
            return "uncertained"

    def _rationalize_strategy(self, prompt_template: str, task_description: str, agent_a, agent_b, critique_a, critique_b, temp: float, model_name: str) -> dict:
        """整合資訊並產生最終策略"""
        self.iteration_count += 1
        self.logger.info(f"\n--- Iteration {self.iteration_count} : 產生最終策略 ---")
        prompt = prompt_template.format(
            task_description=task_description,
            agent_a_argument=agent_a["argument"],
            agent_a_critique=critique_a["argument"],
            agent_b_argument=agent_b["argument"],
            agent_b_critique=critique_b["argument"],
            agent_a_confidence=critique_a["confidence"],
            agent_b_confidence=critique_b["confidence"],
            agent_a_persuasion=critique_a["persuasion"], # a evaluate b
            agent_b_persuasion=critique_b["persuasion"] # b evaluate a
        )
        self.logger.info(f"--- [傳送的 Prompt] ---\n{prompt}\n--------------------")

        start_time = time.perf_counter()
        answer = self._call_llm_api(prompt, model_name, temp, False)
        if model_name.startswith("gemini"):
            answer = answer.text
        elif model_name.startswith("gpt"):
            answer = answer.choices[0].message.content or "ERROR: Empty response from GPT."
        if answer.startswith("ERROR:"):
            self.logger.error(f"--- [以上的LLM 回應失敗] (耗時: {time.perf_counter() - start_time:.2f} 秒) ----")
            return {"argument": answer, "confidence": 0.0}
        duration = time.perf_counter() - start_time
        
        if answer.startswith("ERROR:"):
            self.logger.error(f"--- [以上的LLM 回應失敗] (耗時: {duration:.2f} 秒) --------------")
            raise ValueError(f"無法解析最終策略：{answer}")

        self.logger.info(f"--- [以上的LLM 回應] (耗時: {duration:.2f} 秒) -----------")
        
        try:
            strategy = self._parse_heuristic_byllm(answer, model_name, temp)
            # 解析邏輯
            return {"strategy": strategy, "explanation": answer}
        except (ValueError, IndexError) as e:
            self.logger.error(f"解析 'Strategy' 結果失敗: {e}")
            raise ValueError(f"無法解析最終策略：{answer}")
        
    def _get_last_line(self, text: str) -> str:
        """Helper function: gets the last non-empty line from a string."""
        if not text:
            return ""
        lines = text.strip().split('\n')
        # reversed() is more efficient than lines[::-1] for iterators
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        return ""

    def _find_score_in_text(self, text: str) -> Optional[float]:
        """
        Attempts to find a score in a block of text using two strategies.
        1. Try to extract a float from the last line using regex.
        2. If that fails, try finding any float-looking number in the line.
        """
        last_line = self._get_last_line(text)
        if not last_line:
            return None

        # Strategy 1: Search for number using regex
        match = re.search(r'[-+]?\d*\.\d+|\d+', last_line)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                self.logger.warning(f"Matched string '{match.group(0)}' could not be converted to float.")
        
        # Strategy 2: Find all numbers and return last one
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', last_line)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                self.logger.warning(f"Found numbers {numbers}, but could not convert the last one.")
        
        return None


    def _extract_parts_from_response(self, response) -> Dict[str, str]:
        """Extracts reasoning, code, and output from the raw response object."""
        reasoning_parts = []
        code = ""
        output = ""

        # The loop iterates over the parts and assigns them.
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                reasoning_parts.append(part.text)
            if hasattr(part, 'executable_code') and part.executable_code:
                code = part.executable_code.code or ""
            if hasattr(part, 'code_execution_result') and part.code_execution_result:
                output = part.code_execution_result.output or ""

        return {
            "reasoning": "\n".join(reasoning_parts).strip(),
            "code": code,
            "output": output,
        }

    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """
        Parses a model response safely, extracting reasoning, code, output, and a score.
        This function now coordinates calls to helper methods.
        
        Cognitive Complexity is drastically reduced.
        """
        # Guard clause for an invalid response.
        if not response or not getattr(response, 'candidates', None):

            self.logger.warning("Received an empty or invalid response object.")
            return {"reasoning": "", "code": "", "output": "", "score": None}

        # 1. Extract all text components into a simple dictionary.
        parts = self._extract_parts_from_response(response)
        
        # 2. Determine the score with a clear priority: check code output first, then reasoning.
        # The complex nested logic is now handled by the helper function.
        score = self._find_score_in_text(parts["output"])
        if score is None:
            score = self._find_score_in_text(parts["reasoning"])

        # 3. Assemble the final result.
        return {
            "reasoning": parts["reasoning"],
            "code": parts["code"],
            "output": parts["output"],
            "score": score
        }

    def _extract_parts_from_gpt_response(self, response) -> Dict[str, str]:
        reasoning = []
        code = ""
        output = ""

        for item in response.output:
            if item.type == "code_interpreter_call":
                code = item.code
                if item.outputs:
                    for out in item.outputs:
                        if out.type == "logs":
                            output = out.logs
            elif item.type == "message":
                for block in item.content:
                    if block.type == "output_text":
                        reasoning.append(block.text)

        return {
            "reasoning": "\n".join(reasoning).strip(),
            "code": code,
            "output": output
        }

    def _parse_gpt_response(self, response) -> Dict[str, Any]:
        parts = self._extract_parts_from_gpt_response(response)
        score = self._find_score_in_text(parts["output"]) or self._find_score_in_text(parts["reasoning"])
        return {
            "reasoning": parts["reasoning"],
            "code": parts["code"],
            "output": parts["output"],
            "score": score
        }
    def _evaluate_reasoning(self, reasoning_text: str, model_name:str, history_pairs:str,) -> dict:
        """使用 LLM-as-a-Judge 來評估推理品質"""
        self.logger.info("\n--- [啟動 Evaluator] 正在評估推理品質 ---")
        prompt = EVALUATION_PROMPT_TEMPLATE.format(reasoning_text=reasoning_text, history_pairs=history_pairs)
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
                                        "self_correction": {"type": "integer"},
                                        "clarity": {"type": "integer"},
                                        "implementation_quality": {"type": "integer"},

                                    },
                                    "required": [
                                        "problem_understanding",
                                        "strategic_planning",
                                        "self_correction",
                                        "clarity",
                                        "implementation_quality",
                                    ]
                                },
                                "justifications": {
                                    "type": "object",
                                    "properties": {
                                        "problem_understanding": {"type": "string"},
                                        "strategic_planning": {"type": "string"},
                                        "clarity": {"type": "string"},
                                        "implementation_quality": {"type": "string"},
                                        "self_correction": {"type": "string"},
                                    },
                                    "required": [
                                        "problem_understanding",
                                        "strategic_planning",
                                        "self_correction",
                                        "clarity",
                                        "implementation_quality",
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
                return {"eval_result": eval_result, "duration": duration}
            else:
                return {"eval_result": "errpr", "duration": duration}
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
        numerical_iterations = range(1, num_iterations + 1)  # 第2次到第(num_iterations+1)次
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
        max_iterations = max(num_iterations, num_reasoning_evals)  # +1 因為數值分數從第2次開始
        all_iterations = range(1, max_iterations)
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
    def run(self, model_name:str,task_description: str, points: np.array, max_iterations: int = 6, no_improvement_threshold: int=2, max_history_length: int = 3,temp=0.4):
        debate_start_time = time.perf_counter()
        # --- STEP 1: 推理分類與初步設計 ---
        initial_data = "data = " + str(points).replace('\n', '')
        self.iteration_count = 0
        self.logger.info("="*20 + " 開始新的自我優化流程 " + "="*20)
        self.logger.info(f"任務: {task_description.strip()}")
        self.logger.info(f"最大迭代次數: {max_iterations}, 無進步停止閾值: {no_improvement_threshold}")
        self.logger.info(f"使用的模型: {model_name}, 溫度: {temp}")        
        #使用llm進行推理

        response_agent_a=self._run_agent(prompt_template=AGENT_A_PROMPT_TEMPLATE, task_description=task_description, temp=temp, model_name=model_name)
        response_agent_b=self._run_agent(prompt_template=AGENT_B_PROMPT_TEMPLATE, task_description=task_description, temp=temp, model_name=model_name)
        self.plt_dict["Agent name"] = ["Definite Supporter", "Heuristic Supporter"]
        self.plt_dict["initial_confidence"] = [response_agent_a["confidence"], response_agent_b["confidence"]]
        critique_a=self._run_critique(prompt_template=CRITIQUE_PROMPT, agent_self=response_agent_a, agent_opponent=response_agent_b, temp=temp, model_name=model_name)
        critique_b=self._run_critique(prompt_template=CRITIQUE_PROMPT, agent_self=response_agent_b, agent_opponent=response_agent_a, temp=temp, model_name=model_name)
        self.plt_dict["adjusted_confidence"] = [critique_a["confidence"], critique_b["confidence"]]
        self.plt_dict["persuasion"] = [critique_a["persuasion"], critique_b["persuasion"]]
        debate_result = self._rationalize_strategy(prompt_template=ARBITER_PROMPT_TEMPLATE, task_description=task_description, agent_a=response_agent_a, agent_b=response_agent_b, critique_a=critique_a, critique_b=critique_b, temp=temp, model_name=model_name)
        self.plt_dict["final_selection"] = [debate_result["strategy"], debate_result["strategy"]]
        shortened_strategy = self._call_llm_api(debate_result["strategy"]+"\n please shorten the text above into a brief explanation in around 100 words.", model_name=model_name, temp=temp, generate_code=False)
        if model_name.startswith("gemini"):
            shortened_strategy = shortened_strategy.text
        elif model_name.startswith("gpt"):
            shortened_strategy = shortened_strategy.choices[0].message.content or "ERROR: Empty response from GPT."
        self.logger.info(self.plt_dict)
        self.logger.info(type(self.plt_dict))

        df = pd.DataFrame(self.plt_dict)

        # Create a figures
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.2
        x = np.arange(len(df["Agent name"]))

        # Plotting the bars
        ax.bar(x - bar_width, df["initial_confidence"], width=bar_width, label="initial_confidence")
        ax.bar(x, df["adjusted_confidence"], width=bar_width, label="adjusted_confidence")
        ax.bar(x + bar_width, df["persuasion"], width=bar_width, label="Persuade Score")

        # Add text annotations
        for i in range(len(df)):
            ax.text(x[i], df["adjusted_confidence"][i] + 0.02, f'Selected: {df["final_selection"][i]}',
                    ha='center', va='bottom', fontsize=9, color='black')
        fig.text(0.5, 0.01, shortened_strategy, ha='center', fontsize=8, wrap=True)
        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(df["Agent name"])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Agent Debate Scoring Summary")
        ax.legend()
        plot_filename = f"debate_chart_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"進度圖表已儲存至 {plot_filename}")
        
        
        final_strategy = debate_result.get("strategy", "ERROR")
        final_explanation = debate_result.get("explanation", "No explanation provided.")
        
        debate_end_time = time.perf_counter()
        self.history_log.append({"iteration": 1, "type": "Analysis", "strategy": final_strategy, "explanation": final_explanation, "debate_time": debate_end_time - debate_start_time})
        self.reasoning_times.append(debate_end_time- debate_start_time)
        # 根據分類選擇不同的 Prompt
        self.logger.info(f"\nFINAL STRATEGY --- {final_strategy} ---/n")
        if final_strategy == "definite":
            prompt_template = STEP_ITERATIVE_PROMPT_DEFINITE
        # elif final_strategy == "hybrid":
        #     prompt_template = STEP_ITERATIVE_PROMPT_HYBRID
        elif final_strategy == "heuristic":
            prompt_template = STEP_ITERATIVE_PROMPT_HEURISTIC

        # --- STEP 3: 迭代優化循環 ---
        no_improvement_count = 0
        run_start_time = time.perf_counter()
        #開始迭代處理並期待能夠優化
        for i in range(5, max_iterations + 2):
            self.iteration_count += 1
            self.logger.info(f"\n--- Iteration {i} : 開始優化 ---")
            #如果連續N次的輸出數值結果沒有進步則停止優化
            if no_improvement_count >= no_improvement_threshold:
                self.logger.info(f"\n連續 {no_improvement_threshold} 次沒有進步，提前停止迭代。")
                break
            
            last_attempt = self.history_log[-1]
            prompt_step3 = prompt_template.format(
                task_description=task_description,
                final_strategy=final_strategy,
                final_explanation=final_explanation,
                history_log=self.history_log,
                prev_code=last_attempt.get('code', 'this is the first prompt so it is blank. please give the first algorithm code and implement it'),
                prev_result=last_attempt.get('output', 'this is the first prompt so it is blank'),
                prev_score=last_attempt.get('score', None)
            )

            # 組合完整的上下文prompt
            thinking_start_time = time.perf_counter()
            full_prompt_step3 = f"This is iteration {i}. Your task is to improve upon previous results.\n\n{prompt_step3}\n\nRemember to use the same initial data:\n{initial_data}"
            self.logger.info(f"\n--- [Iteration {i} 的完整 Prompt] ---\n{full_prompt_step3}\n--------------------")
            # call llm獲得解答
            if model_name.startswith("gemini"):
                response_step3 = self._call_llm_api(full_prompt_step3, model_name=model_name, temp=temp, generate_code=True)
                parsed_data3 = self._parse_gemini_response(response_step3)
            else:
                response_step3 = self._call_llm_api(full_prompt_step3, model_name=model_name, temp=temp, generate_code=True)
                parsed_data3 = self._parse_gpt_response(response_step3)
            #將輸出的結果進行parsing
            
            # 即使模型沒給文字，reasoning 也會是空字串，而不是 None
            reasoning_step3 = parsed_data3["reasoning"] or "ERROR"
            generated_code = parsed_data3["code"] or "ERROR: No code provided."
            generated_output = parsed_data3["output"] or "ERROR: No output provided."
            generated_score = parsed_data3["score"]  # 可能是 None 或數值
            thinking_end_time = time.perf_counter()
            #進行推理內容的評論
            eval3 = self._evaluate_reasoning(reasoning_step3, model_name=model_name, history_pairs=str(self.history_log))
            r_time_i = thinking_end_time - thinking_start_time
            eval_step3, e_time_i = eval3.get("eval_result", {}), eval3.get("duration", 0)
            self.reasoning_times.append(r_time_i)
            self.evaluation_times.append(e_time_i)
            self.reasoning_evals.append(eval_step3)
            self.logger.info(f"--- [Iteration {i} 的推理結果] ---\n{reasoning_step3}\n--------------------")
            self.logger.info(f"--- [Iteration {i} 的程式碼] ---\n{generated_code}\n--------------------")
            self.logger.info(f"--- [Iteration {i} 的輸出] ---\n{generated_output}\n--------------------")
            
            # 紀錄分數，如果破紀錄則記錄此紀錄，阿連續N次都沒有進步的話也停止優化
            if generated_score is not None and generated_score >0:
                self.scores.append(generated_score)
                self.logger.info(f"Iteration {i} 完成。分數: {generated_score} (歷史最佳: {self.best_score})")
                if generated_score < self.best_score:
                    self.best_score = generated_score
                    self.best_solution_details = f"Iteration {i}: Score={generated_score}, Code:\n{generated_code}\nOutput:\n{generated_output}"
                    no_improvement_count = 0
                    self.logger.info("*** 新的最佳解! ***")
                else:
                    no_improvement_count += 1
                    self.logger.info(f"未找到更優解。連續未進步次數: {no_improvement_count}")

            else:
                self.logger.warning(f"Iteration {i} 警告：未能獲取有效分數。")
                self.scores.append(self.scores[-1] if self.scores else self.best_score) # 重複上一個分數
                no_improvement_count += 1
                self.logger.info(f"計為一次未進步。連續未進步次數: {no_improvement_count}")


            self.history_log.append({"iteration": i-4, "type": "Optimization", "reasoning": reasoning_step3, "code": generated_code, "output": generated_output, "score": generated_score, "eval": eval_step3, "r_time": r_time_i, "e_time": e_time_i})

            if len(self.history_log) > max_history_length:
                self.history_log.pop(1)
            # 如果模型自己覺得OK就OK
            if "FINISHED" in reasoning_step3:
                self.logger.info("\n模型回傳 'FINISHED'，結束優化流程。")
                break

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
            dp_start_time = time.perf_counter()
            dp_calculation = DP4TSP()
            dp_calculation.run(points)
            dp_run_time = time.perf_counter() - dp_start_time
            self.logger.info(f"DP執行時間: {dp_run_time:.2f} 秒")

        self._plot_progress()

# ===================================================================
# 這個區塊現在是作為 "命令列模式" 或 "測試模式" 的進入點
# 當你直接執行 `python main.py` 時，會執行這裡的程式碼
# ===================================================================
if __name__ == '__main__':
    # 為了在命令列看到日誌輸出，設定一個基本的 logger
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    # 取得 SelfOptimizingFramework 的 logger
    logger = logging.getLogger("SelfOptimizingFramework")
    
    print("===================================================")
    logger.info(">>> Running in Command-Line (Standalone) Mode <<<")
    logger.info(">>> Using default parameters from config.py <<<")
    print("===================================================")

    try:
        # --- 參數設定 (非互動式) ---
        # 1. 定義任務和資料 (取代之前的 input)
        TASK_DESCRIPTION = """
        Solve the Traveling Salesman Problem (TSP) for a given set of 2D points.
        The goal is to find the shortest possible tour that visits each point exactly once and returns to the origin point.
        The distance between points is the Euclidean distance.
        """
        # 1. 建立一個預設的隨機數生成器
        rng = np.random.default_rng(42)

        # 2. 使用生成器的 .random() 方法來生成 20x2 的浮點數陣列
        #    注意：形狀 (shape) 是以一個元組 (tuple) (20, 2) 傳入
        data_points = rng.random((25, 2))

        # 2. 設定模型名稱
        model_name = "gemini-2.5-flash" # 設定一個預設模型
        
        # 3. 載入 API Keys (這部分邏輯不變)
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY") # or "OPENAI_API_KEY"
        if not api_key:
            raise ValueError("API Key (e.g., GOOGLE_API_KEY) not found in .env file.")

        logger.info(f"Task: TSP with {len(data_points)} points.")
        logger.info(f"Model: {model_name}")
        # 將 config.py 的設定也印出來
        logger.info(f"Framework Config: {framework_config}")

        # --- 創建並運行框架 ---
        # 假設你的框架初始化需要 API Key
        framework = SelfOptimizingFramework() 
        
        # 使用 **framework_config 將字典中的所有參數展開，作為關鍵字參數傳入 run 方法
        framework.run(
            model_name=model_name,
            task_description=TASK_DESCRIPTION,
            points=data_points,
            **framework_config  # <-- 關鍵！自動應用 config.py 的設定
        )

        logger.info(">>> Command-Line run finished successfully. <<<")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(f"Error details:\n{traceback.format_exc()}")
