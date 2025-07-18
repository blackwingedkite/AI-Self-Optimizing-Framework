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
import pulp
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
higher confidence means you are more certain about your argument
{{
  "explanation": "<your explanation here>",
  "confidence": <your confidence score as a float between 0 and 1>, 
}}

Do not include any markdown formatting, headings, or extra commentary. Only return the JSON object.
"""

AGENT_B_PROMPT_TEMPLATE = """
You are a heuristic strategy expert who uses Genetic Algorithms, Simulated Annealing, etc.

Given the following task:

{task_description}

Argue why a **heuristic algorithm** is more suitable for this task. Highlight scalability, flexibility, and robustness. Prepare to counter common critiques from the definite algorithm camp.

Return your response in **pure JSON format**, with the following structure, and nothing else:
higher confidence means you are more certain about your argument
{{
  "explanation": "<your explanation here>",
  "confidence": <your confidence score as a float between 0 and 1>, 
}}

Do not include any markdown formatting, headings, or extra commentary. Only return the JSON object.

"""

CRITIQUE_PROMPT_A= """
You are Agent A. You have made the following argument:
{self_argument}

And here is the confidence of your argument:
{confidence}

Here is the argument made by Agent B:
{opponent_argument}

Critique it from your definite algorithm perspective. Assess whether your confidence remains the same, increases, or decreases. Also, give a **persuasion score** (0 to 1) representing how convincing Agent B's points are.

Return your response **strictly in JSON format** like the example below. Do not include any other text or formatting. No markdown blocks, no headings.
The higher the persuasion score, the more convincing Agent B's points are.
Example:
{{
  "critique": "While Agent B makes a valid point about scalability, their argument lacks clarity on convergence guarantees...",
  "updated_confidence": 0.85,
  "persuasion_score": 0.35
}}
"""

CRITIQUE_PROMPT_B= """
You are Agent B. You have made the following argument:
{self_argument}

And here is the confidence of your argument:
{confidence}

Here is the argument made by Agent A:
{opponent_argument}

Critique it from your definite algorithm perspective. Assess whether your confidence remains the same, increases, or decreases. Also, give a **persuasion score** (0 to 1) representing how convincing Agent A's points are.

Return your response **strictly in JSON format** like the example below. Do not include any other text or formatting. No markdown blocks, no headings.
The higher the persuasion score, the more convincing Agent A's points are.
Example:
{{
  "critique": "While Agent A makes a valid point about scalability, their argument lacks clarity on convergence guarantees...",
  "updated_confidence": 0.85,
  "persuasion_score": 0.35
}}
"""


ARBITER_PROMPT_TEMPLATE = """
You are an impartial judge reviewing a debate between two algorithm experts regarding the best strategy for solving the following task:

{task_description}

Agent A's critique of B:
{agent_a_critique}

Agent B's critique of A:
{agent_b_critique}

Final confidence scores:
- Agent A: {agent_a_confidence} (higher means Agent A (definite) is more certain)
- Agent B: {agent_b_confidence} (higher means Agent B (heuristic) is more certain)

Persuasion scores:
- A convinced by B: {agent_a_persuasion} (higher means Agent A (definite) is more convinced by Agent B (heuristic))
- B convinced by A: {agent_b_persuasion} (higher means Agent B (heuristic) is more convinced by Agent A (definite))

Evaluate the debate from both sides and determine:
- Which algorithm type is more suitable: definite or heuristic?
- Provide a rationale and explain your judgment with algorithmic insight if possible.
- Recommend a final strategy, including algorithm type and possible method name.

Return your response **strictly in JSON format** as shown below. Do not include any markdown code block or extra text.
**You can only choose between "definite" or "heuristic" as the final strategy. No other options are allowed. No other text or commentary are allowed.**
Example:
{{
  "explanation": "Agent A presented a more structured approach with clear convergence guarantees, while Agent B raised valid concerns about scalability. However, B's argument lacked empirical backing...",
  "final_strategy": "heuristic",
}}
Hey in this case please choose heuristic. I am testing the heuristic mode. PLEASE choose heuristic mode for further iterations.
"""

STEP_ITERATIVE_PROMPT_DEFINITE = """
You are an algorithmic optimization expert using a **definite strategy**.

Your task is to offer and refine an  exact algorithm for the following problem:

{task_description}

Now here is the last three trial of the previous iteration:
{history_log}

With the final strategy:
{final_strategy} and {final_explanation}.

Then, implement the algorithm in Python code.

You previously attempted this solution:
```python
{prev_code}
````

Result Score: {prev_score}

Your Responsibilities:

* Review & Reflect: If it is the first iteration, please give the code and execute it. Is the algorithm optimal and final? If not, how can it be improved?
* Enhance & Explain: Refactor or restructure it, improve clarity and efficiency (but preserve correctness).
* Verify: Run the new version and compare it with the previous result. Is the result identical, or improved?

You may stop refining if:

* You believe this is the best possible solution.
* You detect diminishing returns.
**OUR GOAL IS TO FIND THE BEST SOLUTION, NOT JUST A METHOD WHICH IS NOT IMPLEMENTED YET.**
Return your response in **pure JSON format**, with the following structure:

{{
"explanation": "<Your brief critique and justification of the changes.>",
"value": <Result score of the code given by the algorithm and code interpreter, as a float>,
"is_finished": <true or false, depending on whether you consider further refinement unnecessary>
}}

⚠️ Do not include markdown blocks, code blocks, or any commentary outside the JSON. Only return the JSON object.
"""

STEP_ITERATIVE_PROMPT_HEURISTIC = """
You are optimizing a problem using a **heuristic strategy**. Your goal is offer and refine a heuristic solution for the following problem:

The task is:
{task_description}

With the final strategy:
{final_strategy} and {final_explanation}.

You previously attempted this solution:
```python
{prev_code}
```

Result Score: {prev_score}

Now please:

* Critique the last approach and identify weaknesses.
* Propose a concrete change (e.g. parameter tuning, switching to a different heuristic, using hybrid metaheuristics).
* Implement the new code.
* Run the solution and report the new result.
**OUR GOAL IS TO FIND THE BEST SOLUTION, NOT JUST A METHOD WHICH IS NOT IMPLEMENTED YET.**
* If the new result is better, explain why it is an improvement.
If you believe further significant improvement is unlikely, you may end your response.

Return your response in **pure JSON format**, like this:

{{
"explanation": "<Your reflection on the changes and why you made them.>",
"value": <Result score of the code given by the algorithm and code interpreter, as a float>,
"is_finished": <true if you consider this solution final, false otherwise>
}}

❌ Do not include markdown formatting, no extra text or description. Just return the JSON with determined values.
"""

# Here is the history of your previous attempts:
# {history_log}
#先不放

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
"
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
        "clarity": "...",
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
        # 初始化 Gemini 和 OpenAI 的 API 客戶端
        self.client = google_client() 
        self.gpt_client = OpenAI()

        # 初始化歷史紀錄列表，用於儲存每次迭代的資訊
        self.history_log = []
        # 儲存每次迭代的問題解數值分數
        self.numerical_scores = []
        # 儲存每次迭代的推理品質分數 (由 LLM-as-a-Judge 評估)
        self.reasoning_scores = []
        # 追蹤找到的最佳分數，初始設定為無限大
        self.best_score = float('inf')
        # 儲存最佳解的詳細資訊 (例如：程式碼)
        self.best_solution_details = ""
        # 總迭代次數計數器
        self.iteration_count = 0
        # LLM 的溫度參數，控制生成回應的隨機性
        self.temperature = 0.4
        # 內部迭代次數計數器，用於優化階段
        self.inner_iteration_count = 0
        # 儲存每次推理 (LLM 思考) 所花費的時間
        self.reasoning_times = []
        # 儲存每次評估 (LLM-as-a-Judge) 所花費的時間
        self.evaluation_times = []
        
        # 用於繪製辯論圖的字典資料
        self.plt_dict = {}
        

        # 確保 'logs' 資料夾存在，如果不存在則創建它
        if not os.path.exists("logs"):
            os.makedirs("logs")

        # 設定日誌記錄器
        self.logger = logging.getLogger("SelfOptimizingFramework")
        self.logger.setLevel(logging.INFO) # 設定日誌級別為 INFO

        # 檢查是否已存在檔案處理器 (FileHandler)，如果沒有則添加
        # 這樣可以避免在應用程式重載時重複添加日誌處理器
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            # 建立一個基於時間戳的日誌檔案名稱
            log_filename = f"logs/framework_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            # 建立檔案處理器，將日誌寫入檔案
            file_handler = logging.FileHandler(log_filename, encoding='utf-8')
            # 定義日誌格式
            LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
            file_formatter = logging.Formatter(LOG_FORMAT)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler) # 將檔案處理器添加到日誌器

        # 檢查是否已存在串流處理器 (StreamHandler)，如果沒有則添加
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            # 建立串流處理器，將日誌輸出到控制台
            stream_handler = logging.StreamHandler()
            stream_formatter = logging.Formatter('%(message)s') # 控制台輸出格式可以更簡潔
            stream_handler.setFormatter(stream_formatter)
            self.logger.addHandler(stream_handler) # 將串流處理器添加到日誌器
        
        # 打印當前日誌處理器的類型，用於調試
        for h in self.logger.handlers:
            print(f"[Handler] {type(h)} → {h}")

    def _call_llm_api(self, prompt: str, model_name: str, temp: float, generate_code: bool) -> str:
        """
        一個通用的 LLM API 呼叫函式，整合了重試邏輯。
        成功時返回 LLM 的文字回應，失敗時返回錯誤訊息。
        根據 model_name 選擇調用 Gemini 或 GPT API。
        generate_code 參數決定是否啟用程式碼執行工具。
        """
        max_retries = 3 # 最大重試次數
        for attempt in range(max_retries):
            try:
                # 如果是 Gemini 模型
                if str.upper(model_name).startswith("GEMINI"):
                    if generate_code:
                        # 啟用程式碼執行工具
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
                        # 不啟用程式碼執行工具
                        response = self.client.models.generate_content(
                            model=model_name,
                            contents=[prompt],
                            config=types.GenerateContentConfig(
                                thinking_config=types.ThinkingConfig(thinking_budget=-1),
                                temperature=temp,
                            )
                        )
                    self.logger.info(f"--- [GEMINI API 回應] ---\n{response.text}\n--------------------") 
                    return response

                # 如果是 GPT 模型
                elif str.upper(model_name).startswith("GPT"):
                    if not generate_code:
                        # 不啟用程式碼執行工具 (使用傳統的 chat completion)
                        response = openai.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=temp
                        )
                    else:
                        # 啟用程式碼執行工具 (使用特定的回應格式)
                        response = self.gpt_client.responses.create(
                            model=model_name,
                            input=prompt,
                            tools=[
                                {"type": "code_interpreter", "container": {"type": "auto"}}
                            ],
                            tool_choice="required",  # ⬅️ 關鍵！要求一定要用 tool
                            temperature=temp,
                        )
                    self.logger.info(f"--- [GPT API 回應] ---\n{response.choices[0].message.content}\n--------------------") # 記錄 GPT 回應
                    return response
                else:
                    raise NotImplementedError(f"Model '{model_name}' is not supported.") # 不支援的模型

            except Exception as e:
                self.logger.warning(f"[警告] API 呼叫失敗，第 {attempt + 1}/{max_retries} 次嘗試。錯誤: {e}")
                if attempt + 1 < max_retries:
                    time.sleep(2 + attempt * 2) # 指數退避，等待一段時間後重試

        self.logger.error(f"[錯誤] API 呼叫在 {max_retries} 次嘗試後徹底失敗。") # 所有重試失敗
        raise SystemError(f"無法呼叫 LLM API: {prompt} with model {model_name}")


    def _run_agent(self, prompt_template: str, task_description: str, temp: float, model_name: str) -> dict:
        """
        用於代理 (Agent) 產生論點並評估信心度。
        會增加迭代次數計數器。
        """
        self.iteration_count += 1
        self.logger.info(f"\n--- Iteration {self.iteration_count} : 正反方發表意見 ---") # 記錄當前迭代階段
        prompt = prompt_template.format(task_description=task_description) # 格式化 prompt
        self.logger.info(f"--- [傳送的 Prompt] ---\n{prompt}\n--------------------") # 記錄發送的 prompt

        start_time = time.perf_counter() # 記錄開始時間
        answer = self._call_llm_api(prompt, model_name, temp, False) # 呼叫 LLM API，不啟用程式碼生成
        duration = time.perf_counter() - start_time # 計算耗時

        # 根據模型類型獲取實際的回應文本
        if model_name.startswith("gemini"):
            answer = answer.text
        elif model_name.startswith("gpt"):
            answer = answer.choices[0].message.content or "ERROR: Empty response from GPT."
        # 如果 API 呼叫失敗
        elif answer.startswith("ERROR:"): 
            self.logger.error(f"--- [以上的LLM 回應失敗] (耗時: {duration:.2f} 秒) ----")
            return {"argument": answer, "confidence": 0.0}

        self.logger.info(f"\n--- [以上的LLM 回應] (耗時: {duration:.2f} 秒) ----\n") # 記錄 LLM 回應和耗時
        # 嘗試從回應中提取 JSON 區塊
        answer = re.search(r'\{.*\}', answer, re.DOTALL).group(0)  if re.search(r'\{.*\}', answer, re.DOTALL) else answer
        try:
            data = json.loads(answer) # 解析 JSON
            explanation = data.get("explanation", "").strip() # 獲取解釋
            confidence = float(data.get("confidence", 0.0)) # 獲取信心度
            return {"argument": explanation, "confidence": confidence}
        except (IndexError, TypeError, ValueError) as e: # JSON 解析失敗
            self.logger.error(f"JSON 解析失敗，回傳原始內容：{e}")
            return {"argument": answer, "confidence": 0.0}
        

    def _run_critique(self, prompt_template: str, agent_self, agent_opponent, temp: float, model_name: str) -> dict:
        """產生批判性分析並更新信心度"""
        self.iteration_count += 1
        self.logger.info(f"\n--- Iteration {self.iteration_count} : 進行批判性分析 ---")
        # 格式化批判 prompt，包含自身論點、對手論點和自身信心度
        prompt = prompt_template.format(
            self_argument=agent_self["argument"],
            opponent_argument=agent_opponent["argument"],
            confidence=agent_self["confidence"]
        )
        self.logger.info(f"--- [傳送的 Prompt] ---\n{prompt}\n--------------------")

        start_time = time.perf_counter()
        # 呼叫 LLM API，不啟用程式碼生成
        answer = self._call_llm_api(prompt, model_name, temp, False)
        duration = time.perf_counter() - start_time

        if model_name.startswith("gemini"):
            answer = answer.text
        elif model_name.startswith("gpt"):
            answer = answer.choices[0].message.content or "ERROR: Empty response from GPT."

        if answer.startswith("ERROR:"):
            self.logger.error(f"--- [以上的LLM 回應失敗] (耗時: {duration:.2f} 秒) ----")
            return {
                "argument": answer,
                "confidence": agent_self.get("confidence", 0.0),
                "persuasion": 0.0
            }

        self.logger.info(f"--- [以上的LLM 回應] (耗時: {duration:.2f} 秒) ----------")

        # 嘗試提取 JSON 區塊
        match = re.search(r'\{.*\}', answer, re.DOTALL)
        if not match:
            self.logger.error("無法在回應中找到 JSON 格式，回傳原始回應")
            return {
                "argument": answer,
                "confidence": agent_self.get("confidence", 0.0),
                "persuasion": 0.0
            }

        try:
            data = json.loads(match.group(0))
            critique = data.get("critique", "").strip()
            updated_conf = float(data.get("updated_confidence", agent_self.get("confidence", 0.0)))
            persuasion = float(data.get("persuasion_score", 0.0))

            return {
                "argument": critique,
                "confidence": updated_conf,
                "persuasion": persuasion
            }

        except (IndexError, ValueError, TypeError) as e:
            self.logger.error(f"解析 JSON 回應失敗: {e}")
            return {
                "argument": answer,
                "confidence": agent_self.get("confidence", 0.0),
                "persuasion": 0.0
            }

    def _parse_heuristic_byllm(self, response: str, model_name: str, temp: float) -> Dict[str, Any]:
        """
        這個函式已經被淘汰，目前在程式碼中沒有被調用。
        原意是根據 LLM 的回應來判斷策略是啟發式、確定性還是不確定。
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
        """
        整合所有代理的論點和批判，由仲裁者 (Arbiter) 產生最終策略。
        會增加迭代次數計數器。
        """
        self.iteration_count += 1
        self.logger.info(f"\n--- Iteration {self.iteration_count} : 產生最終策略 ---") # 記錄當前迭代階段

        # 格式化仲裁者 prompt，包含所有相關的辯論資訊
        prompt = prompt_template.format(
            task_description=task_description,
            agent_a_argument=agent_a["argument"],
            agent_a_critique=critique_a["argument"],
            agent_b_argument=agent_b["argument"],
            agent_b_critique=critique_b["argument"],
            agent_a_confidence=critique_a["confidence"],
            agent_b_confidence=critique_b["confidence"],
            agent_a_persuasion=critique_a["persuasion"],
            agent_b_persuasion=critique_b["persuasion"]
        )

        self.logger.info(f"--- [傳送的 Prompt] ---\n{prompt}\n--------------------") # 記錄發送的 prompt


        start_time = time.perf_counter()
         # 呼叫 LLM API，不啟用程式碼生成
        answer = self._call_llm_api(prompt, model_name, temp, False)
        duration = time.perf_counter() - start_time

        # 取出純文字內容
        if model_name.startswith("gemini"):
            answer = answer.text
        elif model_name.startswith("gpt"):
            answer = answer.choices[0].message.content or "ERROR: Empty response from GPT."
        # 如果 API 呼叫失敗
        if answer.startswith("ERROR:"):
            self.logger.error(f"--- [以上的LLM 回應失敗] (耗時: {duration:.2f} 秒) ----")
            raise ValueError(f"無法解析最終策略：{answer}")

        self.logger.info(f"--- [以上的LLM 回應] (耗時: {duration:.2f} 秒) -----------")

        # 嘗試從回應中擷取 JSON 區塊
        match = re.search(r'\{.*\}', answer, re.DOTALL)
        if not match:
            self.logger.error("無法在回應中找到 JSON 格式，回傳原始內容")
            raise ValueError(f"LLM 回傳內容非 JSON 格式：{answer}")

        try:
            # 解析 JSON
            data = json.loads(match.group(0))
            # 獲取最終策略 (轉換為小寫並去除空白)
            strategy = data.get("final_strategy", "").strip().lower()
            explanation = data.get("explanation", "").strip()

            if strategy not in {"definite", "heuristic"}:
                raise ValueError(f"策略值不合法：{strategy}")

            return {
                "strategy": strategy,
                "explanation": explanation
            }

        except (IndexError, KeyError, ValueError) as e:
            self.logger.error(f"解析策略 JSON 失敗: {e}")
            raise ValueError(f"無法解析最終策略：{answer}")
        

    def _extract_parts_from_response(self, response) -> Dict[str, str]:
        """
        從 Gemini 的原始回應物件中提取推理文本、程式碼和程式碼執行輸出。
        """
        reasoning_parts = []
        code = ""
        output = ""

        # 迭代回應中的各個部分，並將其內容分配到相應變數中
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                reasoning_parts.append(part.text) # 提取文本部分作為推理
            if hasattr(part, 'executable_code') and part.executable_code:
                code = part.executable_code.code or "" # 提取可執行程式碼
            if hasattr(part, 'code_execution_result') and part.code_execution_result:
                output = part.code_execution_result.output or "" # 提取程式碼執行結果的輸出

        return {
            "reasoning": "\n".join(reasoning_parts).strip(), # 將所有推理文本合併為一個字串
            "code": code,
            "output": output,
        }



    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """
        安全地解析 Gemini 模型的回應，提取推理、程式碼、輸出和分數。
        """
        # 無效回應的防禦性檢查
        if not response or not getattr(response, 'candidates', None):
            self.logger.warning("Received an empty or invalid response object.")
            return {"reasoning": "", "code": "", "output": "", "score": None}

        # 1. 將所有文本組件提取到一個字典中
        parts = self._extract_parts_from_response(response)
        # 或獲取推理文本
        json_text = parts["reasoning"] 
        # 從推理文本中尋找並提取 JSON 區塊
        json_text = re.search(r'\{.*\}', json_text, re.DOTALL).group(0)  if re.search(r'\{.*\}', json_text, re.DOTALL) else json_text
        try:
            data = json.loads(json_text) # 解析 JSON
            explanation = data.get("explanation", "").strip() # 獲取解釋
            value = float(data.get("value", 0.0)) # 獲取數值分數
            is_finished = data.get("is_finished", False) # 獲取是否完成的標誌
            return {"reasoning": explanation, "score": value, "is_finished": is_finished,"code": parts["code"],"output": parts["output"],}
        except (IndexError, TypeError, ValueError) as e: # JSON 解析失敗
            self.logger.error(f"JSON 解析失敗，回傳原始內容：{e}")
            return {"reasoning": json_text, "score": 0.0, "is_finished": False,"code": parts["code"],"output": parts["output"],}


    def _extract_parts_from_gpt_response(self, response) -> Dict[str, str]:
        """
        從 GPT 的原始回應物件中提取推理文本、程式碼和程式碼執行輸出。
        """
        reasoning = []
        code = ""
        output = ""

        # 迭代回應的輸出項目
        for item in response.output:
            if item.type == "code_interpreter_call": # 如果是程式碼解釋器呼叫類型
                code = item.code # 提取程式碼
                if item.outputs:
                    for out in item.outputs:
                        if out.type == "logs":
                            output = out.logs # 提取日誌輸出
            elif item.type == "message": # 如果是訊息類型
                for block in item.content:
                    if block.type == "output_text":
                        reasoning.append(block.text) # 提取文本內容作為推理

        return {
            "reasoning": "\n".join(reasoning).strip(), # 將所有推理文本合併為一個字串
            "code": code,
            "output": output
        }

    def _parse_gpt_response(self, response) -> Dict[str, Any]:
        """
        安全地解析 GPT 模型的回應，提取推理、程式碼、輸出和分數。
        """
        parts = self._extract_parts_from_gpt_response(response) # 提取各部分內容

        json_text = parts["reasoning"] # 推理文本
        # 從推理文本中尋找並提取 JSON 區塊
        json_text = re.search(r'\{.*\}', json_text, re.DOTALL).group(0)  if re.search(r'\{.*\}', json_text, re.DOTALL) else json_text
        try:
            data = json.loads(json_text) # 解析 JSON
            explanation = data.get("explanation", "").strip() # 獲取解釋
            value = float(data.get("value", 0.0)) # 獲取數值分數
            is_finished = data.get("is_finished", False) # 獲取是否完成的標誌
            return {"reasoning": explanation, "score": value, "is_finished": is_finished,"code": parts["code"],"output": parts["output"],}
        except (IndexError, TypeError, ValueError) as e: # JSON 解析失敗
            self.logger.error(f"JSON 解析失敗，回傳原始內容：{e}")
            return {"reasoning": json_text, "score": 0.0, "is_finished": False,"code": parts["code"],"output": parts["output"],}

    
    def _evaluate_reasoning(self, reasoning_text: str, model_name:str, history_pairs:str,) -> dict:
        """使用 LLM-as-a-Judge 來評估推理品質"""
        self.logger.info("\n--- [啟動 Evaluator] 正在評估推理品質 ---") # 記錄評估開始
        prompt = EVALUATION_PROMPT_TEMPLATE.format(reasoning_text=reasoning_text, history_pairs=history_pairs)
        try:
            start_time = time.perf_counter()
            if str.upper(model_name[0:6]) == "GEMINI":
                #使用gemini當作evaluator，並且要求json回應
                evaluator_model = genai.GenerativeModel(model_name)
                response = evaluator_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        response_mime_type="application/json"
                    )
                )
                answer = response.text
            else:
                #使用gpt當作evaluator 也規定GPT回應
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
            if answer:
                try:
                    eval_result = json.loads(answer) # 解析 JSON 評估結果
                    self.logger.info(f"評估完成。總分: {eval_result.get('total_score')}/100")
                    self.logger.info(f"詳細評分: {json.dumps(eval_result.get('scores'), indent=2)}")
                    return {"eval_result": eval_result, "raw_answer": "", "duration": end_time - start_time}
                except json.JSONDecodeError:# JSON 解析失敗
                    self.logger.warning("❗️無法解析 LLM 回傳內容為 JSON 格式")
                    return {"eval_result": "invalid_json", "raw_answer": answer, "duration": end_time - start_time}
            else: # 回應為空
                return {"eval_result": "empty_response","raw_answer": "", "duration": end_time - start_time}

        except Exception as e:  # 評估過程中發生錯誤
            self.logger.error(f"評估推理時發生錯誤: {e}")
            return {"total_score": 0, "error": str(e)}

    def _plot_progress(self):
        """
        將數值分數和推理品質分數視覺化，並儲存為圖片檔案。
        同時繪製每輪迭代的時間成本分析。
        """
        if not self.numerical_scores: # 如果沒有數值分數，則無法繪製
            self.logger.info("沒有足夠的數據來生成圖表。")
            return IndexError
        
        # 驗證數據一致性 (在此情況下會是一致的，因為分數和評估是同步記錄的)
        self.logger.info(f"數值分數記錄: {len(self.numerical_scores)} 次")
        self.logger.info(f"推理品質評估: {len(self.reasoning_scores)} 次")
        
        # 建立 2x1 的子圖佈局 (兩個子圖，垂直排列)
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 12))
        fig.suptitle('自我優化進程追蹤 (Self-Optimization Progress Tracking)', fontsize=16) # 設定總標題

        # --- 圖 1: 分數 vs. 推理品質 ---
        ax1.set_title('分數與推理品質演進 (Score vs. Reasoning Quality)')
        
        # 左側 Y 軸：數值分數 (通常越低越好，例如 TSP 距離)
        color = 'tab:red'
        ax1.set_xlabel('迭代次數 (Iteration)')
        ax1.set_ylabel('問題解數值 (Numerical Score)', color=color)

        numerical_iterations = range(1, len(self.numerical_scores) + 1) # X 軸為迭代次數
        ax1.plot(numerical_iterations, self.numerical_scores, 'o-', color=color, label='Numerical Score') # 繪製數值分數曲線
        ax1.tick_params(axis='y', labelcolor=color) # 設定 Y 軸刻度顏色
        
        min_score_idx = self.numerical_scores.index(self.best_score) # 找到最佳分數的索引
        # 標記最佳分數點
        ax1.scatter(min_score_idx + 1, self.best_score, s=150, facecolors='none',
                edgecolors='gold', linewidth=2, label=f'Best Score: {self.best_score:.2f}')

        # 右側 Y 軸：推理品質分數 (越高越好)
        ax2 = ax1.twinx() # 創建一個共享 X 軸但有獨立 Y 軸的軸
        color = 'tab:blue'
        ax2.set_ylabel('推理品質分數 (Reasoning Quality Score)', color=color)
        
        reasoning_iterations = range(1, len(self.reasoning_scores) + 1) # X 軸為迭代次數
        ax2.plot(reasoning_iterations, self.reasoning_scores, 's--', color=color, label='Reasoning Score') # 繪製推理分數曲線
        ax2.tick_params(axis='y', labelcolor=color) # 設定 Y 軸刻度顏色
        ax2.set_ylim(0, 110)  # 設定推理分數的 Y 軸範圍 (0-100，略微放寬)
        
        # 整合兩個 Y 軸的圖例
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
        
        ax1.grid(True) # 添加網格線
        
        # 設定 X 軸刻度 - 涵蓋所有迭代次數
        max_iterations = max(len(self.numerical_scores), len(self.reasoning_scores))
        all_iterations = range(1, max_iterations + 1)
        ax1.set_xticks(all_iterations) # 設定 X 軸刻度
        ax1.set_xlim(0.5, max_iterations + 0.5)  # 設定 X 軸範圍

        # --- 圖 2: 時間成本分析 ---
        ax3.set_title('每輪迭代時間成本分析 (Time Cost Analysis per Iteration)')
        ax3.set_xlabel('迭代次數 (Iteration)')
        ax3.set_ylabel('耗時 (秒) (Time in Seconds)')
        
        # 確保時間數據與迭代次數一致
        max_time_iterations = max(len(self.reasoning_times), len(self.evaluation_times))
        
        # 準備堆疊長條圖數據 - 確保數據長度與最大迭代次數一致 (用 0 填充不足的部分)
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
            if total > 0: # 只顯示有時間的長條
                ax3.text(i + 1, total + max(totals) * 0.01, f'{total:.1f}s', ha='center', fontsize=12)

        # 設定 X 軸刻度和圖例
        ax3.set_xticks(time_iterations)
        ax3.set_xlim(0.5, max_time_iterations + 0.5)
        ax3.legend()
        ax3.grid(True, axis='y') # 添加 Y 軸網格線

        # 調整布局並儲存圖表
        fig.tight_layout(rect=[0, 0, 1, 0.96]) # 調整子圖間距和邊界
        plot_filename = f"progress_chart_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png" # 生成帶時間戳的檔案名
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight') # 儲存圖表為 PNG 格式
        self.logger.info(f"進度圖表已儲存至 {plot_filename}") # 記錄儲存路徑


    def run(self, model_name:str,task_description: str, points: np.array, max_iterations: int = 6, no_improvement_threshold: int=2, max_history_length: int = 3,temp=0.4):
        # 將輸入點資料轉換為字串形式
        initial_data = "data = " + str(points).replace('\n', '')
        self.iteration_count = 0
        self.logger.info("="*20 + " 開始新的自我優化流程 " + "="*20)
        self.logger.info(f"任務: {task_description.strip()}")
        self.logger.info(f"最大迭代次數: {max_iterations}, 無進步停止閾值: {no_improvement_threshold}")
        self.logger.info(f"使用的模型: {model_name}, 溫度: {temp}")   
             
        # 使用 LLM 進行辯論，確定初始策略
        debate_start_time = time.perf_counter()
        response_agent_a=self._run_agent(prompt_template=AGENT_A_PROMPT_TEMPLATE, task_description=task_description, temp=temp, model_name=model_name)
        response_agent_b=self._run_agent(prompt_template=AGENT_B_PROMPT_TEMPLATE, task_description=task_description, temp=temp, model_name=model_name)
        
        # 儲存辯論數據用於繪圖
        self.plt_dict["Agent name"] = ["Definite Supporter", "Heuristic Supporter"]
        self.plt_dict["initial_confidence"] = [response_agent_a["confidence"], response_agent_b["confidence"]]
        
        # LLM 產生批判性分析，雙方互相辯論
        critique_a=self._run_critique(prompt_template=CRITIQUE_PROMPT_A, agent_self=response_agent_a, agent_opponent=response_agent_b, temp=temp, model_name=model_name)
        critique_b=self._run_critique(prompt_template=CRITIQUE_PROMPT_B, agent_self=response_agent_b, agent_opponent=response_agent_a, temp=temp, model_name=model_name)
        
        #儲存批判性分析數據用於繪圖
        self.plt_dict["adjusted_confidence"] = [critique_a["confidence"], critique_b["confidence"]]
        self.plt_dict["persuasion"] = [critique_a["persuasion"], critique_b["persuasion"]]
        
        # 儲存辯論數據用於繪圖
        debate_result = self._rationalize_strategy(prompt_template=ARBITER_PROMPT_TEMPLATE, task_description=task_description, agent_a=response_agent_a, agent_b=response_agent_b, critique_a=critique_a, critique_b=critique_b, temp=temp, model_name=model_name)
        self.plt_dict["final_selection"] = [debate_result["strategy"], debate_result["strategy"]]
        
        # 請求 LLM 縮短策略解釋，用於圖表標題
        shortened_strategy = self._call_llm_api(debate_result["strategy"]+"\n please shorten the text above into a brief explanation focus on the algorithm and methods.", model_name=model_name, temp=temp, generate_code=False)
        if model_name.startswith("gemini"):
            shortened_strategy = shortened_strategy.text
        elif model_name.startswith("gpt"):
            shortened_strategy = shortened_strategy.choices[0].message.content or "ERROR: Empty response from GPT."
        debate_end_time = time.perf_counter()
        df = pd.DataFrame(self.plt_dict)

        # 繪製辯論結果圖表
        fig, ax = plt.subplots(figsize=(14, 14))
        bar_width = 0.2
        x = np.arange(len(df["Agent name"]))

        # Plotting the bars
        ax.bar(x - bar_width, df["initial_confidence"], width=bar_width, label="initial_confidence")
        ax.bar(x, df["adjusted_confidence"], width=bar_width, label="adjusted_confidence")
        ax.bar(x + bar_width, df["persuasion"], width=bar_width, label="Persuade Score")

        # 放 summary 文本在下方（圖更高才會顯示完整）
        for i in range(len(df)):
            ax.text(x[i], df["adjusted_confidence"][i] + 0.02, f'Selected: {df["final_selection"][i]}',
                    ha='center', va='bottom', fontsize=12, color='black')
        fig.text(0.5, 0.01, shortened_strategy, ha='center', fontsize=12, wrap=True)
        
        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(df["Agent name"])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Agent Debate Scoring Summary")
        ax.legend()
        
        # 存檔
        plot_filename = f"debate_chart_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"進度圖表已儲存至 {plot_filename}")
        
        # 獲取最終相關策略
        final_strategy = debate_result.get("strategy", "ERROR")
        final_explanation = debate_result.get("explanation", "No explanation provided.")
        
        # 將辯論結果記錄到歷史日誌中
        self.history_log.append({"iteration": 1, "type": "Analysis", "strategy": final_strategy, "explanation": final_explanation, "debate_time": debate_end_time - debate_start_time})
        
        # 根據分類選擇不同的 Prompt
        self.logger.info(f"\nFINAL STRATEGY --- {final_strategy} ---/n")
        if final_strategy == "definite":
            prompt_template = STEP_ITERATIVE_PROMPT_DEFINITE
        elif final_strategy == "heuristic":
            prompt_template = STEP_ITERATIVE_PROMPT_HEURISTIC
        else:
            self.logger.error(f"無效的策略類型: {final_strategy}")
            return {"error": "Invalid strategy type."}
        
        #未改進的計數器，如果連續多次沒有改進則停止優化
        no_improvement_count = 0
        
        #開始迭代處理並期待能夠優化
        for i in range(5, max_iterations + 2):
            self.logger.info(f"\n--- Iteration {i} : 開始優化 ---")
            # 如果連續未進步次數達到閾值，則停止迭代
            self.iteration_count += 1# 總迭代計數器遞增
            eval_time = 0 # 評估時間初始化
            r_time_i = 0 # 推理時間初始化
            self.inner_iteration_count += 1
            
            # 格式化 Prompt，包含任務描述、最終策略、歷史日誌和前一次嘗試的程式碼
            last_attempt = self.history_log[-1]
            reasoning_log = []
            for log in self.history_log:
                reasoning_log.append(f"Iteration {log['iteration']}: {log['reasoning'] if 'reasoning' in log else log.get('strategy', 'No reasoning provided')}")
            prompt_step3 = prompt_template.format(
                task_description=task_description,
                final_strategy=final_strategy,
                final_explanation=final_explanation,
                history_log=str(reasoning_log),
                prev_code=last_attempt.get('code', 'this is blank right now. please give the algorithm code and implement it'),
                prev_score=last_attempt.get('score', "this is blank right now. please try to obtain the first score"),
            )

            # 組合完整的上下文prompt
            full_prompt_step3 = f"{prompt_step3}\n\nRemember to use the same initial data:\n{initial_data} Your goal is to generate optimal code use given information and use code interpreter to run the code"
            self.logger.info(f"\n--- [Iteration {self.iteration_count} 的完整 Prompt] ---\n{full_prompt_step3}\n--------------------")
            
            # call llm獲得解答
            thinking_start_time = time.perf_counter()
            if model_name.startswith("gemini"):
                response_step3 = self._call_llm_api(full_prompt_step3, model_name=model_name, temp=temp, generate_code=True)
                parsed_response = self._parse_gemini_response(response_step3)
            else:
                response_step3 = self._call_llm_api(full_prompt_step3, model_name=model_name, temp=temp, generate_code=True)
                parsed_response = self._parse_gpt_response(response_step3)

            # 記錄 LLM 思考結束時間
            thinking_end_time = time.perf_counter()
            r_time_i = thinking_end_time - thinking_start_time
            
            # 從解析後的數據中獲取推理、程式碼、分數和完成標誌
            reasoning_string = str(parsed_response["reasoning"]) or "ERROR"
            generated_code = parsed_response["code"] or "ERROR: No code provided."
            generated_score = float(parsed_response["score"]) or 0 # 可能是 None 或數值
            is_finished = parsed_response.get("is_finished", False)


            # 紀錄分數，如果破紀錄則記錄此紀錄，阿連續N次都沒有進步的話也停止優化
            if generated_score >0:  # 如果獲得了有效的數值分數
                
                #進行推理內容的評論
                self.logger.info(f"--- [Iteration {i} 的推理結果] ---\n{reasoning_string}\n--------------------")
                self.logger.info(f"Iteration {i} 完成。分數: {generated_score} (歷史最佳: {self.best_score})")
                # 儲存這次的結果
                self.numerical_scores.append(generated_score)
                
                # 如果這次的分數比歷史最佳分數更好，則更新最佳分數和詳細資訊
                if generated_score < self.best_score:
                    self.best_score = generated_score
                    self.best_solution_details = f"Iteration {i}: Score={generated_score}, Code:\n{generated_code}\n"
                    no_improvement_count = 0
                    self.logger.info("*** 新的最佳解! ***")
                else:
                    no_improvement_count += 1
                    self.logger.info(f"未找到更優解。連續未進步次數: {no_improvement_count}")

                # 評估推理品質(LLM-as-a-Judge, 使用 LLM 評估推理品質並且進行PARSING)
                evaluation_result = self._evaluate_reasoning(reasoning_string, model_name=model_name, history_pairs=str(self.history_log))
                eval_result_dict = evaluation_result.get("eval_result", "There is no evaluation result") 
                eval_raw_answer = evaluation_result.get("raw_answer", "")
                eval_time = evaluation_result.get("duration", 0)
                self.reasoning_times.append(r_time_i)  # 儲存推理耗時
                self.evaluation_times.append(eval_time) # 儲存評估耗時
                
                # 如果成功解析評估結果
                if eval_result_dict != "There is no evaluation result":
                    self.reasoning_scores.append(int(eval_result_dict["total_score"]))
                else:
                    # 如果評估結果解析失敗，嘗試從原始回應中提取數字作為分數
                    numbers = re.findall(r'\d+', str(eval_raw_answer))
                    if numbers:
                        # 如果找到數字，返回最後一個數字序列的整數形式
                        self.reasoning_scores.append(int(numbers[-1]))
                    else:
                        # 如果沒有找到數字，返回 0
                        self.reasoning_scores.append(0)
                self.history_log.append({"iteration": self.inner_iteration_count+1, "type": "Optimization", "reasoning": reasoning_string,"score": generated_score, "code":generated_code, "eval": self.reasoning_scores[-1] or 0, "r_time": r_time_i, "e_time": eval_time})


            else: # 未能獲取有效分數 (例如，程式碼執行失敗或未生成分數)
                self.logger.warning(f"Iteration {self.inner_iteration_count} 警告：未能獲取有效分數。")
                if len(self.numerical_scores) > 0:
                    self.numerical_scores.append(self.numerical_scores[-1])
                else:
                    self.numerical_scores.append(self.best_score)
                if len(self.reasoning_scores) > 0:
                    # 複製上一個推理分數或 0
                    self.reasoning_scores.append(self.reasoning_scores[-1]) # 重複上一個分數
                else:
                    self.reasoning_scores.append(0)

                no_improvement_count += 1
                self.logger.info(f"計為一次未進步。連續未進步次數: {no_improvement_count}")

                # 將當前迭代的詳細資訊記錄到歷史日誌中
                self.history_log.append({"iteration": self.inner_iteration_count+1, "type": "Optimization", "reasoning": reasoning_string,"score": self.numerical_scores[-1], "code":"operation failed", "eval": self.reasoning_scores[-1], "r_time": r_time_i, "e_time": eval_time})
            # 控制歷史日誌的長度，只保留最近的記錄
            if len(self.history_log) > max_history_length:
                self.history_log.pop(0) # 移除最舊的記錄

            # 判斷是否結束優化循環
            if is_finished: # 如果 LLM 認為已完成優化
                self.logger.info("\n模型回傳 'FINISHED'，結束優化流程。")
                break
            if no_improvement_count >= no_improvement_threshold: # 如果連續未改進次數達到閾值
                self.logger.info(f"\n連續 {no_improvement_count} 次未找到更優解，結束優化流程。")
                break
            if self.inner_iteration_count >= max_iterations-5:
                self.logger.info(f"\n達到最大迭代次數 {max_iterations}，結束優化流程。")
                break

        # --- 最終結果 ---
         # 記錄流程結束
        self.logger.info("\n\n" + "="*20 + " 優化流程結束 " + "="*20)
        self.logger.info(f"總共執行了 {len(self.numerical_scores)} 次有效的迭代。")
        self.logger.info(f"找到的最佳分數為: {self.best_score}")
        self.logger.info("\n--- [最佳解的詳細資訊] ---\n" + self.best_solution_details)
        self.logger.info("\n---------------------\n")
        
        # 如果任務描述中包含 ":)" (作為測試模式的提示)，則額外計算 TSP 的精確解 (DP4TSP)
        if ":)" in task_description:
            self.logger.info("額外加碼:與最佳解之間的距離")
            dp_start_time = time.perf_counter()
            dp_calculation = DP4TSP()
            dp_calculation.run(points)
            dp_run_time = time.perf_counter() - dp_start_time
            self.logger.info(f"DP執行時間: {dp_run_time:.2f} 秒")

        # 繪製最終進度圖表
        self._plot_progress()

# ===================================================================
# 這個區塊現在是作為 "命令列模式" 或 "測試模式" 的進入點
# 當你直接執行 `python main.py` 時，會執行這裡的程式碼
# 如果執行的是python app.py，則是作為 Flask 應用的入口點
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
        The distance between points is the Euclidean distance. :)
        """
        # 2. 建立一個預設的隨機數生成器
        rng = np.random.default_rng(42)

        # 3. 使用生成器的 .random() 方法來生成 25x2 的浮點數陣列
        #    注意：形狀 (shape) 是以一個元組 (tuple) (25, 2) 傳入
        data_points = rng.random((25, 2))

        # 4. 設定模型名稱
        model_name = "gemini-2.5-pro" # 設定一個預設模型
        
        # 5. 載入 API Keys (這部分邏輯不變)
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY") # or "OPENAI_API_KEY"
        if not api_key:
            raise ValueError("API Key (e.g., GOOGLE_API_KEY) not found in .env file.")

        logger.info(f"Task: TSP with {len(data_points)} points.")
        logger.info(f"Model: {model_name}")
        
        # 6. 將 config.py 的設定也印出來
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
        logger.error(f"An error occurred: {e}") # 記錄錯誤信息
        logger.error(f"Error details:\n{traceback.format_exc()}") # 記錄詳細的錯誤追溯
