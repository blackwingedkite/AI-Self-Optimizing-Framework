
    # def _extract_code_and_result(self, response: genai.types.GenerateContentResponse): old and too naive
    #     """從回應中提取程式碼、執行結果和數值分數"""
    #     code = ""
    #     output = ""
    #     score = None

    #     for part in response.candidates[0].content.parts:
    #         if part.executable_code:
    #             code = part.executable_code.code
    #             self.logger.info(f"--- [偵測到程式碼] ---\n{code}\n--------------------")
    #         if part.code_execution_result:
    #             output = part.code_execution_result.output
    #             self.logger.info(f"--- [程式碼執行結果] ---\n{output}\n--------------------")
        
    #     # 從輸出中提取數值分數 (假設最後一行是分數)
    #     if output:
    #         try:
    #             # 尋找最後一個數字
    #             numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    #             if numbers:
    #                 score = float(numbers[-1])
    #         except (ValueError, IndexError):
    #             self.logger.warning(f"無法從執行輸出中解析分數: {output}")
    #             score = None
        
    #     return code, output, score


    # def _parse_full_response(self, response: genai.types.GenerateContentResponse) -> dict: old good 2
    #     """
    #     從模型回應中安全地解析出所有部分，並確保回傳值的型別正確。
    #     """
    #     text_parts = []
    #     code = ""
    #     output = ""
    #     score = None

    #     # 防禦性檢查：確保 response 和 candidates 存在
    #     if not response or not response.candidates:
    #         self.logger.warning("收到了空的或無效的回應物件。")
    #         return {
    #             "reasoning": "",
    #             "code": "",
    #             "output": "",
    #             "score": None
    #         }

    #     for part in response.candidates[0].content.parts:
    #         # 1. 安全地取得所有文字片段
    #         if hasattr(part, 'text') and part.text:
    #             text_parts.append(part.text)
            
    #         # 2. 安全地取得程式碼
    #         if hasattr(part, 'executable_code') and part.executable_code:
    #             code = part.executable_code.code or "" # 確保不為 None
            
    #         # 3. 安全地取得程式碼執行結果
    #         if hasattr(part, 'code_execution_result') and part.code_execution_result:
    #             output = part.code_execution_result.output or "" # 確保不為 None

    #     # 組合所有文字片段作為完整的 reasoning
    #     full_reasoning = "\n".join(text_parts).strip()

    #     if output:
    #         try:
    #             lines = output.strip().split("\n")[-3:]  # 看最後 3 行
    #             for line in reversed(lines):
    #                 match = re.search(r"[-+]?\d*\.\d+|\d+", line)
    #                 if match:
    #                     score = float(match.group())
    #                     break
    #         except Exception as e:
    #             self.logger.warning(f"解析分數時失敗: {e}")
                
    #     return {
    #         "reasoning": full_reasoning,
    #         "code": code,
    #         "output": output,
    #         "score": score
    #     }
    
    
    # def _plot_progress(self):
    #     """將數值分數和推理品質分數視覺化"""
    #     if not self.scores or not self.reasoning_evals:
    #         self.logger.info("沒有足夠的數據來生成圖表。")
    #         return

    #     # 計算迭代次數並建立軸線
    #     num_iterations = len(self.scores)
    #     iterations_axis = range(1, num_iterations + 1)
        
    #     # 建立 2x1 的子圖
    #     fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 12))
    #     fig.suptitle('自我優化進程追蹤 (Self-Optimization Progress Tracking)', fontsize=16)

    #     # --- 圖 1: 分數 vs. 推理品質 ---
    #     ax1.set_title('分數與推理品質演進 (Score vs. Reasoning Quality)')
        
    #     # 左側 Y 軸：數值分數 (越低越好)
    #     color = 'tab:red'
    #     ax1.set_xlabel('迭代次數 (Iteration)')
    #     ax1.set_ylabel('問題解數值 (Numerical Score)', color=color)
    #     ax1.plot(iterations_axis, self.scores, 'o-', color=color, label='Numerical Score')
    #     ax1.tick_params(axis='y', labelcolor=color)
        
    #     # 標示出最佳分數（最低分數）
    #     if self.scores:
    #         min_score_val = min(self.scores)
    #         min_score_idx = self.scores.index(min_score_val)
    #         ax1.scatter(min_score_idx + 1, min_score_val, s=150, facecolors='none', 
    #                 edgecolors='gold', linewidth=2, label=f'Best Score: {min_score_val:.2f}')

    #     # 右側 Y 軸：推理品質分數 (越高越好)
    #     ax2 = ax1.twinx()
    #     color = 'tab:blue'
    #     reasoning_scores = [e.get('total_score', 0) for e in self.reasoning_evals]
    #     ax2.set_ylabel('推理品質分數 (Reasoning Quality Score)', color=color)
    #     iterations_axis = range(1, num_iterations + 2)
    #     ax2.plot(iterations_axis, reasoning_scores, 's--', color=color, label='Reasoning Score')
    #     ax2.tick_params(axis='y', labelcolor=color)
    #     ax2.set_ylim(0, 110)  # 分數範圍 0-100，稍微放寬到 110 以便顯示
        
    #     # 整合兩個軸的圖例
    #     lines, labels = ax1.get_legend_handles_labels()
    #     lines2, labels2 = ax2.get_legend_handles_labels()
    #     ax2.legend(lines + lines2, labels + labels2, loc='upper right')
        
    #     # 添加網格線
    #     ax1.grid(True)
    #     ax1.set_xticks(iterations_axis)  # 確保 X 軸刻度對應迭代次數

    #     # --- 圖 2: 時間成本分析 ---
    #     ax3.set_title('每輪迭代時間成本分析 (Time Cost Analysis per Iteration)')
    #     ax3.set_xlabel('迭代次數 (Iteration)')
    #     ax3.set_ylabel('耗時 (秒) (Time in Seconds)')
        
    #     # 準備堆疊長條圖數據 - 確保數據長度與迭代次數一致
    #     reasoning_times_padded = self.reasoning_times + [0] * (num_iterations - len(self.reasoning_times))
    #     evaluation_times_padded = self.evaluation_times + [0] * (num_iterations - len(self.evaluation_times))

    #     # 繪製堆疊長條圖
    #     ax3.bar(iterations_axis, reasoning_times_padded, label='推理耗時 (Reasoning Time)', color='coral')
    #     ax3.bar(iterations_axis, evaluation_times_padded, bottom=reasoning_times_padded, 
    #         label='評估耗時 (Evaluation Time)', color='skyblue')
        
    #     # 在長條圖上標示總時間
    #     totals = [i + j for i, j in zip(reasoning_times_padded, evaluation_times_padded)]
    #     for i, total in enumerate(totals):
    #         if total > 0:
    #             ax3.text(i + 1, total + 0.1, f'{total:.1f}s', ha='center')

    #     # 設定 X 軸刻度和圖例
    #     ax3.set_xticks(iterations_axis)
    #     ax3.legend()
    #     ax3.grid(True, axis='y')

    #     # 調整布局並儲存圖表
    #     fig.tight_layout(rect=[0, 0, 1, 0.96])
    #     plot_filename = f"progress_chart_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    #     plt.savefig(plot_filename)
    #     self.logger.info(f"進度圖表已儲存至 {plot_filename}")
    #     plt.show()    
