# app.py (修改後版本)

from flask import Flask, request, render_template_string, Response, stream_with_context, jsonify, send_from_directory
import io
import logging
import threading
import time
import os
import numpy as np
from dotenv import load_dotenv
import traceback
import glob

# 導入你的主要框架和設定
from main import SelfOptimizingFramework
from config import framework_config # 導入預設設定

load_dotenv()

app = Flask(__name__)

# --- 全域變數來處理狀態 ---
# 使用一個字典來保存日誌流，雖然在多使用者情境下不完美，但在 demo 中很實用
log_stream = io.StringIO()
# 使用一個執行緒鎖來確保日誌寫入的執行緒安全
log_lock = threading.Lock()
# 全域變數來保存背景執行緒，方便管理
background_thread = None

# --- Logging Setup ---
# 設定 logger 將日誌輸出到我們的記憶體流中
def setup_logger():
    # 清空之前的 handler，避免重複添加
    logger = logging.getLogger("SelfOptimizingFramework")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    
    # 建立一個 handler，將日誌寫入到 log_stream
    # 使用 lock 來確保多執行緒寫入時不會出錯
    class ThreadSafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            with log_lock:
                super().emit(record)

    handler = ThreadSafeStreamHandler(log_stream)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # 也加入一個輸出到 console 的 handler，方便在伺服器端除錯
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()


# --- HTML Template (已更新 JavaScript) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Self-Optimizing Agent</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; background-color: #f8f9fa; color: #343a40; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1, h2 { color: #0056b3; }
        textarea, input { width: 100%; padding: 10px; margin-bottom: 10px; border-radius: 4px; border: 1px solid #ced4da; box-sizing: border-box; }
        textarea { height: 120px; resize: vertical; }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }
        .section { margin-bottom: 25px; }
        /* 修改 .log 為 .log-container */
        .log-container { white-space: pre-wrap; background: #212529; color: #f8f9fa; padding: 15px; border-radius: 4px; max-height: 500px; overflow-y: auto; font-family: "Courier New", Courier, monospace; }
        input[type="submit"] { background-color: #007bff; color: white; border: none; padding: 12px 20px; font-size: 16px; cursor: pointer; border-radius: 5px; transition: background-color 0.3s; }
        input[type="submit"]:hover { background-color: #0056b3; }
        input[type="submit"]:disabled { background-color: #6c757d; cursor: not-allowed; }
        .status-message { background-color: #e9ecef; border: 1px solid #dee2e6; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        img { max-width: 100%; border: 1px solid #dee2e6; border-radius: 4px; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Self-Optimizing Agent Demo</h1>
        
        <div id="status" class="status-message">Ready. Fill the form and click "Run Optimization".</div>

        <form id="run-form" method="POST">
            <div class="section">
                <label for="task_description"><b>Task Description(HINT:":) in str"=測試exact解答):</b></label>
                <textarea id="task_description" name="task_description">{{ task_description }}</textarea>
            </div>
            
            <div class="section form-grid">
                <div>
                    <label for="n_points">Points (for TSP):</label>
                    <input id="n_points" type="number" name="n_points" value="{{ n_points }}" min="3" max="50">
                </div>
                <div>
                    <label for="max_iterations">Max Iterations(note:至少6次因為原本debate就佔了五次):</label>
                    <input id="max_iterations" type="number" name="max_iterations" value="{{ max_iterations }}">
                </div>
                <div>
                    <label for="no_improvement_threshold">No Improvement Threshold:</label>
                    <input id="no_improvement_threshold" type="number" name="no_improvement_threshold" value="{{ no_improvement_threshold }}">
                </div>
                <div>
                    <label for="max_history_length">Max History Length:</label>
                    <input id="max_history_length" type="number" name="max_history_length" value="{{ max_history_length }}">
                </div>
                <div>
                    <label for="temp">Temperature:</label>
                    <input id="temp" type="number" step="0.1" name="temp" value="{{ temp }}">
                </div>
                <div>
                    <label for="model_name">Model Name:</label>
                    <input id="model_name" type="text" name="model_name" value="{{ model_name }}">
                </div>
            </div>

            <input type="submit" value="Run Optimization">
        </form>

        <h2>Live Log Output</h2>
        <div id="log-output" class="log-container"></div>
        
        <div id="progress-plot-container" style="display:none;">
            <h2>Progress Chart</h2>
            <img id="progress-plot" src="" alt="Progress Chart">
        </div>

        <div id="debate-plot-container" style="display:none; margin-top: 20px;">
            <h2>Debate Chart</h2>
            <img id="debate-plot" src="" alt="Debate Chart">
        </div>

    </div>

    <script>
        // *** 全新的 JavaScript 區塊 ***
        const form = document.getElementById('run-form');
        const submitButton = form.querySelector('input[type="submit"]');
        const statusDiv = document.getElementById('status');
        const logOutputDiv = document.getElementById('log-output');
        const progressPlotContainer = document.getElementById('progress-plot-container');
        const progressPlotImage = document.getElementById('progress-plot');
        const debatePlotContainer = document.getElementById('debate-plot-container');
        const debatePlotImage = document.getElementById('debate-plot');

        let eventSource;

        form.addEventListener('submit', async (e) => {
            e.preventDefault(); // 防止表單傳統提交

            if (submitButton.disabled) return;

            // 1. 清空舊的輸出
            logOutputDiv.innerHTML = '';
            progressPlotContainer.style.display = 'none';
            progressPlotImage.src = '';
            debatePlotContainer.style.display = 'none';
            debatePlotImage.src = '';
            
            // 2. 禁用按鈕並更新狀態
            submitButton.disabled = true;
            submitButton.value = 'Running...';
            statusDiv.textContent = 'Starting optimization process...';
            statusDiv.style.backgroundColor = '#fff3cd'; // Yellowish warning color

            // 3. 發送請求到後端來啟動任務
            try {
                const formData = new FormData(form);
                const response = await fetch('/run', {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) {
                    throw new Error(`Server error: ${response.statusText}`);
                }
                
                const data = await response.json();
                if(data.status === 'started') {
                    // 4. 如果任務成功啟動，開始監聽 SSE 日誌流
                    startListening();
                } else {
                    throw new Error(data.error || 'Failed to start the task.');
                }

            } catch (error) {
                statusDiv.textContent = `Error starting task: ${error.message}`;
                statusDiv.style.backgroundColor = '#f8d7da'; // Reddish error color
                submitButton.disabled = false;
                submitButton.value = 'Run Optimization';
            }
        });

        function startListening() {
            // 建立一個 EventSource 連接到我們的日誌流端點
            eventSource = new EventSource('/log-stream');
            
            statusDiv.textContent = 'Process running. Streaming logs...';
            statusDiv.style.backgroundColor = '#d1ecf1'; // Bluish info color

            // 監聽 'message' 事件 (預設事件)
            eventSource.onmessage = function(event) {
                // event.data 包含伺服器發送的日誌行
                // 使用純文字避免 HTML 注入風險
                const textNode = document.createTextNode(event.data + '\\n');
                logOutputDiv.appendChild(textNode);
                // 自動滾動到最底部
                logOutputDiv.scrollTop = logOutputDiv.scrollHeight;
            };

            // 監聽我們自定義的 'end' 事件
            eventSource.addEventListener('end', function(event) {
                const finalData = JSON.parse(event.data);
                
                statusDiv.textContent = `Process finished. Status: ${finalData.message}`;
                statusDiv.style.backgroundColor = '#d4edda'; // Greenish success color
                
                // 檢查並顯示進度圖表
                if (finalData.progress_plot) {
                    progressPlotImage.src = '/plot/' + finalData.progress_plot;
                    progressPlotContainer.style.display = 'block';
                }

                // 檢查並顯示辯論圖表
                if (finalData.debate_plot) {
                    debatePlotImage.src = '/plot/' + finalData.debate_plot;
                    debatePlotContainer.style.display = 'block';
                }

                
                // 關閉連接並重設按鈕
                eventSource.close();
                submitButton.disabled = false;
                submitButton.value = 'Run Optimization';
            });
            
            // 處理錯誤
            eventSource.onerror = function(err) {
                console.error("EventSource failed:", err);
                statusDiv.textContent = 'Connection to log stream lost. Process may have failed.';
                statusDiv.style.backgroundColor = '#f8d7da'; // Reddish error color
                eventSource.close();
                submitButton.disabled = false;
                submitButton.value = 'Run Optimization';
            };
        }
    </script>
</body>
</html>
"""

def long_running_task(form_data):
    """這個函數在背景執行緒中執行，不會卡住主應用程式"""
    global log_stream
    
    # 清空上一次執行的 log
    with log_lock:
        log_stream.truncate(0)
        log_stream.seek(0)

    try:
        
        logger.info("Background task started.")
        
        # --- 從傳入的 form_data 中獲取參數 ---
        task_description = form_data.get('task_description')
        n_points = int(form_data.get('n_points'))
        max_iterations = int(form_data.get('max_iterations'))
        no_improvement_threshold = int(form_data.get('no_improvement_threshold'))
        max_history_length = int(form_data.get('max_history_length'))
        temp = float(form_data.get('temp'))
        model_name = form_data.get('model_name').strip()
        
        logger.info(f"Configuration: model={model_name}, points={n_points}, iterations={max_iterations}")

        # --- 運行框架 ---
        rng = np.random.default_rng(42)
        data_points = rng.random((n_points, 2))
        # --- 初始化 SelfOptimizingFramework ---
        # 注意：這裡假設 SelfOptimizingFramework 已經正確實現
        framework = SelfOptimizingFramework()
        
        # --- 執行框架 ---
        # 這裡的 run 方法應該是你在 main.py 中定義的
        framework.run(
            task_description=task_description,
            points=data_points,
            max_iterations=max_iterations,
            no_improvement_threshold=no_improvement_threshold,
            max_history_length=max_history_length,
            temp=temp,
            model_name=model_name,
        )

        logger.info("Framework run completed successfully.")

    except Exception as e:
        error_info = traceback.format_exc()
        logger.error(f"An error occurred in the background task: {e}\n{error_info}")


@app.route('/plot/<path:filename>')
def send_plot(filename):
    """這個路由用來提供圖片檔案"""
    # 從根目錄提供檔案
    return send_from_directory(app.root_path, filename, as_attachment=False)

@app.route('/', methods=['GET'])
def index():
    """顯示主頁面"""
    # 使用 config.py 的預設值，並讓模板渲染它們
    return render_template_string(
        HTML_TEMPLATE,
        task_description="""
Solve the Traveling Salesman Problem (TSP) for a given set of 2D points. 
There are 25 points. Time complexity is quite important so you ought to consider that. 
(now is testing mode. PLEASE USE HEURISTIC MODE TO ITERATE)
        """,
        n_points=25,
        model_name="gemini-2.5-pro",
        **framework_config # 將 config.py 的值展開傳入
    )

@app.route('/run', methods=['POST'])
def run_task():
    """啟動背景任務的端點"""
    global background_thread
    if background_thread and background_thread.is_alive():
        return jsonify({"status": "error", "error": "A task is already running."}), 409 # 409 Conflict

    form_data = request.form.to_dict()
    
    # 建立並啟動背景執行緒
    background_thread = threading.Thread(target=long_running_task, args=(form_data,))
    background_thread.start()
    
    return jsonify({"status": "started"})

@app.route('/log-stream')
def log_stream_route():
    """這是 SSE 端點，它會流式傳輸日誌"""
    def generate_logs():
        # 使用一個變數追蹤我們已經讀取到的日誌位置
        last_pos = 0
        
        # 當背景執行緒在執行時，持續檢查新日誌
        while background_thread and background_thread.is_alive():
            with log_lock:
                # 移動到上次讀取結束的位置
                log_stream.seek(last_pos)
                # 讀取所有新內容
                new_logs = log_stream.read()
                # 更新下次開始讀取的位置
                last_pos = log_stream.tell()
            
            if new_logs:
                # SSE 格式要求每條訊息以 "data: " 開頭，以 "\n\n" 結尾
                for line in new_logs.strip().split('\n'):
                    yield f"data: {line}\n\n"
            
            # 等待一小段時間，避免 CPU 過度空轉
            time.sleep(0.5)
        
        # --- 執行緒結束後，發送最後的日誌和結束信號 ---
        # 確保讀取到所有剩餘的日誌
        with log_lock:
            log_stream.seek(last_pos)
            remaining_logs = log_stream.read()
        if remaining_logs:
            for line in remaining_logs.strip().split('\n'):
                yield f"data: {line}\n\n"

        # 尋找最新的圖表檔案
        progress_plot_filename = None
        debate_plot_filename = None

        try:
            # 尋找最新的 progress chart
            # 你的 framework.run() 中儲存進度圖的檔名應為 progress_chart_...
            progress_files = sorted(glob.glob(os.path.join(app.root_path, "progress_chart_*.png")), key=os.path.getmtime, reverse=True)
            if progress_files:
                progress_plot_filename = os.path.basename(progress_files[0])
                logger.info(f"Found progress plot file: {progress_plot_filename}")
        except Exception as e:
            logger.error(f"Error finding progress plot file: {e}")

        try:
            # 尋找最新的 debate chart
            # 你的 framework.run() 中儲存辯論圖的檔名應為 debate_chart_...
            debate_files = sorted(glob.glob(os.path.join(app.root_path, "debate_chart_*.png")), key=os.path.getmtime, reverse=True)
            if debate_files:
                debate_plot_filename = os.path.basename(debate_files[0])
                logger.info(f"Found debate plot file: {debate_plot_filename}")
        except Exception as e:
            logger.error(f"Error finding debate plot file: {e}")
        # 發送一個自定義的 'end' 事件，包含最終狀態和圖片檔名
        import json
        end_data = json.dumps({
            "message": "Task completed.", 
            "progress_plot": progress_plot_filename,
            "debate_plot": debate_plot_filename
        })
        yield f"event: end\ndata: {end_data}\n\n"
    # 使用 Flask 的 Response 物件和 stream_with_context 來處理流式響應
    return Response(stream_with_context(generate_logs()), mimetype='text/event-stream')


if __name__ == '__main__':
    # 注意：Flask 的開發伺服器預設是單執行緒的。
    # 為了讓背景執行緒和主應用程式能同時運行，需要啟用 threaded=True
    # debug=True 會自動啟用 threaded=True
    app.run(debug=True, host='0.0.0.0', port=5001)