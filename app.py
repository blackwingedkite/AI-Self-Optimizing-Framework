from flask import Flask, request, render_template_string, send_from_directory
import io
import sys
from main import SelfOptimizingFramework
import os
import numpy as np
from dotenv import load_dotenv
import logging
import glob
import traceback

load_dotenv()

app = Flask(__name__)

# --- Logging Setup ---
# 將 log 導向到一個在記憶體中的文字流，以便在網頁上顯示
log_stream = io.StringIO()
log_handler = logging.StreamHandler(log_stream)
# 格式化 log 輸出，使其更易讀
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# 取得並設定我們主要框架的 logger
logger = logging.getLogger("SelfOptimizingFramework")
logger.setLevel(logging.INFO)
# 清除任何既有的 handlers，確保 log 不會重複輸出
logger.handlers.clear()
logger.addHandler(log_handler)


# --- HTML Template ---
# 我們將使用一個字串來儲存 HTML 模板，方便修改
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
        .log { white-space: pre-wrap; background: #e9ecef; padding: 15px; border-radius: 4px; max-height: 400px; overflow-y: auto; font-family: "Courier New", Courier, monospace; }
        input[type="submit"] { background-color: #007bff; color: white; border: none; padding: 12px 20px; font-size: 16px; cursor: pointer; border-radius: 5px; transition: background-color 0.3s; }
        input[type="submit"]:hover { background-color: #0056b3; }
        input[type="submit"]:disabled { background-color: #6c757d; cursor: not-allowed; }
        .warning { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        img { max-width: 100%; border: 1px solid #dee2e6; border-radius: 4px; margin-top: 10px; }
    </style>
    <script>
        // ADDED: Simple JavaScript to provide feedback to the user on form submission
        function handleSubmit(form) {
            const submitButton = form.querySelector('input[type="submit"]');
            const loadingMessage = document.getElementById('loadingMessage');
            
            submitButton.value = 'Running... Please Wait';
            submitButton.disabled = true;
            loadingMessage.style.display = 'block';
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>AI Self-Optimizing Agent Demo</h1>
        
        <!-- ADDED: Warning message to inform the user about the long wait time -->
        <div id="loadingMessage" class="warning" style="display: none;">
            <strong>Processing...</strong> The AI is thinking. This may take several minutes. Please do not close or refresh the page.
        </div>

        <form method="POST" onsubmit="handleSubmit(this)">
            <div class="section">
                <label for="task_description"><b>Task Description:</b></label>
                <textarea id="task_description" name="task_description">{{ task_description }}</textarea>
            </div>
            
            <div class="section form-grid">
                <div>
                    <label for="n_points">Points (for TSP):</label>
                    <input id="n_points" type="number" name="n_points" value="{{ n_points }}" min="3" max="30">
                </div>
                <div>
                    <label for="max_iterations">Max Iterations:</label>
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

        {% if log_output %}
            <h2>Log Output</h2>
            <div class="log">{{ log_output }}</div>
        {% endif %}

        {% if plot_filename %}
            <h2>Progress Chart</h2>
            <img src="/plot/{{ plot_filename }}" alt="Progress Chart">
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/plot/<path:filename>')
def send_plot(filename):
    # 這個路由用來提供圖片檔案
    return send_from_directory('.', filename, as_attachment=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    # --- Default Values ---
    task_description = "Solve the Traveling Salesman Problem (TSP) for a given set of 2D points."
    n_points = 10
    max_iterations = 5
    no_improvement_threshold = 2
    max_history_length = 5
    temp = 0.4
    model_name = "gemini-1.5-pro-latest" # 使用一個常見的預設模型
    plot_filename = None
    log_output = ""

    # 清空上一次執行的 log
    log_stream.truncate(0)
    log_stream.seek(0)

    if request.method == 'POST':
        try:
            # --- Get data from form ---
            task_description = request.form.get('task_description', task_description)
            n_points = int(request.form.get('n_points', 10))
            max_iterations = int(request.form.get('max_iterations', max_iterations))
            no_improvement_threshold = int(request.form.get('no_improvement_threshold', no_improvement_threshold))
            max_history_length = int(request.form.get('max_history_length', max_history_length))
            temp = float(request.form.get('temp', temp))
            model_name = request.form.get('model_name', model_name).strip()

            # --- Run the framework ---
            points = np.random.rand(n_points, 2)

            # CHANGED: Pass model_name during initialization, not in run()
            framework = SelfOptimizingFramework(
                api_key=os.getenv("GOOGLE_API_KEY"),
            )

            # CHANGED: Removed model_name from the run() call
            framework.run(
                task_description=task_description,
                points=points,
                max_iterations=max_iterations,
                no_improvement_threshold=no_improvement_threshold,
                max_history_length=max_history_length,
                temp=temp,
                model_name=model_name,
            )

            # --- Find the latest plot file ---
            # 使用 glob 尋找所有符合格式的圖檔，並根據修改時間排序
            plot_files = sorted(glob.glob("progress_chart_*.png"), key=os.path.getmtime, reverse=True)
            if plot_files:
                plot_filename = os.path.basename(plot_files[0])
        
        except Exception as e:
            # 如果發生任何錯誤，將其記錄下來並顯示在 log 區域
            error_info = traceback.format_exc()
            logger.error(f"An error occurred: {e}\n{error_info}")


    # 讀取 log 內容並傳遞給模板
    log_output = log_stream.getvalue()
    
    return render_template_string(
        HTML_TEMPLATE,
        task_description=task_description,
        n_points=n_points,
        max_iterations=max_iterations,
        no_improvement_threshold=no_improvement_threshold,
        max_history_length=max_history_length,
        temp=temp,
        model_name=model_name, # FIXED: Pass model_name back to the template
        log_output=log_output,
        plot_filename=plot_filename
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
