from flask import Flask, request, render_template_string, send_from_directory
import io
import sys
from main import SelfOptimizingFramework
import os
import numpy as np
from dotenv import load_dotenv
import logging
import glob

load_dotenv()

app = Flask(__name__)

# 儲存 log 內容的 buffer
log_stream = io.StringIO()
log_stream.truncate(0)
log_stream.seek(0)
# 替換 logger handler
log_handler = logging.StreamHandler(log_stream)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger("SelfOptimizingFramework")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(log_handler)

# 模板 HTML（可自訂美化）
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Self-Optimizing Agent</title>
    <style>
        textarea { width: 100%; height: 150px; }
        .section { margin-bottom: 20px; }
        .log { white-space: pre-wrap; background: #f5f5f5; padding: 10px; border: 1px solid #ccc; }
        input[type=number] { width: 80px; }
    </style>
</head>
<body>
    <h1>AI Self-Optimizing Agent Demo</h1>
    <form method="POST">
        <div class="section">
            <label>Task Description:</label><br>
            <textarea name="task_description">{{ task_description }}</textarea>
        </div>
        <div class="section">
            <label>Number of Points (for TSP):</label>
            <input type="number" name="n_points" value="{{ n_points }}" min="3" max="30">
        </div>
        <div class="section">
            <label>Max Iterations:</label>
            <input type="number" name="max_iterations" value="{{ max_iterations }}">
            <label>No Improvement Threshold:</label>
            <input type="number" name="no_improvement_threshold" value="{{ no_improvement_threshold }}">
            <label>Max History Length:</label>
            <input type="number" name="max_history_length" value="{{ max_history_length }}">
            <label>Temperature:</label>
            <input type="number" step="0.1" name="temp" value="{{ temp }}">
            <label>Type in your model(gemini-2.5-pro)</label>
            <input type="string" name="model_name" value="{{ model_name }}">
        </div>

        <input type="submit" value="Run">
    </form>

    {% if log_output %}
        <h2>Log Output</h2>
        <div class="log">{{ log_output }}</div>
    {% endif %}

    {% if plot_filename %}
        <h2>Progress Chart</h2>
        <img src="/plot/{{ plot_filename }}" style="max-width: 100%; border: 1px solid #000;">
    {% endif %}
</body>
</html>
"""

@app.route('/plot/<path:filename>')
def send_plot(filename):
    return send_from_directory('.', filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    task_description = "Solve the Traveling Salesman Problem (TSP) for a given set of 2D points."
    n_points = 10
    max_iterations = 5
    no_improvement_threshold = 2
    max_history_length = 5
    temp = 0.4
    model_name = "gemini-2.5-pro"
    plot_filename = None

    log_stream.truncate(0)
    log_stream.seek(0)

    if request.method == 'POST':
        task_description = request.form.get('task_description', task_description)
        n_points = int(request.form.get('n_points', 10))
        max_iterations = int(request.form.get('max_iterations', max_iterations))
        no_improvement_threshold = int(request.form.get('no_improvement_threshold', no_improvement_threshold))
        max_history_length = int(request.form.get('max_history_length', max_history_length))
        temp = float(request.form.get('temp', temp))
        model_name = str(request.form.get('model_name', model_name))

        points = np.random.rand(n_points, 2)

        framework = SelfOptimizingFramework(
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

        framework.run(
            task_description=task_description,
            points=points,
            max_iterations=max_iterations,
            no_improvement_threshold=no_improvement_threshold,
            max_history_length=max_history_length,
            temp=temp,
            model_name=model_name,
        )

        # 尋找最新的圖檔
        plot_files = sorted(glob.glob("progress_chart_*.png"), key=os.path.getmtime, reverse=True)
        if plot_files:
            plot_filename = os.path.basename(plot_files[0])

    log_output = log_stream.getvalue()
    return render_template_string(
        HTML_TEMPLATE,
        task_description=task_description,
        n_points=n_points,
        max_iterations=max_iterations,
        no_improvement_threshold=no_improvement_threshold,
        max_history_length=max_history_length,
        temp=temp,
        # model_name=model_name,
        log_output=log_output,
        plot_filename=plot_filename
    )

if __name__ == '__main__':
    app.run(debug=True)
