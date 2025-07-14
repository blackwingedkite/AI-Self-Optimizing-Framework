# 🧠 AI Self-Optimizing Framework

一個基於 Google Gemini 模型的 AI 代理程式，具備**自我優化能力**，可透過迭代方式解決任意明確定義的優化問題。

---

## 🚀 專案特色

* **自我優化迴圈**：AI 分析問題 → 撰寫初始解法 → 評估並改進 → 反覆進化直至滿足條件。
* **問題類型無關（Problem-Agnostic）**：內建為 TSP，但任何有明確定義的優化問題都可支援。
* **視覺化追蹤**：自動產生進度圖表，顯示數值與推理品質變化，幫助理解 AI 的「思考歷程」。
* **簡單 UI**：透過 Flask 提供直覺操作的 Web 介面。

---

## 📁 專案結構

| 檔案                 | 功能簡介                                                            |
| ------------------ | --------------------------------------------------------------- |
| `app.py`           | 專案入口。啟動 Flask 網頁伺服器，處理使用者輸入與結果展示。                               |
| `main.py`          | 核心 AI 框架 `SelfOptimizingFramework` 實作，處理問題分類、初始實作、迭代優化、模型互動等流程。 |
| `dp_for_tsp.py`    | 使用動態規劃求解 TSP 最佳解，供與 AI 解法做比較（僅限 TSP 問題）。                        |
| `.env`             | 儲存 Google Gemini API 金鑰。                                        |
| `requirements.txt` | 所需套件清單。                                                         |

---

## ⚙️ 安裝與啟動

### 1. 環境需求

* Python 3.8+
* 建議使用虛擬環境（如 `venv`、`conda`）

### 2. 安裝套件

```bash
pip install -r requirements.txt
```

### 3. 設定 API 金鑰

在專案根目錄建立 `.env` 檔，並加入以下內容：

```env
GOOGLE_API_KEY="your-api-key-here"
```

### 4. 啟動應用程式

```bash
python app.py
```

成功後將看到：

```
* Running on http://127.0.0.1:5001
```

打開瀏覽器並前往該網址。

---

## 💡 如何使用

### 🧭 預設：解決 TSP 問題

1. 啟動網頁後，保留預設的 Task Description：

   ```text
   Solve the Traveling Salesman Problem (TSP) for a given set of 2D points.
   The goal is to find the shortest possible tour that visits each point exactly once and returns to the origin point.
   The distance between points is the Euclidean distance.
   ```
2. 點擊 **Run Optimization**
3. 等待 AI 完成多輪優化（需幾分鐘）
4. 查看最終路徑與進度圖表，AI 結果也會與動態規劃的最佳解做比較。

---

### ✏️ 自訂任務

你可以改寫 Task Description 來解決任何其他優化問題。

**範例：歐洲五國旅行問題**

```text
I am a traveller in Europe and I want to visit five cities. 
The locations of these cities are (1,5), (2,-3), (10,-11), (-2,10), (5,5). 
My goal is to find the shortest possible path that visits each city exactly once, 
starting from any city and ending at any city. 
The distance between cities is the standard Euclidean distance. 
Please provide the optimal sequence of cities to visit and the total minimum distance.
```

* 「Points」欄位會被忽略，請在任務描述中明確列出點位。
* 可調整其他參數如迭代次數、溫度等。
* 點擊執行，即可看到 AI 解決過程與結果。

---

## 📊 輸出範例

* **推理品質變化圖**
* **路徑總長度演進**
* **每輪花費時間**
* **AI 解法與理論最佳解（TSP 問題專屬）比較**

---

## 🧪 測試與比較

若使用預設 TSP 任務，系統會呼叫 `dp_for_tsp.py` 算出理論最短路徑，並與 AI 產出的結果比較差距。

---

## 🛡️ 注意事項

* 每次優化會多次呼叫 Gemini API，請注意配額與金鑰安全。
* 非常依賴任務描述的清晰度，請具體定義問題！

---

## 📍未來展望

* 支援更多模型（非 Gemini）
* 增加任務類型模組（如排程、資源分配等）
* Web 介面強化（如中途暫停、重跑、版本紀錄）

---

有了這個 README，就能吸引開發者理解並使用你的框架了！
需要我幫你補上範例圖、部署方式、或轉成 GitHub Pages 展示也沒問題，要不要試著進一步包裝成 demo app？ 😏
