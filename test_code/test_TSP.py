import numpy as np
from typing import Tuple, List

def calculate_distance_matrix(points: np.ndarray) -> np.ndarray:
    """
    根據點的座標計算距離矩陣
    
    Args:
        points: n×2的陣列，每行代表一個點的(x, y)座標
        
    Returns:
        n×n的距離矩陣
    """
    n = len(points)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # 計算歐幾里得距離
                distance_matrix[i][j] = np.sqrt(
                    (points[i][0] - points[j][0])**2 + 
                    (points[i][1] - points[j][1])**2
                )
    
    return distance_matrix

def tsp_dp(distance_matrix: np.ndarray) -> Tuple[float, List[int]]:
    """
    使用動態規劃解決TSP問題
    
    Args:
        distance_matrix: n×n的距離矩陣，distance_matrix[i][j]表示城市i到城市j的距離
        
    Returns:
        Tuple[float, List[int]]: (最短距離, 最短路徑)
    """
    n = len(distance_matrix)
    if n <= 1:
        return 0, [0] if n == 1 else []
    
    # dp[mask][i] 表示從起點0出發，經過mask中包含的所有城市，最後到達城市i的最短距離
    # mask用位元表示，第i位為1表示已訪問城市i
    dp = [[float('inf')] * n for _ in range(1 << n)]
    parent = [[None] * n for _ in range(1 << n)]
    
    # 初始化：從起點0開始，只訪問城市0
    dp[1][0] = 0  # mask=1表示只訪問了城市0
    
    # 遍歷所有可能的訪問狀態
    for mask in range(1 << n):
        for u in range(n):
            # 如果城市u不在當前mask中，跳過
            if not (mask & (1 << u)):
                continue
            
            # 如果當前狀態不可達，跳過
            if dp[mask][u] == float('inf'):
                continue
            
            # 嘗試從城市u前往其他未訪問的城市v
            for v in range(n):
                if mask & (1 << v):  # 城市v已經訪問過
                    continue
                
                new_mask = mask | (1 << v)  # 加入城市v
                new_dist = dp[mask][u] + distance_matrix[u][v]
                
                if new_dist < dp[new_mask][v]:
                    dp[new_mask][v] = new_dist
                    parent[new_mask][v] = u
    
    # 找到最短的完整路徑（訪問所有城市後回到起點）
    full_mask = (1 << n) - 1  # 所有城市都訪問過
    min_cost = float('inf')
    last_city = -1
    
    for i in range(1, n):  # 不考慮起點0
        cost = dp[full_mask][i] + distance_matrix[i][0]  # 回到起點的成本
        if cost < min_cost:
            min_cost = cost
            last_city = i
    
    # 重構路徑
    path = []
    current_city = last_city
    current_mask = full_mask
    
    while current_city is not None:
        path.append(current_city)
        next_city = parent[current_mask][current_city]
        current_mask ^= (1 << current_city)  # 移除當前城市
        current_city = next_city
    
    path.reverse()
    path.append(0)  # 回到起點
    
    return min_cost, path

def solve_tsp_from_points(points: np.ndarray) -> Tuple[float, List[int], np.ndarray]:
    """
    從點座標直接求解TSP問題
    
    Args:
        points: n×2的陣列，每行代表一個點的(x, y)座標
        
    Returns:
        Tuple[float, List[int], np.ndarray]: (最短距離, 最短路徑, 距離矩陣)
    """
    distance_matrix = calculate_distance_matrix(points)
    min_cost, path = tsp_dp(distance_matrix)
    return min_cost, path, distance_matrix
def print_solution(distance_matrix, min_cost, path):
    """
    印出TSP問題的解
    """
    n = len(distance_matrix)
    print(f"城市數量: {n}")
    print(f"最短距離: {min_cost:.2f}")
    print(f"最短路徑: {' -> '.join(map(str, path))}")
    
    # 驗證路徑距離
    total_dist = 0
    for i in range(len(path) - 1):
        dist = distance_matrix[path[i]][path[i+1]]
        total_dist += dist
        print(f"城市{path[i]} -> 城市{path[i+1]}: {dist:.2f}")
    
    print(f"總距離驗證: {total_dist:.2f}")

# 測試範例
if __name__ == "__main__":
    # 你的輸入格式
    points = np.array([[0.29303984, 0.3331745 ], [0.10365188, 0.09248663], [0.40899293, 0.11684273], [0.82672189, 0.91661797], [0.52290687, 0.38457594], [0.647582  , 0.97536996], [0.89547405, 0.89837302], [0.07951031, 0.01150751], [0.31426983, 0.73265848], [0.47019639, 0.25308003], [0.18179352, 0.6371844 ], [0.92110219, 0.87675957], [0.44479641, 0.56412684], [0.05515804, 0.69835458], [0.24239975, 0.20209481], [0.28680622, 0.8695735 ], [0.71774221, 0.45502662], [0.16620445, 0.40050214], [0.34522652, 0.07450954], [0.78136975, 0.62966154]])
    # 直接求解
    min_cost, path, distance_matrix = solve_tsp_from_points(points)
    print(f"最短距離: {min_cost}")
    print(f"最短路徑: {path}")
    # 範例1: 4個城市的距離矩陣
    # print("=== 範例1: 4個城市 ===")
    # distance_matrix_4 = np.array([
    #     [0, 10, 15, 20],
    #     [10, 0, 35, 25],
    #     [15, 35, 0, 30],
    #     [20, 25, 30, 0]
    # ])
    
    # min_cost, path = tsp_dp(distance_matrix_4)
    # print_solution(distance_matrix_4, min_cost, path)
    
    # print("\n" + "="*50 + "\n")
    
    # 範例2: 5個城市的隨機距離矩陣
    # print("=== 範例2: 5個城市 ===")
    # np.random.seed(42)
    # n = 5
    # distance_matrix_5 = np.random.randint(1, 100, size=(n, n))
    # print(distance_matrix_5)
    # # 確保對角線為0，且矩陣對稱
    # for i in range(n):
    #     distance_matrix_5[i][i] = 0
    #     for j in range(i+1, n):
    #         distance_matrix_5[j][i] = distance_matrix_5[i][j]
    
    # print("距離矩陣:")
    # print(distance_matrix_5)
    # print()
    
    # min_cost, path = tsp_dp(distance_matrix_5)
    # print_solution(distance_matrix_5, min_cost, path)