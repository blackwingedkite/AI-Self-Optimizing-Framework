import numpy as np
from typing import Tuple, List
import re

class DP4TSP:
    def __init__(self):
        self.points = np.ndarray()
        self.distance_matrix = np.ndarray()
        self.min_cost = 0
        self.path = []

    def _calculate_distance_matrix(self, points: np.ndarray) -> np.ndarray:
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

    def _tsp_dp(self, distance_matrix: np.ndarray) -> Tuple[float, List[int]]:
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

    def _print_solution(self, distance_matrix, min_cost, path):
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
    # 額外的輔助函數：驗證數據一致性(by多嘴的claude)
    def run(self, points):
        distance_matrix = self._calculate_distance_matrix(points)
        min_cost, path = self._tsp_dp(distance_matrix)
        self._print_solution(distance_matrix, min_cost, path)

