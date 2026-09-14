# ==============================================================================
# [실습] 지하철 노선망으로 배우는 탐색 알고리즘
# ==============================================================================

# [자료 1] 무가중치 그래프 (DFS / BFS 용)
subway_graph = {
    "시청": ["신도림", "동대문", "용산", "강남"],
    "신도림": ["시청", "동대문"],
    "동대문": ["시청", "신도림", "강남", "잠실"],
    "용산": ["시청", "강남"],
    "강남": ["시청", "동대문", "용산", "잠실"],
    "잠실": ["동대문", "강남"]
}

# [자료 2] 가중치 그래프 (다익스트라 용: 소요 시간(분))
subway_weighted_map = {
    "시청": [("신도림", 12), ("동대문", 10), ("용산", 8), ("강남", 15)],
    "신도림": [("시청", 12), ("동대문", 14)],
    "동대문": [("시청", 10), ("신도림", 14), ("강남", 13), ("잠실", 11)],
    "용산": [("시청", 8), ("강남", 9)],
    "강남": [("시청", 15), ("동대문", 13), ("용산", 9), ("잠실", 10)],
    "잠실": [("동대문", 11), ("강남", 10)]
}

subway_graph = {
    "시청": ["신도림", "동대문", "용산", "강남"],
    "신도림": ["시청", "동대문"],
    "동대문": ["시청", "신도림", "강남", "잠실"],
    "용산": ["시청", "강남"],
    "강남": ["시청", "동대문", "용산", "잠실"],
    "잠실": ["동대문", "강남"]
}
# --- 1. DFS (깊이 우선 탐색) ---
def dfs(graph, start_station):
    visited = []
    stack = [start_station]

    while stack:
        current = stack.pop()  # 맨 뒤(가장 최근) 데이터 꺼내기

        if current not in visited:
            visited.append(current)
            for neighbor in reversed(graph[current]):
                if neighbor not in visited:
                    stack.append(neighbor)
    return visited


# --- 2. BFS (너비 우선 탐색) ---
def bfs(graph, start_station):
    visited = []
    queue = [start_station]

    while queue:
        current = queue.pop(0)  # 줄의 맨 앞(가장 먼저 들어온 것) 데이터 꺼내기

        if current not in visited:
            visited.append(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
    return visited


# --- 3. 다익스트라 최단 경로 알고리즘 ---
def dijkstra(graph, start_station, end_station):
    times = {station: 999999 for station in graph}
    times[start_station] = 0
    previous = {station: None for station in graph}
    visited = []

    for _ in range(len(graph)):
        # 미방문 정류장 중 소요 시간이 가장 작은 곳 선택
        current_station = None
        min_time = 999999

        for station in graph:
            if station not in visited and times[station] < min_time:
                min_time = times[station]
                current_station = station

        if current_station is None:
            break

        visited.append(current_station)

        # 이웃 정류장 소요 시간 갱신
        for next_station, travel_time in graph[current_station]:
            new_time = times[current_station] + travel_time
            if new_time < times[next_station]:
                times[next_station] = new_time
                previous[next_station] = current_station

    # 경로 역추적
    path = []
    curr = end_station
    while curr is not None:
        path.append(curr)
        if curr == start_station:
            break
        curr = previous[curr]

    if path[-1] != start_station:
        return None, []

    return times[end_station], list(reversed(path))