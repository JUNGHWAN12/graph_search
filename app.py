import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import copy
import time

# 페이지 설정
st.set_page_config(
    page_title="지하철 노선망 탐색 알고리즘 시각화",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# 1. 커스텀 CSS (프리미엄 모던 메트로 스타일)
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * { font-family: 'Pretendard', sans-serif; }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 50%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .algo-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-dfs { background-color: #FEF3C7; color: #92400E; }
    .badge-bfs { background-color: #DBEAFE; color: #1E40AF; }
    .badge-dijkstra { background-color: #D1FAE5; color: #065F46; }
    
    .ds-box {
        font-family: 'Consolas', monospace;
        background: #1E293B;
        color: #F8FAFC;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 6px 0;
        font-size: 0.95rem;
    }
    .step-log {
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. 기본 그래프 데이터 (search.py 기반 기본값)
# ------------------------------------------------------------------------------
DEFAULT_PYTHON_CODE = '''# ==============================================================================
# [자료 1] 무가중치 그래프 (DFS / BFS 용)
# ==============================================================================
subway_graph = {
    "시청": ["신도림", "동대문", "용산", "강남"],
    "신도림": ["시청", "동대문"],
    "동대문": ["시청", "신도림", "강남", "잠실"],
    "용산": ["시청", "강남"],
    "강남": ["시청", "동대문", "용산", "잠실"],
    "잠실": ["동대문", "강남"]
}

# ==============================================================================
# [자료 2] 가중치 그래프 (다익스트라 용: 소요 시간(분))
# ==============================================================================
subway_weighted_map = {
    "시청": [("신도림", 12), ("동대문", 10), ("용산", 8), ("강남", 15)],
    "신도림": [("시청", 12), ("동대문", 14)],
    "동대문": [("시청", 10), ("신도림", 14), ("강남", 13), ("잠실", 11)],
    "용산": [("시청", 8), ("강남", 9)],
    "강남": [("시청", 15), ("동대문", 13), ("용산", 9), ("잠실", 10)],
    "잠실": [("동대문", 11), ("강남", 10)]
}
'''

# 서울 실제 지리적 방위각 기반 정밀 2D 좌표 프리셋 (확장 8개 역 포함)
GEO_COORDINATES = {
    "홍대": (-0.85, 0.25),
    "신촌": (-0.45, 0.35),
    "시청": (0.0, 0.35),
    "동대문": (0.65, 0.40),
    "신도림": (-0.85, -0.35),
    "용산": (-0.15, -0.10),
    "강남": (0.35, -0.55),
    "잠실": (0.85, -0.45)
}

# ------------------------------------------------------------------------------
# 3. 레이아웃 알고리즘 함수 (교차 최소화 & 깨끗한 그래프 배치)
# ------------------------------------------------------------------------------
def compute_graph_layout(G: nx.Graph, layout_mode: str = "kamada_kawai"):
    """
    그래프 시각화의 교차를 최소화하고 균형 잡힌 노드 좌표를 계산합니다.
    - kamada_kawai: 그래프 최단 경로 거리 기반 스프링 에너지 최소화 (가장 깔끔하고 교차 적음)
    - spring: Fruchterman-Reingold 힘 지향 물리력 기반 분산 배치
    - geographic: 서울 실제 지리적 방위각 기반 정밀 배치
    - circular: 원형 둘레 균등 분산 배치
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}

    # 1. 서울 실제 지리적 배치
    if layout_mode == "geographic":
        if all(node in GEO_COORDINATES for node in nodes):
            return {node: GEO_COORDINATES[node] for node in nodes}
        pos = {}
        fixed_nodes = []
        for node in nodes:
            if node in GEO_COORDINATES:
                pos[node] = GEO_COORDINATES[node]
                fixed_nodes.append(node)
        if len(fixed_nodes) == n:
            return pos
        if len(fixed_nodes) > 0:
            try:
                return nx.spring_layout(G, pos=pos, fixed=fixed_nodes, seed=42, k=1.2)
            except Exception:
                pass

    # 2. 원형 배치 (Circular / Ring)
    if layout_mode == "circular":
        return nx.circular_layout(G)

    # 3. 카마다-카와이 알고리즘 (Kamada-Kawai: 교차 최소화 & 대칭 균형 최고)
    if layout_mode == "kamada_kawai":
        try:
            if nx.is_connected(G):
                return nx.kamada_kawai_layout(G, weight="weight")
            else:
                return nx.spring_layout(G, seed=42, k=1.8 / (n ** 0.5), iterations=150)
        except Exception:
            pass

    # 4. 스프링 포스 알고리즘 (Fruchterman-Reingold)
    k_optimal = max(0.8, 2.0 / (n ** 0.5))
    return nx.spring_layout(G, seed=42, k=k_optimal, iterations=200)


# ------------------------------------------------------------------------------
# 4. 파이썬 코드 파싱 및 그래프 변환 함수
# ------------------------------------------------------------------------------
def parse_python_graph_code(code_str: str):
    """
    사용자가 입력한 파이썬 코드를 실행하여 subway_graph와 subway_weighted_map을 추출합니다.
    누락된 그래프는 상호 변환하여 보완합니다.
    """
    local_scope = {}
    try:
        exec(code_str, {}, local_scope)
    except Exception as e:
        return None, None, f"파이썬 코드 실행 중 오류가 발생했습니다: {e}"

    graph = local_scope.get("subway_graph") or local_scope.get("graph")
    weighted_map = local_scope.get("subway_weighted_map") or local_scope.get("weighted_graph") or local_scope.get("graph_weighted")

    # 가중치 맵만 있는 경우 무가중치 그래프 생성
    if weighted_map and not graph:
        graph = {}
        for u, neighbors in weighted_map.items():
            graph[u] = [v for v, _ in neighbors]

    # 무가중치 그래프만 있는 경우 기본 가중치(10분)로 가중치 맵 생성
    if graph and not weighted_map:
        weighted_map = {}
        for u, neighbors in graph.items():
            weighted_map[u] = [(v, 10) for v in neighbors]

    if not graph or not weighted_map:
        return None, None, "코드에서 'subway_graph' 또는 'subway_weighted_map' 딕셔너리를 찾을 수 없습니다."

    return graph, weighted_map, None


# ------------------------------------------------------------------------------
# 5. 단계별(Step-by-Step) 알고리즘 트래커
# ------------------------------------------------------------------------------
def trace_dfs(graph, start_station):
    """DFS의 각 단계(Stack 상태, 방문 노드, 현재 노드, 탐색 트리 간선)를 기록"""
    steps = []
    visited = []
    # 스택에 (역 이름, 부모 역 이름) 튜플 저장하여 실제 이동한 간선 추적
    stack = [(start_station, None)]
    tree_edges = []

    steps.append({
        "step_num": 0,
        "current": None,
        "action": f"탐색 시작: 출발역 '{start_station}'을 스택에 푸시",
        "stack": [start_station],
        "visited": list(visited),
        "highlight_edges": []
    })

    while stack:
        current, parent_node = stack.pop()
        
        if current not in visited:
            visited.append(current)
            if parent_node is not None:
                tree_edges.append((parent_node, current))

            action_desc = f"스택에서 pop() -> 현재 역 '{current}' 방문 처리"
            
            added_neighbors = []
            for neighbor in reversed(graph.get(current, [])):
                if neighbor not in visited:
                    stack.append((neighbor, current))
                    added_neighbors.append(neighbor)
            
            if added_neighbors:
                action_desc += f" (미방문 이웃역 {added_neighbors[::-1]} 스택에 역순 푸시)"
                
            steps.append({
                "step_num": len(steps),
                "current": current,
                "action": action_desc,
                "stack": [s for s, _ in stack],
                "visited": list(visited),
                "highlight_edges": list(tree_edges)
            })
        else:
            steps.append({
                "step_num": len(steps),
                "current": current,
                "action": f"스택에서 pop() -> '{current}'은 이미 방문함 (건너뜀)",
                "stack": [s for s, _ in stack],
                "visited": list(visited),
                "highlight_edges": list(tree_edges)
            })

    steps.append({
        "step_num": len(steps),
        "current": None,
        "action": f"탐색 완료: 스택이 비었습니다. 총 {len(visited)}개 역 방문 완료!",
        "stack": [],
        "visited": list(visited),
        "highlight_edges": list(tree_edges)
    })
    return steps


def trace_bfs(graph, start_station):
    """BFS의 각 단계(Queue 상태, 방문 노드, 현재 노드, 탐색 트리 간선)를 기록"""
    steps = []
    visited = []
    queue = [start_station]
    # 각 역을 처음 발견한 부모 역을 기록하여 실제 방문 간선(Tree Edge) 추적
    parent_map = {start_station: None}
    tree_edges = []

    steps.append({
        "step_num": 0,
        "current": None,
        "action": f"탐색 시작: 출발역 '{start_station}'을 큐에 인큐(줄서기)",
        "queue": list(queue),
        "visited": list(visited),
        "highlight_edges": []
    })

    while queue:
        current = queue.pop(0)

        if current not in visited:
            visited.append(current)
            # 부모 노드가 있으면 실제 탐색 간선으로 추가 (예: 시청 -> 용산)
            if parent_map.get(current) is not None:
                tree_edges.append((parent_map[current], current))

            action_desc = f"큐의 맨 앞에서 pop(0) -> 현재 역 '{current}' 방문 처리"
            
            added_neighbors = []
            for neighbor in graph.get(current, []):
                if neighbor not in visited and neighbor not in queue:
                    queue.append(neighbor)
                    parent_map[neighbor] = current  # neighbor의 부모는 current
                    added_neighbors.append(neighbor)

            if added_neighbors:
                action_desc += f" (이웃역 {added_neighbors} 큐의 뒤에 추가)"

            steps.append({
                "step_num": len(steps),
                "current": current,
                "action": action_desc,
                "queue": list(queue),
                "visited": list(visited),
                "highlight_edges": list(tree_edges)
            })
        else:
            steps.append({
                "step_num": len(steps),
                "current": current,
                "action": f"큐에서 pop(0) -> '{current}'은 이미 방문함 (건너뜀)",
                "queue": list(queue),
                "visited": list(visited),
                "highlight_edges": list(tree_edges)
            })

    steps.append({
        "step_num": len(steps),
        "current": None,
        "action": f"탐색 완료: 큐가 비었습니다. 총 {len(visited)}개 역 방문 완료!",
        "queue": [],
        "visited": list(visited),
        "highlight_edges": list(tree_edges)
    })
    return steps


def trace_dijkstra(graph_weighted, start_station, end_station):
    """다익스트라 알고리즘의 각 단계(times 테이블 갱신, 선택된 노드, previous 추적)를 기록"""
    steps = []
    times = {station: 999999 for station in graph_weighted}
    times[start_station] = 0
    previous = {station: None for station in graph_weighted}
    visited = []

    steps.append({
        "step_num": 0,
        "current": start_station,
        "action": f"초기화: 출발역 '{start_station}' 소요 시간 0분, 나머지 999999분 설정",
        "times": copy.deepcopy(times),
        "previous": copy.deepcopy(previous),
        "visited": list(visited),
        "updated_stations": [start_station]
    })

    for round_idx in range(len(graph_weighted)):
        current_station = None
        min_time = 999999

        for station in graph_weighted:
            if station not in visited and times[station] < min_time:
                min_time = times[station]
                current_station = station

        if current_station is None:
            break

        visited.append(current_station)
        
        updated_in_this_step = []
        updates_desc = []
        for next_station, travel_time in graph_weighted[current_station]:
            new_time = times[current_station] + travel_time
            if new_time < times[next_station]:
                old_time = times[next_station]
                times[next_station] = new_time
                previous[next_station] = current_station
                updated_in_this_step.append(next_station)
                updates_desc.append(f"{next_station}({old_time}분 ➔ {new_time}분)")

        action_text = f"[{round_idx+1}단계] 미방문 역 중 최단 시간인 '{current_station}'({min_time}분) 확정"
        if updates_desc:
            action_text += f" ➔ 이웃 역 단축 갱신: {', '.join(updates_desc)}"
        else:
            action_text += " ➔ 더 빠른 갱신 경로 없음"

        steps.append({
            "step_num": len(steps),
            "current": current_station,
            "action": action_text,
            "times": copy.deepcopy(times),
            "previous": copy.deepcopy(previous),
            "visited": list(visited),
            "updated_stations": updated_in_this_step
        })

        if current_station == end_station:
            break

    # 경로 역추적
    path = []
    curr = end_station
    while curr is not None:
        path.append(curr)
        if curr == start_station:
            break
        curr = previous[curr]

    final_path = list(reversed(path)) if path and path[-1] == start_station else []
    total_time = times[end_station] if final_path else None

    steps.append({
        "step_num": len(steps),
        "current": end_station,
        "action": f"목적지 '{end_station}' 도달 완료! 최단 시간: {total_time}분, 최단 경로: {' ➔ '.join(final_path)}",
        "times": copy.deepcopy(times),
        "previous": copy.deepcopy(previous),
        "visited": list(visited),
        "updated_stations": [],
        "final_path": final_path,
        "total_time": total_time
    })

    return steps, final_path, total_time


# ------------------------------------------------------------------------------
# 6. Plotly 고해상도 깔끔 그래프 렌더러
# ------------------------------------------------------------------------------
def create_graph_figure(
    graph,
    weighted_map,
    highlight_nodes=None,
    current_node=None,
    highlight_edges=None,
    queue_or_stack_nodes=None,
    start_node=None,
    end_node=None,
    title="지하철 노선망 그래프",
    layout_mode="kamada_kawai"
):
    """Plotly를 이용해 겹침 없이 깔끔한 네트워크 시각화 생성"""
    G = nx.Graph()
    for u, neighbors in weighted_map.items():
        for item in neighbors:
            if isinstance(item, (tuple, list)):
                v, w = item[0], item[1]
            else:
                v, w = item, 10
            G.add_edge(u, v, weight=w)

    pos = compute_graph_layout(G, layout_mode)

    # 기본 간선 그리기
    edge_x = []
    edge_y = []

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2.8, color='#94A3B8'),
        hoverinfo='none',
        mode='lines'
    )

    # 하이라이트 간선 (최단 경로 등)
    hl_edge_x = []
    hl_edge_y = []
    if highlight_edges:
        hl_set = set()
        for u, v in highlight_edges:
            hl_set.add((u, v))
            hl_set.add((v, u))
        for u, v in G.edges():
            if (u, v) in hl_set:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                hl_edge_x.extend([x0, x1, None])
                hl_edge_y.extend([y0, y1, None])

    hl_edge_trace = go.Scatter(
        x=hl_edge_x, y=hl_edge_y,
        line=dict(width=6.5, color='#EF4444'),
        hoverinfo='none',
        mode='lines'
    )

    # 간선 가중치 배지 어노테이션
    annotations = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        weight_val = d.get('weight', '')
        
        is_hl = False
        if highlight_edges:
            for hu, hv in highlight_edges:
                if (u == hu and v == hv) or (u == hv and v == hu):
                    is_hl = True
                    break

        annotations.append(dict(
            x=mid_x,
            y=mid_y,
            text=f"<b>{weight_val}분</b>",
            showarrow=False,
            font=dict(
                size=11, 
                color='#DC2626' if is_hl else '#1E293B',
                family='Pretendard, sans-serif'
            ),
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#EF4444' if is_hl else '#CBD5E1',
            borderwidth=1.5 if is_hl else 1,
            borderpad=3,
            opacity=0.98
        ))

    # 노드 색상 및 크기 결정
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    node_borders = []
    node_border_widths = []

    highlight_nodes = highlight_nodes or []
    queue_or_stack_nodes = queue_or_stack_nodes or []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"<b>{node}</b>")

        if node == current_node:
            node_colors.append('#F59E0B')      # 현재 방문 노드: 밝은 주황
            node_sizes.append(44)
            node_borders.append('#78350F')
            node_border_widths.append(4)
        elif node == start_node:
            node_colors.append('#8B5CF6')      # 출발역: 보라색
            node_sizes.append(40)
            node_borders.append('#4C1D95')
            node_border_widths.append(3.5)
        elif node == end_node and (highlight_edges or node in highlight_nodes):
            node_colors.append('#10B981')      # 목적지 달성: 에메랄드 그린
            node_sizes.append(40)
            node_borders.append('#064E3B')
            node_border_widths.append(3.5)
        elif node in highlight_nodes:
            node_colors.append('#3B82F6')      # 방문 완료 / 최단 경로: 파란색
            node_sizes.append(36)
            node_borders.append('#1E3A8A')
            node_border_widths.append(2.5)
        elif node in queue_or_stack_nodes:
            node_colors.append('#FCD34D')      # 큐/스택 대기 중: 밝은 노랑
            node_sizes.append(34)
            node_borders.append('#B45309')
            node_border_widths.append(2.5)
        else:
            node_colors.append('#FFFFFF')      # 미방문: 깔끔한 흰색
            node_sizes.append(32)
            node_borders.append('#64748B')
            node_border_widths.append(2)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="bottom center",
        textfont=dict(size=13, color='#0F172A', family='Pretendard, sans-serif'),
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(color=node_borders, width=node_border_widths)
        )
    )

    fig = go.Figure(
        data=[edge_trace, hl_edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=17, color='#1E293B', family='Pretendard, sans-serif')),
            showlegend=False,
            hovermode='closest',
            annotations=annotations,
            margin=dict(b=40, l=40, r=40, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#F8FAFC',
            paper_bgcolor='#FFFFFF',
            height=490
        )
    )
    return fig


# ------------------------------------------------------------------------------
# 7. 단일 스텝 뷰 렌더링 헬퍼 함수 (StreamlitDuplicateElementId 방지 고유 Key 적용)
# ------------------------------------------------------------------------------
def render_traversal_step(step, total_steps, algo_choice, graph, weighted_map, start_station, layout_mode, chart_key="t1_chart"):
    """DFS / BFS의 단일 스텝에 대한 상태 배지, 그래프, 자료구조 뷰를 렌더링"""
    badge_class = "badge-dfs" if "DFS" in algo_choice else "badge-bfs"
    badge_name = "DFS (스택 / LIFO)" if "DFS" in algo_choice else "BFS (큐 / FIFO)"
    ds_label = "📦 현재 스택 (Stack) 상태 [맨 뒤가 TOP]" if "DFS" in algo_choice else "🚶 현재 큐 (Queue) 상태 [맨 앞이 FRONT]"

    st.markdown(
        f'<span class="algo-badge {badge_class}">{badge_name}</span> <b>단계 {step["step_num"]} / {total_steps-1}</b>: '
        f'{step["action"]}',
        unsafe_allow_html=True
    )

    col_g, col_ds = st.columns([7, 5])

    with col_g:
        cur_node = step.get("current")
        vis_nodes = step.get("visited", [])
        ds_items = step.get("stack") if "DFS" in algo_choice else step.get("queue", [])
        hl_edges = step.get("highlight_edges", [])

        fig = create_graph_figure(
            graph=graph,
            weighted_map=weighted_map,
            highlight_nodes=vis_nodes,
            current_node=cur_node,
            highlight_edges=hl_edges,
            queue_or_stack_nodes=ds_items,
            start_node=start_station,
            title=f"{algo_choice} 탐색 진행도 (현재: {cur_node or '대기'})",
            layout_mode=layout_mode
        )
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

    with col_ds:
        st.markdown(f"#### {ds_label}")
        if ds_items:
            ds_str = " ➔ ".join([f"[{item}]" for item in ds_items])
            st.markdown(f'<div class="ds-box">{ds_str}</div>', unsafe_allow_html=True)
            if "DFS" in algo_choice:
                st.caption(f"👉 다음 꺼낼 역: **`{ds_items[-1]}`** (`stack.pop()` 실행)")
            else:
                st.caption(f"👉 다음 꺼낼 역: **`{ds_items[0]}`** (`queue.pop(0)` 실행)")
        else:
            st.markdown('<div class="ds-box">(비어 있음)</div>', unsafe_allow_html=True)

        st.markdown("#### 🏁 방문 순서 (Visited)")
        if vis_nodes:
            vis_str = " ➔ ".join([f"**{i+1}. {stn}**" for i, stn in enumerate(vis_nodes)])
            st.markdown(f'<div class="step-log">{vis_str}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="step-log">(아직 방문한 역 없음)</div>', unsafe_allow_html=True)


def render_dijkstra_step(step, total_steps, d_start, d_end, d_final_path, d_total_time, graph, weighted_map, stations, layout_mode, chart_key="t2_chart", df_key="t2_df"):
    """다익스트라 단일 스텝에 대한 상태 배지, 그래프, 테이블 뷰를 렌더링"""
    step_idx = step["step_num"]
    is_last = (step_idx == total_steps - 1)

    st.markdown(
        f'<span class="algo-badge badge-dijkstra">다익스트라</span> <b>단계 {step_idx} / {total_steps-1}</b>: '
        f'{step["action"]}',
        unsafe_allow_html=True
    )

    col_dg, col_dt = st.columns([7, 5])

    with col_dg:
        hl_path = d_final_path if is_last else step["visited"]
        hl_edges = [(d_final_path[i], d_final_path[i+1]) for i in range(len(d_final_path)-1)] if is_last and d_final_path else []

        fig_d = create_graph_figure(
            graph=graph,
            weighted_map=weighted_map,
            highlight_nodes=hl_path,
            current_node=step["current"],
            highlight_edges=hl_edges,
            queue_or_stack_nodes=step.get("updated_stations", []),
            start_node=d_start,
            end_node=d_end,
            title=f"다익스트라 최단 경로 탐색 ({d_start} ➔ {d_end})",
            layout_mode=layout_mode
        )
        st.plotly_chart(fig_d, use_container_width=True, key=chart_key)

    with col_dt:
        st.markdown("#### ⏱️ 역별 최소 소요 시간 (`times`) & 직전 역 (`previous`)")
        df_records = []
        for stn in stations:
            t_val = step["times"].get(stn, 999999)
            t_str = f"{t_val}분" if t_val < 999999 else "∞ (미도달)"
            prev_val = step["previous"].get(stn) or "-"
            status = "🟢 확정(방문완료)" if stn in step["visited"] else ("🟡 이번 단계 갱신" if stn in step["updated_stations"] else "⚪ 미확정")
            df_records.append({"지하철역": stn, "최소 시간": t_str, "직전 경유역": prev_val, "상태": status})

        df = pd.DataFrame(df_records)
        st.dataframe(df, hide_index=True, use_container_width=True, key=df_key)

        if is_last and d_final_path:
            st.success(f"🎯 **최종 최단 경로:** {' ➔ '.join(d_final_path)} (총 **{d_total_time}분** 소요)")
            st.info(f"💡 **역추적 원리:** 도착역 `{d_end}`에서부터 `previous` 딕셔너리를 거슬러 올라가 출발역 `{d_start}`까지 도달한 뒤 뒤집어(`reversed`) 경로를 구합니다.")


# ------------------------------------------------------------------------------
# 8. 메인 대시보드 레이아웃
# ------------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-title">🚇 지하철 노선망 탐색 알고리즘 시각화</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">DFS(깊이 우선), BFS(너비 우선), 다익스트라(최단 경로)의 작동 원리를 인터랙티브하게 체험하세요.</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 사이드바: 파이썬 코드 입력창 & 그래프 커스터마이징
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ 그래프 정의 & 레이아웃")
        
        layout_choice = st.selectbox(
            "📐 노선도 배치 알고리즘",
            [
                "🎯 카마다-카와이 (교차 최소화 & 균형 배치)",
                "🗺️ 서울 지하철 지리 배치 (실제 방위각 정렬)",
                "⚖️ 스프링 포스 (힘 지향 물리 분산)",
                "⭕ 원형 순환선 (Circular / Ring)"
            ],
            index=0,
            help="선이 꼬이지 않도록 그래프의 최적 위치를 계산하는 알고리즘을 선택합니다."
        )

        layout_map = {
            "🎯 카마다-카와이 (교차 최소화 & 균형 배치)": "kamada_kawai",
            "🗺️ 서울 지하철 지리 배치 (실제 방위각 정렬)": "geographic",
            "⚖️ 스프링 포스 (힘 지향 물리 분산)": "spring",
            "⭕ 원형 순환선 (Circular / Ring)": "circular"
        }
        selected_layout_mode = layout_map[layout_choice]

        st.divider()

        preset = st.selectbox(
            "📋 예제 프리셋 불러오기",
            ["기본 6개 역 (search.py)", "확장 8개 역 노선망 (신촌·홍대 추가)"]
        )

        preset_code = DEFAULT_PYTHON_CODE
        if preset == "확장 8개 역 노선망 (신촌·홍대 추가)":
            preset_code = '''# 확장 8개 역 노선망 (실제 순환/환승망 연결)
subway_graph = {
    "시청": ["신촌", "동대문", "용산", "강남"],
    "신촌": ["시청", "홍대"],
    "홍대": ["신촌", "신도림"],
    "신도림": ["홍대", "용산", "동대문"],
    "동대문": ["시청", "신도림", "강남", "잠실"],
    "용산": ["시청", "신도림", "강남"],
    "강남": ["시청", "동대문", "용산", "잠실"],
    "잠실": ["동대문", "강남"]
}

subway_weighted_map = {
    "시청": [("신촌", 7), ("동대문", 10), ("용산", 8), ("강남", 15)],
    "신촌": [("시청", 7), ("홍대", 5)],
    "홍대": [("신촌", 5), ("신도림", 6)],
    "신도림": [("홍대", 6), ("용산", 9), ("동대문", 14)],
    "동대문": [("시청", 10), ("신도림", 14), ("강남", 13), ("잠실", 11)],
    "용산": [("시청", 8), ("신도림", 9), ("강남", 9)],
    "강남": [("시청", 15), ("동대문", 13), ("용산", 9), ("잠실", 10)],
    "잠실": [("동대문", 11), ("강남", 10)]
}
'''

        code_input = st.text_area(
            "파이썬 코드 편집:",
            value=preset_code,
            height=300,
            help="subway_graph 와 subway_weighted_map 딕셔너리를 정의해주세요."
        )

        if st.button("🔄 그래프 다시 적용하기", use_container_width=True):
            st.rerun()

    # 파이썬 코드 파싱
    graph, weighted_map, error_msg = parse_python_graph_code(code_input)
    if error_msg:
        st.error(f"❌ {error_msg}")
        return

    stations = list(graph.keys())
    if not stations:
        st.warning("⚠️ 그래프에 등록된 역(노드)이 없습니다.")
        return

    # --------------------------------------------------------------------------
    # 5대 탭 메뉴 구성
    # --------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 [실습 1] DFS vs BFS 단계별 탐색",
        "⚡ [실습 2] 다익스트라 최단 경로",
        "⚖️ [비교] DFS vs BFS 동시 비교",
        "🚧 [시나리오] 공사·지연 우회 시뮬레이터",
        "📖 [학습] search.py 코드 및 원리"
    ])

    # ==========================================================================
    # TAB 1: DFS vs BFS 단계별 탐색 (스택 & 큐 내부 상태 시각화 & 애니메이션)
    # ==========================================================================
    with tab1:
        st.subheader("🔍 깊이 우선(DFS) & 너비 우선(BFS) 단계별 탐색 시뮬레이션")
        st.caption("자료구조(스택 LIFO vs 큐 FIFO)에 따라 방문 순서가 어떻게 달라지는지 애니메이션으로 확인해보세요.")

        col_ctrl1, col_ctrl2 = st.columns([3, 3])
        with col_ctrl1:
            algo_choice = st.radio("탐색 알고리즘 선택", ["DFS (깊이 우선 탐색)", "BFS (너비 우선 탐색)"], horizontal=True, key="t1_algo")
        with col_ctrl2:
            start_station = st.selectbox("출발역 선택", stations, index=0, key="t1_start")

        # 스텝 트레이스 생성
        if "DFS" in algo_choice:
            steps = trace_dfs(graph, start_station)
        else:
            steps = trace_bfs(graph, start_station)

        total_steps = len(steps)

        # 세션 상태 초기화 및 슬라이더 키 관리
        state_key = f"step_idx_t1_{algo_choice}_{start_station}"
        if state_key not in st.session_state:
            st.session_state[state_key] = 0

        # 애니메이션 컨트롤 바
        st.markdown("##### 🎬 탐색 애니메이션 컨트롤러")
        c_btn1, c_btn2, c_btn3, c_btn4, c_speed = st.columns([1.5, 1.2, 1.2, 1.3, 3.5])
        
        with c_btn1:
            play_clicked = st.button("▶️ 자동 재생", key="t1_play", use_container_width=True, type="primary")
        with c_btn2:
            if st.button("⏮️ 이전", key="t1_prev", use_container_width=True):
                st.session_state[state_key] = max(0, st.session_state[state_key] - 1)
        with c_btn3:
            if st.button("⏭️ 다음", key="t1_next", use_container_width=True):
                st.session_state[state_key] = min(total_steps - 1, st.session_state[state_key] + 1)
        with c_btn4:
            if st.button("🔄 처음으로", key="t1_reset", use_container_width=True):
                st.session_state[state_key] = 0
        with c_speed:
            anim_speed = st.select_slider(
                "⏱️ 재생 속도",
                options=[0.2, 0.4, 0.7, 1.0, 1.5, 2.0],
                value=0.7,
                format_func=lambda s: f"{s}초/단계",
                key="t1_speed"
            )

        # 수동 슬라이더
        slider_val = st.slider(
            "직접 단계 이동 (Step Slider)",
            0, total_steps - 1,
            value=st.session_state[state_key],
            key=f"slider_{state_key}"
        )
        st.session_state[state_key] = slider_val

        # 동적 렌더링 컨테이너
        display_placeholder = st.empty()

        # 자동 재생 루프 실행
        if play_clicked:
            start_from = st.session_state[state_key]
            if start_from >= total_steps - 1:
                start_from = 0

            prog_bar = st.progress(0.0)
            anim_timestamp = time.time()
            for idx in range(start_from, total_steps):
                st.session_state[state_key] = idx
                prog_bar.progress((idx + 1) / total_steps)
                with display_placeholder.container():
                    render_traversal_step(
                        step=steps[idx],
                        total_steps=total_steps,
                        algo_choice=algo_choice,
                        graph=graph,
                        weighted_map=weighted_map,
                        start_station=start_station,
                        layout_mode=selected_layout_mode,
                        chart_key=f"t1_chart_anim_{idx}_{anim_timestamp}"
                    )
                time.sleep(anim_speed)
            prog_bar.empty()
        else:
            with display_placeholder.container():
                render_traversal_step(
                    step=steps[st.session_state[state_key]],
                    total_steps=total_steps,
                    algo_choice=algo_choice,
                    graph=graph,
                    weighted_map=weighted_map,
                    start_station=start_station,
                    layout_mode=selected_layout_mode,
                    chart_key=f"t1_chart_static_{st.session_state[state_key]}"
                )

        # 핵심 포인트 교육 카드
        with st.expander("💡 핵심 탐색 원리 보기", expanded=False):
            if "DFS" in algo_choice:
                st.markdown("""
                - **LIFO (Last In First Out)**: 가장 최근에 스택에 넣은 이웃역을 바로 깊게 파고듭니다.
                - `stack.pop()`으로 맨 뒤 요소를 꺼냅니다.
                - 작은 번호나 특정 순서로 방문하기 위해 `reversed()`로 스택에 넣습니다.
                """)
            else:
                st.markdown("""
                - **FIFO (First In First Out)**: 가장 먼저 들어온 역부터 차례대로 방문하여 물결처럼 퍼져나갑니다.
                - `queue.pop(0)`으로 맨 앞 요소를 꺼냅니다.
                - 모든 간선의 가중치가 같을 때 최단 환승 경로를 보장합니다.
                """)

    # ==========================================================================
    # TAB 2: 다익스트라 최단 경로 시뮬레이터 (애니메이션 지원)
    # ==========================================================================
    with tab2:
        st.subheader("⚡ 다익스트라(Dijkstra) 최단 경로 시뮬레이션")
        st.caption("소요 시간(가중치)이 서로 다른 네트워크에서 가장 빠른 최단 경로와 소요 시간을 애니메이션으로 관찰하세요.")

        c_d1, c_d2 = st.columns([3, 3])
        with c_d1:
            d_start = st.selectbox("출발역", stations, index=0, key="t2_start")
        with c_d2:
            default_end_idx = 5 if len(stations) > 5 else len(stations)-1
            d_end = st.selectbox("도착역", stations, index=default_end_idx, key="t2_end")

        d_steps, d_final_path, d_total_time = trace_dijkstra(weighted_map, d_start, d_end)
        d_total_steps = len(d_steps)

        # 세션 상태 초기화
        d_state_key = f"step_idx_t2_{d_start}_{d_end}"
        if d_state_key not in st.session_state:
            st.session_state[d_state_key] = d_total_steps - 1

        st.markdown("##### 🎬 다익스트라 탐색 애니메이션 컨트롤러")
        cd_btn1, cd_btn2, cd_btn3, cd_btn4, cd_speed = st.columns([1.5, 1.2, 1.2, 1.3, 3.5])

        with cd_btn1:
            d_play_clicked = st.button("▶️ 자동 재생", key="t2_play", use_container_width=True, type="primary")
        with cd_btn2:
            if st.button("⏮️ 이전", key="t2_prev", use_container_width=True):
                st.session_state[d_state_key] = max(0, st.session_state[d_state_key] - 1)
        with cd_btn3:
            if st.button("⏭️ 다음", key="t2_next", use_container_width=True):
                st.session_state[d_state_key] = min(d_total_steps - 1, st.session_state[d_state_key] + 1)
        with cd_btn4:
            if st.button("🔄 처음으로", key="t2_reset", use_container_width=True):
                st.session_state[d_state_key] = 0
        with cd_speed:
            d_anim_speed = st.select_slider(
                "⏱️ 재생 속도",
                options=[0.2, 0.4, 0.7, 1.0, 1.5, 2.0],
                value=0.7,
                format_func=lambda s: f"{s}초/단계",
                key="t2_speed"
            )

        d_slider_val = st.slider(
            "직접 단계 이동 (Dijkstra Step Slider)",
            0, d_total_steps - 1,
            value=st.session_state[d_state_key],
            key=f"slider_{d_state_key}"
        )
        st.session_state[d_state_key] = d_slider_val

        d_display_placeholder = st.empty()

        if d_play_clicked:
            d_start_from = st.session_state[d_state_key]
            if d_start_from >= d_total_steps - 1:
                d_start_from = 0

            d_prog_bar = st.progress(0.0)
            d_anim_timestamp = time.time()
            for idx in range(d_start_from, d_total_steps):
                st.session_state[d_state_key] = idx
                d_prog_bar.progress((idx + 1) / d_total_steps)
                with d_display_placeholder.container():
                    render_dijkstra_step(
                        step=d_steps[idx],
                        total_steps=d_total_steps,
                        d_start=d_start,
                        d_end=d_end,
                        d_final_path=d_final_path,
                        d_total_time=d_total_time,
                        graph=graph,
                        weighted_map=weighted_map,
                        stations=stations,
                        layout_mode=selected_layout_mode,
                        chart_key=f"t2_chart_anim_{idx}_{d_anim_timestamp}",
                        df_key=f"t2_df_anim_{idx}_{d_anim_timestamp}"
                    )
                time.sleep(d_anim_speed)
            d_prog_bar.empty()
        else:
            with d_display_placeholder.container():
                render_dijkstra_step(
                    step=d_steps[st.session_state[d_state_key]],
                    total_steps=d_total_steps,
                    d_start=d_start,
                    d_end=d_end,
                    d_final_path=d_final_path,
                    d_total_time=d_total_time,
                    graph=graph,
                    weighted_map=weighted_map,
                    stations=stations,
                    layout_mode=selected_layout_mode,
                    chart_key=f"t2_chart_static_{st.session_state[d_state_key]}",
                    df_key=f"t2_df_static_{st.session_state[d_state_key]}"
                )

    # ==========================================================================
    # TAB 3: DFS vs BFS 동시 비교
    # ==========================================================================
    with tab3:
        st.subheader("⚖️ DFS(깊이 우선) vs BFS(너비 우선) 나란히 비교")
        st.caption("동일한 출발역에서 두 알고리즘이 지하철망을 어떤 순서로 탐색하는지 직접 대조해보세요.")

        cmp_start = st.selectbox("비교할 출발역 선택", stations, index=0, key="t3_start")

        dfs_trace = trace_dfs(graph, cmp_start)[-1]
        dfs_result = dfs_trace["visited"]
        dfs_edges = dfs_trace["highlight_edges"]

        bfs_trace = trace_bfs(graph, cmp_start)[-1]
        bfs_result = bfs_trace["visited"]
        bfs_edges = bfs_trace["highlight_edges"]

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown(f"### 🌲 DFS (깊이 우선 탐색)")
            st.markdown('<span class="algo-badge badge-dfs">자료구조: 스택 (LIFO)</span>', unsafe_allow_html=True)
            st.write(f"**방문 순서:** {' ➔ '.join(dfs_result)}")
            
            fig_dfs = create_graph_figure(
                graph=graph,
                weighted_map=weighted_map,
                highlight_nodes=dfs_result,
                highlight_edges=dfs_edges,
                start_node=cmp_start,
                title=f"DFS 탐색 경로 (깊게 전진)",
                layout_mode=selected_layout_mode
            )
            st.plotly_chart(fig_dfs, use_container_width=True, key="t3_dfs_chart")
            st.markdown("""
            **특징:**
            - 막다른 길이 나올 때까지 끝까지 파고든 후 되돌아옵니다 (Backtracking).
            - 미로 탈출이나 모든 경우의 수를 확인할 때 유용합니다.
            """)

        with col_c2:
            st.markdown(f"### 🌊 BFS (너비 우선 탐색)")
            st.markdown('<span class="algo-badge badge-bfs">자료구조: 큐 (FIFO)</span>', unsafe_allow_html=True)
            st.write(f"**방문 순서:** {' ➔ '.join(bfs_result)}")

            fig_bfs = create_graph_figure(
                graph=graph,
                weighted_map=weighted_map,
                highlight_nodes=bfs_result,
                highlight_edges=bfs_edges,
                start_node=cmp_start,
                title=f"BFS 탐색 경로 (넓게 번짐)",
                layout_mode=selected_layout_mode
            )
            st.plotly_chart(fig_bfs, use_container_width=True, key="t3_bfs_chart")
            st.markdown("""
            **특징:**
            - 출발역에서 1정거장 거리 ➔ 2정거장 거리 순으로 동심원을 그리며 퍼져나갑니다.
            - 가중치가 없는 그래프에서 **최소 환승/최소 거리**를 보장합니다.
            """)

    # ==========================================================================
    # TAB 4: 공사/지연 시뮬레이터 (What-if 분석)
    # ==========================================================================
    with tab4:
        st.subheader("🚧 실시간 지하철 공사 및 지연 우회 시나리오")
        st.caption("수업 활동지 과제: '동대문역 공사로 시간이 대폭 늘어나면 최단 경로는 어떻게 바뀔까?'")

        st.info("💡 아래 슬라이더로 특정 구간의 소요 시간을 늘리거나 줄여보세요. 다익스트라 최단 경로가 실시간으로 다른 노선으로 우회합니다.")

        all_edges = []
        for u, neighbors in weighted_map.items():
            for item in neighbors:
                if isinstance(item, (tuple, list)):
                    v, w = item[0], item[1]
                else:
                    v, w = item, 10
                if (v, u, w) not in all_edges and (u, v, w) not in all_edges:
                    all_edges.append((u, v, w))

        c_sc1, c_sc2 = st.columns([4, 8])

        with c_sc1:
            st.markdown("#### ⏱️ 구간별 소요 시간 조절")
            custom_weights = {}
            for u, v, w in all_edges:
                key = f"edge_{u}_{v}"
                new_w = st.slider(f"{u} ↔ {v}", min_value=1, max_value=60, value=w, key=key)
                custom_weights[(u, v)] = new_w
                custom_weights[(v, u)] = new_w

            mod_weighted_map = {}
            for u, neighbors in weighted_map.items():
                mod_neighbors = []
                for item in neighbors:
                    if isinstance(item, (tuple, list)):
                        v, orig_w = item[0], item[1]
                    else:
                        v, orig_w = item, 10
                    mod_neighbors.append((v, custom_weights.get((u, v), orig_w)))
                mod_weighted_map[u] = mod_neighbors

            sc_start = st.selectbox("출발역 선택", stations, index=0, key="sc_start")
            sc_end = st.selectbox("도착역 선택", stations, index=5 if len(stations)>5 else len(stations)-1, key="sc_end")

        with c_sc2:
            sc_steps, sc_path, sc_time = trace_dijkstra(mod_weighted_map, sc_start, sc_end)
            sc_edges = [(sc_path[i], sc_path[i+1]) for i in range(len(sc_path)-1)] if sc_path else []

            st.markdown(f"### 📍 결과: {' ➔ '.join(sc_path)} (총 **{sc_time}분**)")
            
            fig_sc = create_graph_figure(
                graph=graph,
                weighted_map=mod_weighted_map,
                highlight_nodes=sc_path,
                highlight_edges=sc_edges,
                start_node=sc_start,
                end_node=sc_end,
                title=f"우회 시뮬레이션 최단 경로 ({sc_start} ➔ {sc_end})",
                layout_mode=selected_layout_mode
            )
            st.plotly_chart(fig_sc, use_container_width=True, key="t4_scenario_chart")

    # ==========================================================================
    # TAB 5: 코드 및 개념 학습장
    # ==========================================================================
    with tab5:
        st.subheader("📖 search.py 소스 코드 및 핵심 알고리즘 원리")
        
        col_cd1, col_cd2 = st.columns(2)

        with col_cd1:
            st.markdown("#### 1. 자료구조 비교")
            st.markdown("""
            | 비교 항목 | 스택 (Stack) | 큐 (Queue) |
            | :--- | :--- | :--- |
            | **규칙** | LIFO (Last In First Out) | FIFO (First In First Out) |
            | **파이썬 코드** | `stack.pop()` (맨 뒤 추출) | `queue.pop(0)` (맨 앞 추출) |
            | **탐색 알고리즘** | **DFS** (깊이 우선) | **BFS** (너비 우선) |
            | **비유** | 프링글스 감자칩 통, 책 쌓기 | 은행 번호표 창구, 줄서기 |
            """)

            st.markdown("#### 2. 다익스트라 3단계 메커니즘")
            st.markdown("""
            1. **초기화**: 시작역 소요 시간 0분, 나머지 모든 역 무한대(999999분) 설정
            2. **선택**: 미방문 역 중 현재 소요 시간이 가장 작은 역 선택 및 확정
            3. **갱신**: `선택된 역 시간 + 이동 시간 < 기존 이웃 역 시간`이면 더 빠른 시간으로 업데이트하고 `previous` 기록
            """)

        with col_cd2:
            st.markdown("#### 💻 search.py 원본 코드")
            try:
                with open("search.py", "r", encoding="utf-8") as f:
                    code_content = f.read()
                st.code(code_content, language="python")
            except Exception as e:
                st.code(code_input, language="python")


if __name__ == "__main__":
    main()
