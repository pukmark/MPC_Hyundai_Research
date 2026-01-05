from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class _State:
    time: float
    node: int
    soc: float
    parent: int | None


def find_fastest_feasible_path(
    Tmat: Iterable[Iterable[float]],
    Emat: Iterable[Iterable[float]],
    start: int,
    goal: int,
    soc_init: float,
    soc_min: float,
) -> Tuple[List[int], float, List[float]]:
    """
    Dijkstra's algorithm with a running SOC constraint.

    Args:
        Tmat: Square matrix of travel times where Tmat[i][j] is time from i to j.
        Emat: Square matrix of SOC deltas where Emat[i][j] is the change in SOC
            applied when moving i -> j (positive to gain, negative to consume).
        start: Index of the start node (0-based).
        goal: Index of the destination node (0-based).
        soc_init: Initial SOC at the start node.
        soc_min: Minimum SOC that must be maintained at every node.

    Returns:
        path: List of node indices from start to goal (inclusive).
        total_time: Total travel time for the returned path.
        soc_trace: SOC upon arrival at each node in the path.

    Raises:
        ValueError: If matrix shapes mismatch, indices are out of range, or no
            feasible path satisfies the SOC constraint.
    """
    T = np.asarray(Tmat, dtype=float)
    E = np.asarray(Emat, dtype=float)

    if T.shape != E.shape or T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("Tmat and Emat must be square matrices of the same shape")

    n = T.shape[0]
    if not (0 <= start < n and 0 <= goal < n):
        raise ValueError("start and goal must be valid node indices")

    # Priority queue of (accumulated time, state_id) keyed by minimum total time.
    states: List[_State] = []
    init_soc = min(float(soc_init), 100.0)
    states.append(_State(time=0.0, node=start, soc=init_soc, parent=None))
    pq: List[Tuple[float, int]] = [(0.0, 0)]

    # For each node, keep labels that are not dominated by both lower time and higher SOC.
    labels_by_node: list[list[int]] = [[] for _ in range(n)]
    labels_by_node[start].append(0)

    while pq:
        current_time, state_id = heapq.heappop(pq)
        state = states[state_id]

        # Skip if this entry is stale.
        if current_time != state.time:
            continue

        if state.node == goal:
            path, soc_trace = _reconstruct(states, state_id)
            return path, state.time, soc_trace

        for neighbor in range(n):
            if neighbor == state.node:
                continue

            travel_time = T[state.node, neighbor]
            if not np.isfinite(travel_time):
                continue

            delta_soc = E[state.node, neighbor]
            next_soc = min(state.soc + float(delta_soc), 100.0)
            if next_soc < soc_min:
                continue

            next_time = state.time + float(travel_time)
            candidate = _State(
                time=next_time, node=neighbor, soc=next_soc, parent=state_id
            )

            if _is_dominated(candidate, labels_by_node[neighbor], states):
                continue

            state_index = len(states)
            states.append(candidate)
            labels_by_node[neighbor] = _prune_dominated(
                candidate, labels_by_node[neighbor], states
            )
            labels_by_node[neighbor].append(state_index)
            heapq.heappush(pq, (next_time, state_index))

    raise ValueError("No feasible path found that satisfies the SOC constraint")


def find_fastest_with_charging(
    Tmat: Iterable[Iterable[float]],
    Emat: Iterable[Iterable[float]],
    start: int,
    goal: int,
    soc_init: float,
    soc_min: float,
    charging_nodes: Iterable[int],
    soc_reset: float = 100.0,
) -> Tuple[List[int], float, List[float]]:
    """
    Try direct path; if infeasible, route via a single charging stop (SOC resets).
    """
    # Attempt direct path first.
    try:
        return find_fastest_feasible_path(Tmat, Emat, start, goal, soc_init, soc_min)
    except ValueError:
        pass

    T = np.asarray(Tmat, dtype=float)
    n = T.shape[0]
    chargers = sorted(set(charging_nodes))
    for c in chargers:
        if not (0 <= c < n):
            raise ValueError(f"Charging node index out of range: {c}")

    best = None  # (total_time, combined_path, combined_soc)

    for charger in chargers:
        # Skip trivial cases where charger equals start or goal; direct already tried.
        if charger in (start, goal):
            continue
        try:
            path_a, time_a, soc_a = find_fastest_feasible_path(
                Tmat, Emat, start, charger, soc_init, soc_min
            )
            path_b, time_b, soc_b = find_fastest_feasible_path(
                Tmat, Emat, charger, goal, soc_reset, soc_min
            )
        except ValueError:
            continue

        total_time = time_a + time_b
        combined_path = path_a + path_b[1:]
        combined_soc = soc_a + soc_b[1:]

        if best is None or total_time < best[0]:
            best = (total_time, combined_path, combined_soc)

    if best is None:
        raise ValueError("No feasible path found (direct or via charging nodes)")

    return best[1], best[0], best[2]


def _is_dominated(candidate: _State, label_ids: list[int], states: List[_State]) -> bool:
    """Return True if any existing label has <= time and >= SOC."""
    for idx in label_ids:
        label = states[idx]
        if label.time <= candidate.time and label.soc >= candidate.soc:
            return True
    return False


def _prune_dominated(
    candidate: _State, label_ids: list[int], states: List[_State]
) -> list[int]:
    """Remove labels dominated by the candidate to keep the frontier compact."""
    pruned: list[int] = []
    for idx in label_ids:
        label = states[idx]
        if not (candidate.time <= label.time and candidate.soc >= label.soc):
            pruned.append(idx)
    return pruned


def _reconstruct(states: List[_State], state_id: int) -> Tuple[List[int], List[float]]:
    path: List[int] = []
    soc_trace: List[float] = []
    cursor: int | None = state_id
    while cursor is not None:
        state = states[cursor]
        path.append(state.node)
        soc_trace.append(state.soc)
        cursor = state.parent

    path.reverse()
    soc_trace.reverse()
    return path, soc_trace
