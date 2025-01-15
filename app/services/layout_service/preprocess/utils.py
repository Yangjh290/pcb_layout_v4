from queue import Queue


def find_point(target_point: tuple[float, float], wires:  list[tuple[tuple[float, float], tuple[float, float]]]):
    """找到和当前点相连的线"""
    results: list = []
    for wire in wires:
        if target_point in wire:
            results.append(wire)

    if len(results) == 0:
        return None
    else:
        return results


def add_wires_to_queue(point: tuple[float, float], queue: Queue, wires: list[tuple[tuple[float, float], tuple[float, float]]]):
    """将线添加到队列中"""
    if wires is None:
        return
    wire_set =set()
    for wire in wires:
        wire_set.add(wire[0])
        wire_set.add(wire[1])

    wire_set.remove(point)
    for item in wire_set:
        queue.put(item)
