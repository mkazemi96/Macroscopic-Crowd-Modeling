from heapq import heappush, heappop
from ..tools import tqdm


def dijkstra(start, isgoal, *, verbose=False):
	frontier = [start]
	explored = set()
	bar = tqdm(desc="dijkstra", total=1, mininterval=1)
	while frontier:
		bar.update()
		node = heappop(frontier)
		if node.idx in explored:
			continue  # nodes can be pushed into the frontier multiple times, and are ignored if found again
		explored.add(node.idx)
		
		if verbose:
			print(
				f"{node.idx=:6}, "
				f"{node.cost=:7.4f}, "
				# f"{node.cost_to_go=:7.4f}, {node.cost_to_goal=:7.4f}, "  # just A* things
				f"{node.time=:7.4f}, {node.position=!s:>26}"
			)
		
		if isgoal(node):
			if verbose:
				print(f"Solution found. {len(frontier)=}, {len(explored)=}")
			bar.close()
			return node
		
		for child in node.children():
			if child.idx not in explored:
				heappush(frontier, child)
				bar.total += 1
	else:
		raise ValueError("Emptied frontier without finding solution!")
