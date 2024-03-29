import numpy as np
import heapq
import torch
from functools import partial
from comb_utils import get_neighbourhood_func
from collections import namedtuple

DijkstraOutput = namedtuple("DijkstraOutput", ["shortest_path", "is_unique", "transitions"])

# -----------------------------------------------------------------------------

class HammingLoss(torch.nn.Module):
    def forward(self, suggested, target):
        errors = suggested * (1.0 - target) + (1.0 - suggested) * target
        return errors.mean(dim=0).sum()
        # return (torch.mean(suggested*(1.0-target)) + torch.mean((1.0-suggested)*target)) * 25.0

# -----------------------------------------------------------------------------

def dijkstra(matrix, neighbourhood_fn="8-grid", request_transitions=False):

    x_max, y_max = matrix.shape
    neighbors_func = partial(get_neighbourhood_func(neighbourhood_fn), x_max=x_max, y_max=y_max)

    costs = np.full_like(matrix, 1.0e10)
    costs[0][0] = matrix[0][0]
    num_path = np.zeros_like(matrix)
    num_path[0][0] = 1
    priority_queue = [(matrix[0][0], (0, 0))]
    certain = set()
    transitions = dict()

    while priority_queue:
        cur_cost, (cur_x, cur_y) = heapq.heappop(priority_queue)
        if (cur_x, cur_y) in certain:
            pass

        for x, y in neighbors_func(cur_x, cur_y):
            if (x, y) not in certain:
                if matrix[x][y] + costs[cur_x][cur_y] < costs[x][y]:
                    costs[x][y] = matrix[x][y] + costs[cur_x][cur_y]
                    heapq.heappush(priority_queue, (costs[x][y], (x, y)))
                    transitions[(x, y)] = (cur_x, cur_y)
                    num_path[x, y] = num_path[cur_x, cur_y]
                elif matrix[x][y] + costs[cur_x][cur_y] == costs[x][y]:
                    num_path[x, y] += 1

        certain.add((cur_x, cur_y))
    # retrieve the path
    cur_x, cur_y = x_max - 1, y_max - 1
    on_path = np.zeros_like(matrix)
    on_path[-1][-1] = 1
    while (cur_x, cur_y) != (0, 0):
        cur_x, cur_y = transitions[(cur_x, cur_y)]
        on_path[cur_x, cur_y] = 1.0

    is_unique = num_path[-1, -1] == 1

    if request_transitions:
        return DijkstraOutput(shortest_path=on_path, is_unique=is_unique, transitions=transitions)
    else:
        return DijkstraOutput(shortest_path=on_path, is_unique=is_unique, transitions=None)

# -----------------------------------------------------------------------------

def get_solver(neighbourhood_fn):

    def solver(matrix):
        return dijkstra(matrix, neighbourhood_fn).shortest_path

    return solver

# -----------------------------------------------------------------------------

class ShortestPath(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, lambda_val, neighbourhood_fn="8-grid"):

        ctx.save_for_backward(input)
        ctx.neighbourhood_fn = neighbourhood_fn
        ctx.lambda_val = lambda_val

        ctx.weights = input.detach().cpu().numpy() # detach weights
        
        # predict shortest path:
        suggested_tours = [dijkstra(wts, ctx.neighbourhood_fn).shortest_path for wts in list(ctx.weights)]
        ctx.suggested_tours = np.asarray(suggested_tours)
        
        return torch.from_numpy(ctx.suggested_tours).float().to(input.device)

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors

        lambda_val = ctx.lambda_val
        neighbourhood_fn = ctx.neighbourhood_fn
        suggested_tours = ctx.suggested_tours
        
        assert grad_output.shape == ctx.suggested_tours.shape
        grad_output_numpy = grad_output.detach().cpu().numpy()

        # smooth outputs:
        weights_prime = np.maximum(ctx.weights + lambda_val * grad_output_numpy, 0.0)

        # re-predict shortest path:
        better_paths = [dijkstra(wts, ctx.neighbourhood_fn).shortest_path for wts in list(weights_prime)]
        better_paths = np.asarray(better_paths)

        # smooth loss:
        gradient = -(suggested_tours - better_paths) / lambda_val

        return torch.from_numpy(gradient).to(grad_output.device), None, None
