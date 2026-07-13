import torch

@torch.no_grad()
def get_leaves(model, loader, device):
    model.eval()
    leaves = []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        leaves += model.eval_forward(inputs, return_leaves=True).tolist()

    return leaves 

def get_leaf_stats(leaves, n_leaves) -> list[float]:
    s = [leaves.count(i) for i in range(n_leaves)]
    stats = [leaves.count(i)/len(leaves) for i in range(n_leaves)]
    return stats

@torch.no_grad()
def get_forest_leaves(model, loader, device):
    model.eval()
    leaves = [[] for n in range(model.n_trees)]
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        all_leaves = model.eval_forward(inputs, return_leaves=True)
        for n in range(model.n_trees):
            leaves[n] += all_leaves[:, n].tolist()
    return leaves 

def get_forest_leaf_stats(leaves, n_leaves) -> list[float]:
    forest_stats = []
    for tree_leaves in leaves:
        s = [tree_leaves.count(i) for i in range(n_leaves)]
        stats = [tree_leaves.count(i)/len(tree_leaves) for i in range(n_leaves)]
        forest_stats.append(stats)
    return forest_stats

def get_parition_count(mem1: int, mem2: int, depth: int, leaf_size: int, router_size: int):
    import math
    n_leaves =  2 ** depth
    n_routers = n_leaves - 1
    mem1_remain = mem1 - n_routers * router_size
    mem1_leaves = math.floor(mem1_remain / leaf_size)
    return mem1_leaves



