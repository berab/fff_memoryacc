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
