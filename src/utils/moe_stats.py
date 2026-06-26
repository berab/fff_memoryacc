import torch

@torch.no_grad()
def get_experts(model, loader, device):
    model.eval()
    experts = []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        experts += model.eval_forward(inputs, return_experts=True).tolist()

    return experts 

def get_expert_stats(experts, n_experts) -> list[float]:
    s = [experts.count(i) for i in range(n_experts)]
    stats = [experts.count(i)/len(experts) for i in range(n_experts)]
    return stats

@torch.no_grad()
def get_forest_experts(model, loader, device):
    model.eval()
    experts = [[] for n in range(model.n_trees)]
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        all_experts = model.eval_forward(inputs, return_experts=True)
        for n in range(model.n_trees):
            experts[n] += all_experts[:, n].tolist()
    return experts 

def get_forest_expert_stats(experts, n_experts) -> list[float]:
    forest_stats = []
    for tree_experts in experts:
        s = [tree_experts.count(i) for i in range(n_experts)]
        stats = [tree_experts.count(i)/len(tree_experts) for i in range(n_experts)]
        forest_stats.append(stats)
    return forest_stats
