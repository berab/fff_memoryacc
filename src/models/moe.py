import torch 
import math
from torch import nn

def compute_entropy_safe(p: torch.Tensor, minus_p: torch.Tensor) -> torch.Tensor:
	EPSILON = 1e-6
	p = torch.clamp(p, min=EPSILON, max=1-EPSILON)
	minus_p = torch.clamp(minus_p, min=EPSILON, max=1-EPSILON)

	return -p * torch.log(p+EPSILON) - minus_p * torch.log(minus_p+EPSILON)

class MoE(nn.Module):
    def __init__(self, in_features: int, expert_width: int, out_features: int, n_experts: int):
        super().__init__()
        self.in_features = in_features
        self.expert_width = expert_width
        self.n_experts = n_experts
        self.out_features = out_features
        self.activation = nn.ReLU()

        self.router = nn.Linear(in_features, n_experts)
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, expert_width),
                nn.ReLU(),
                nn.Linear(expert_width, out_features),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor):
        if self.training:
            return self.training_forward(x)
        else:
            return self.eval_forward(x)

    def training_forward(self, x: torch.Tensor):
        x = x.view(len(x), -1)
        route = torch.softmax(self.router(x), dim=-1)
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )
        out = (route.unsqueeze(-1) * expert_outputs).sum(dim=1)
        entropies = compute_entropy_safe(route, 1-route)
        return out, route, entropies

    def eval_forward(self, x: torch.Tensor, return_experts: bool = False) -> torch.Tensor:
        x = x.view(len(x), -1)
        idx = self.router(x).argmax(dim=-1)
        if return_experts:
            return idx

        out = torch.empty(x.size(0), self.out_features, device=x.device)
        for i, expert in enumerate(self.experts):
            mask = idx == i
            if mask.any():
                out[mask] = expert(x[mask])
        return out

    def get_config(self):
        return {
            'expert_width': self.expert_width,
            'n_experts': self.n_experts,
        }

    def get_element_memory_kb(self) -> dict:
        """
        Calculates the parameter memory usage of a single router node 
        and a single leaf node in the FFF model, returning the values in kilobytes.
        """
        # PyTorch float32 elements use 4 bytes of memory
        bytes_per_param = 4 

        # --- 1. Single Router Memory ---
        router_params = self.in_features * self.n_experts + self.n_experts
        router_kb = (router_params * bytes_per_param) / 1024

        # --- 2. Single Expert Memory ---
        w1_params = self.in_features * self.expert_width
        b1_params = self.expert_width
        w2_params = self.expert_width * self.out_features
        b2_params = self.out_features

        expert_params = w1_params + b1_params + w2_params + b2_params
        expert_kb = (expert_params * bytes_per_param) / 1024
        return  round(router_kb, 4), round(expert_kb, 4)
