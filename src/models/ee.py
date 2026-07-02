import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyExit(nn.Module):
    def __init__(self, in_features, width, out_features, exit_th=0.85):
        super().__init__()
        self.width = width
        self.exit_th = exit_th
        self.in_features = in_features
        self.out_features = out_features
        
        # --- Backbone Layer 1 ---
        self.fc1 = nn.Linear(in_features, width)
        
        # --- Early Exit 1 ---
        self.early_fc1 = nn.Linear(width, out_features)
        
        # --- Remaining Backbone (Layers 2 & 3) ---
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, out_features)
        self.relu = nn.ReLU()

    def get_config(self) -> dict:
        conf = {'width': self.width,
                "exit_th": self.exit_th,
                }
        return conf

    def forward(self, x):
        # Returns output, True if early-exited
        # 1. Run through the 1st layer
        x = x.view(len(x), -1)
        x = self.relu(self.fc1(x))
        out = torch.empty((len(x), self.out_features), device=x.device)
        
        # 2. Check the Early Exit
        early_out = self.early_fc1(x)
        early_probs = F.softmax(early_out, dim=-1)
        max_prob, _ = torch.max(early_probs, dim=-1)
        
        # If confident and not training, skip the rest of the network
        early_exits = max_prob > self.exit_th
        out[early_exits] = early_out[early_exits]
        
        # 3. If not confident, continue through the rest of the deep network
        x = self.relu(self.fc2(x))
        final_out = self.fc3(x)
        out[~early_exits] = final_out[~early_exits]
        return out, early_exits.sum()

    # Inside your model class, update the forward pass for training:
    def forward_v2(self, x):
        # Layer 1
        x = self.relu(self.fc1(x))
        early_out = self.early_fc1(x)
        
        # ALWAYS compute the rest during training, even if confident
        if not self.training:
            early_probs = F.softmax(early_out, dim=-1)
            max_prob, _ = torch.max(early_probs, dim=-1)
            if max_prob.item() > self.exit_th:
                return early_out, True # Exited early
                
        # Layers 2 & 3
        x = self.relu(self.fc2(x))
        final_out = self.fc3(x)
        
        if not self.training:
            return final_out, False
        
        # Return both during training
        return early_out, final_out
