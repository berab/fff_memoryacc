import torch
from tqdm import tqdm
from utils.prob import get_exp, get_halfnormal

# TODO: Loss for maximizing sample entropy or minimizing class entropy
def train_epoch(model, optim, loader, criterion, epoch, device, reg_alpha: float = 0.0, 
                entropy_alpha: float = 0.0, dist_reg = None, dist_alpha = 0.0, n_mem1 = 1):
    model.train()
    correct, dist_loss, running_loss, running_reg_loss, running_entropy_loss, running_dist_loss = 0, 0.0, 0.0, 0.0, 0.0, 0.0
    n_leaves = model.n_leaves
    if dist_reg == "exp":
        prob_dist = get_exp(n_mem1, n_leaves)
        prob_dist = prob_dist.to(device)
    elif dist_reg == "halfnormal":
        prob_dist = get_halfnormal(n_mem1, n_leaves)
        prob_dist = prob_dist.to(device)
    else:
        prob_dist = None

    for _, (inputs, targets) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, mixtures, entropies = model(inputs)

        if prob_dist != None:
            leaf_dist, _ = mixtures.mean(dim=0).sort(descending=True)
            dist_loss = (leaf_dist - prob_dist).abs().sum()
        reg_loss = (1 / mixtures.std(dim=1)).mean()
        entropy_loss = entropies.mean()


        # back propagation
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, targets) + reg_alpha * reg_loss + entropy_alpha * entropy_loss + dist_loss * dist_alpha
        optim.zero_grad()
        loss.backward()
        optim.step()

        # other stats
        running_loss += criterion(outputs, targets).item()
        running_reg_loss += reg_loss.item()
        running_entropy_loss += entropy_loss.item()
        correct += (preds == targets).sum().item()

    return (running_loss/len(loader), correct/len(loader.dataset), running_reg_loss/len(loader), 
            running_entropy_loss/len(loader), running_dist_loss/len(loader))

# TODO: Loss for maximizing sample entropy or minimizing class entropy
def ia_train_epoch(model, optim, loader, criterion, epoch, device, reg_alpha: float = 0.0, 
                   entropy_alpha: float = 0.0, a_alpha: float = 0.0):
    model.train()
    correct, running_loss, running_reg_loss, running_entropy_loss = 0, 0.0, 0.0, 0.0
    for i, (inputs, targets) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, mixtures, entropies = model(inputs, a_alpha)

        reg_loss = (1 / mixtures.std(dim=1)).mean()
        entropy_loss = entropies.mean()

        # back propagation
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, targets) + reg_alpha * reg_loss + entropy_alpha * entropy_loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        # other stats
        running_loss += loss.item()
        running_reg_loss += reg_loss.item()
        running_entropy_loss += entropy_loss.item()
        correct += (preds == targets).sum().item()

    return (running_loss/len(loader), correct/len(loader.dataset), running_reg_loss/len(loader), 
            running_entropy_loss/len(loader))

# TODO: Loss for maximizing sample entropy or minimizing class entropy
def moe_train_epoch(model, optim, loader, criterion, epoch, device, reg_alpha: float = 0.0, 
                    entropy_alpha: float = 0.0):
    model.train()
    correct, running_loss = 0, 0.0
    running_reg_loss, running_entropy_loss = 0.0, 0.0
    for i, (inputs, targets) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, probs, entropies = model(inputs)

        reg_loss = (1 / probs.std(dim=1)).mean()
        entropy_loss = entropies.mean()

        # back propagation
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, targets) + reg_alpha * reg_loss + entropy_alpha * entropy_loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        # other stats
        running_loss += loss.item()
        running_reg_loss += reg_loss.item()
        correct += (preds == targets).sum().item()

    return (running_loss/len(loader), correct/len(loader.dataset), running_reg_loss/len(loader), 
            running_entropy_loss/len(loader))

@torch.no_grad()
def eval_model(model, loader, criterion, device):
    model.eval()
    correct, running_loss = 0, 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        # stats
        _, preds = torch.max(outputs.data, 1)
        running_loss += criterion(outputs, targets).item()
        correct += (preds == targets).sum().item()

    return running_loss/len(loader), correct/len(loader.dataset), 

def train_epoch_ff(model, optim, loader, criterion, epoch, device):
    model.train()
    correct, running_loss = 0, 0.0

    for i, (inputs, targets) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        # back propagation
        loss = criterion(outputs, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # stats
        _, preds = torch.max(outputs.data, 1)
        running_loss += loss.item()
        correct += (preds == targets).sum().item()

    return running_loss/len(loader), correct/len(loader.dataset)

@torch.no_grad()
def eval_model_ff(model, loader, criterion, device):
    model.eval()
    correct, running_loss = 0, 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        # stats
        _, preds = torch.max(outputs.data, 1)
        running_loss += criterion(outputs, targets).item()
        correct += (preds == targets).sum().item()
    return running_loss/len(loader), correct/len(loader.dataset)

# TODO: Loss for maximizing sample entropy or minimizing class entropy
def train_epoch_rf(model, optim, loader, criterion, epoch, device, n: int, reg_alpha: float = 0.0):
    model.train()
    correct, running_loss, running_reg_loss = 0, 0.0, 0.0
    for i, (inputs, targets) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, mixtures = model(inputs, n)

        reg_loss = 1 / mixtures.std(dim=1).mean()

        # back propagation
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, targets) + reg_alpha * reg_loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        # other stats
        running_loss += loss.item()
        running_reg_loss += reg_loss.item()
        correct += (preds == targets).sum().item()

    return running_loss/len(loader), correct/len(loader.dataset), running_reg_loss/len(loader)

@torch.no_grad()
def eval_model_rf(model, loader, criterion, device):
    model.eval()
    correct, running_loss = 0, 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs).mean(dim=1)

        # stats
        _, preds = torch.max(outputs.data, 1)
        running_loss += criterion(outputs, targets).item()
        correct += (preds == targets).sum().item()

    return running_loss/len(loader), correct/len(loader.dataset), 

# TODO: Loss for maximizing sample entropy or minimizing class entropy
def train_epoch_ee(model, optim, loader, criterion, epoch, device):
    model.train()
    correct, running_loss, total_early = 0, 0.0, []
    for i, (inputs, targets) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, early = model(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # other stats
        total_early.append(early)
        running_loss += loss.item()
        correct += (preds == targets).sum().item()

    return running_loss/len(loader), correct/len(loader.dataset), sum(total_early)/len(loader.dataset)

@torch.no_grad()
def eval_model_ee(model, loader, criterion, device):
    model.eval()
    correct, running_loss, total_early = 0, 0.0, []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, early = model(inputs)

        # stats
        _, preds = torch.max(outputs.data, 1)
        running_loss += criterion(outputs, targets).item()
        correct += (preds == targets).sum().item()
        total_early.append(early)

    return running_loss/len(loader), correct/len(loader.dataset), sum(total_early)/len(loader.dataset)


# TODO: Loss for maximizing sample entropy or minimizing class entropy
def train_epoch_ee_v2(model, optim, loader, criterion, epoch, device):
    model.train()
    correct, running_loss, total_early = 0, 0.0, []
    for i, (inputs, targets) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, early = model.forward_v2(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # other stats
        total_early.append(early.item())
        running_loss += loss.item()
        correct += (preds == targets).sum().item()

    return running_loss/len(loader), correct/len(loader.dataset), sum(total_early)/len(loader.dataset)
