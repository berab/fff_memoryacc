import torch
from tqdm import tqdm

# TODO: Loss for maximizing sample entropy or minimizing class entropy
def train_epoch(model, optim, loader, criterion, epoch, device):
    model.train()
    correct, running_loss = 0, 0.0
    for i, (inputs, targets) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)


        # back propagation
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # other stats
        running_loss += loss.item()
        correct += (preds == targets).sum().item()

    return running_loss/len(loader), correct/len(loader.dataset)

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
