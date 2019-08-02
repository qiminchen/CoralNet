import torch


class MetricUtil:
    def __init__(self):
        self.running_loss = 0.0
        self.running_accuracy = 0.0
        self.pred = []
        self.gt = []

    def compute_metric(self, model, phase, inputs, labels,
                       criterion, optimizer, nbatch):
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, predicts = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # backward and optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()
            elif phase == 'valid':
                self.pred += predicts
                self.gt += labels.data
        # statistics
        self.running_loss += loss.item()
        self.running_accuracy += torch.sum(predicts == labels.data)
        return (self.running_loss / nbatch,
                self.running_accuracy.item() / (nbatch * inputs.size(0)))
