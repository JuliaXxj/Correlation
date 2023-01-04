import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from datetime import datetime
import copy

# to get activation
ACTIVATION = None


def get_activation(name, tensor_logger, detach, is_lastlayer=False):
    if is_lastlayer:
        def hook(model, input, output):
            raw = torch.flatten(output, start_dim=1, end_dim=-1).cpu().detach().numpy()
            # use argmax instead of broadcasting just in case comparing floating point is finicky

            mask = np.zeros(raw.shape, dtype=bool)

            mask[np.arange(raw.shape[0]), raw.argmax(axis=1)] = 1

            tensor_logger[name] = np.concatenate((tensor_logger[name], mask),
                                                 axis=0) if name in tensor_logger else mask

        return hook

    if detach:
        def hook(model, input, output):
            raw = torch.flatten(
                output, start_dim=1, end_dim=-1).cpu().detach().numpy()
            raw = raw > 0
            logging.debug("{}, {}".format(name, raw.shape))
            tensor_logger[name] = np.concatenate((tensor_logger[name], raw),
                                                 axis=0) if name in tensor_logger else raw
            logging.debug(tensor_logger[name].shape)

        return hook
    else:
        # keep the gradient, so cannot convert to bit here
        def hook(model, input, output):
            raw = torch.sigmoid(torch.flatten(
                output, start_dim=1, end_dim=-1))
            logging.debug("{}, {}".format(name, raw.shape))
            tensor_logger[name] = torch.cat((tensor_logger[name], raw),
                                            axis=0) if name in tensor_logger else raw
            logging.debug(tensor_logger[name].shape)

        return hook


def get_gradient(name, gradient_logger, detach):
    def hook(model, grad_input, grad_output):
        raw = grad_output
        assert (len(raw) == 1)
        raw = raw[0].cpu().detach().numpy()
        gradient_logger[name] = np.concatenate((gradient_logger[name], raw), axis=0) if name in gradient_logger else raw

    return hook


def get_neurons(name, neuron_logger, detach, is_lastlayer=False):
    if is_lastlayer:
        def hook(model, input, output):
            raw = torch.flatten(output, start_dim=1, end_dim=-1).cpu().detach().numpy()
            # use argmax instead of broadcasting just in case comparing floating point is finicky

            mask = np.zeros(raw.shape, dtype=bool)

            mask[np.arange(raw.shape[0]), raw.argmax(axis=1)] = 1

            neuron_logger[name] = np.concatenate((neuron_logger[name], mask),
                                                 axis=0) if name in neuron_logger else mask

        return hook

    if detach:
        def hook(model, input, output):
            raw = torch.flatten(
                output, start_dim=1, end_dim=-1).cpu().detach().numpy()
            # raw = raw > 0
            logging.debug("{}, {}".format(name, raw.shape))
            neuron_logger[name] = np.concatenate((neuron_logger[name], raw),
                                                 axis=0) if name in neuron_logger else raw
            logging.debug(neuron_logger[name].shape)

        return hook
    else:
        # keep the gradient, so cannot convert to bit here
        def hook(model, input, output):
            raw = torch.sigmoid(torch.flatten(
                output, start_dim=1, end_dim=-1))
            logging.debug("{}, {}".format(name, raw.shape))
            neuron_logger[name] = torch.cat((neuron_logger[name], raw),
                                            axis=0) if name in neuron_logger else raw
            logging.debug(neuron_logger[name].shape)

        return hook


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.tensor_log = {}
        self.gradient_log = {}
        self.neuron_log = {}
        self.hooks = []
        self.bw_hooks = []
        self.n_hooks = []

    def reset_hooks(self):
        self.tensor_log = {}
        for h in self.hooks:
            h.remove()

    def reset_bw_hooks(self):
        self.input_labels = None
        self.gradient_log = {}
        for h in self.bw_hooks:
            h.remove()

    def reset_n_hooks(self):
        self.neuron_log = {}
        for h in self.n_hooks:
            h.remove()

    def register_log(self, detach):
        raise NotImplementedError

    def register_gradient(self, detach):
        raise NotImplementedError

    def register_neurons(self, detach):
        raise NotImplementedError

    def model_savename(self):
        raise NotImplementedError

    def get_pattern(self, input, layers, device, flatten=True):
        self.eval()
        self.register_log()
        self.forward(input.to(device))
        tensor_log = copy.deepcopy(self.tensor_log)
        if flatten:
            return np.concatenate([tensor_log[l] for l in layers], axis=1)
        return tensor_log

    def get_features(self, input, layers, device, flatten=True):
        self.eval()
        self.register_neurons()
        self.forward(input.to(device))
        neuron_log = copy.deepcopy(self.neuron_log)
        if flatten:
            return np.concatenate([neuron_log[l] for l in layers], axis=1)
        return neuron_log

class FeedforwardNeuralNetModel(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()

        # Linear function
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)

    def register_log(self, detach=True):
        self.reset_hooks()
        # first layer should not make any difference?
        self.hooks.append(self.fc1.register_forward_hook(get_activation('fc1', self.tensor_log, detach)))
        self.hooks.append(self.fc2.register_forward_hook(get_activation('fc2', self.tensor_log, detach)))
        self.hooks.append(self.fc3.register_forward_hook(get_activation('fc3', self.tensor_log, detach)))
        # self.hooks.append(self.fc4.register_forward_hook(get_activation('fc4', self.tensor_log, detach)))
        self.hooks.append(
            self.fc4.register_forward_hook(get_activation('fc4', self.tensor_log, detach, is_lastlayer=False)))

    def register_gradient(self, detach=True):
        self.reset_bw_hooks()
        # first layer should not make any difference?
        self.bw_hooks.append(self.fc1.register_backward_hook(get_gradient('fc1', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc2.register_backward_hook(get_gradient('fc2', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc3.register_backward_hook(get_gradient('fc3', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc4.register_backward_hook(get_gradient('fc4', self.gradient_log, detach)))

    def register_neurons(self, detach=True):
        self.reset_n_hooks()
        self.n_hooks.append(self.fc1.register_forward_hook(get_neurons('fc1', self.neuron_log, detach)))
        self.n_hooks.append(self.fc2.register_forward_hook(get_neurons('fc2', self.neuron_log, detach)))
        self.n_hooks.append(self.fc3.register_forward_hook(get_neurons('fc3', self.neuron_log, detach)))
        self.n_hooks.append(self.fc4.register_forward_hook(get_neurons('fc4', self.neuron_log, detach, is_lastlayer=False)))

    def forward(self, x):
        out = F.relu(self.fc1(x.view(-1, 28 * 28)))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        #    out = F.log_softmax(out, dim=1)
        return out

    def model_savename(self):
        return "FFN" + datetime.now().strftime("%H-%M-%S")