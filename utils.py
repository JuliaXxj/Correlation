import shutil
# import imageio.v2 as imageio
import imageio
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import logging
import copy
import torch
import torch.nn.functional as F

def generate_gif(gif_filename, img_folder, image_names, delete_folder=False):
    """
    Generate the gif of the process of how the image gets transformed from all the images image_names saved in img_folder.
    Save the generated gif file with name gif_filename.
    Delete the img_folder if delete_folder is True.
    """
    with imageio.get_writer(gif_filename, mode='I') as writer:
        for img_name in image_names:
            image = imageio.imread(os.path.join(img_folder, img_name))
            writer.append_data(image)

    if delete_folder:
        shutil.rmtree(img_folder)


def cifar10_imshow(img, img_title=None, save_img=False, img_path=None):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    figure = plt.figure()
    plt.title("Pred prob: {}".format(img_title))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if save_img:
        plt.savefig(img_path)
    plt.show()


def cifar10_img(img):
    """
    Transform the image from cifar10 to a form that can be inputted for matplotlib.
    """
    img = torchvision.utils.make_grid(img)
    # unnormalize the image, see the transform applied to img in train_cifar10.py.
    img = img / 2 + 0.5
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))


def get_grad_each_label(gradient_log, target_log, layers, labels) -> dict:
    """
    return a dictionary of (k,v), in which k is one lable, v is a numpy array of all gradient for that particular label k
    """
    for l in layers:
        logging.debug(gradient_log[l].shape)
    res = {}
    for label in labels:
        res[label] = copy.deepcopy(np.concatenate([gradient_log[l][target_log == label] for l in layers],
                                                  axis=1))
        logging.debug(res[label].shape)

    return res


def test(model, device, test_loader, trace=False, detach=True):
    model.eval()
    test_loss = 0
    correct = 0
    if trace:
        model.register_log(detach)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            # sum up batch loss
            test_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if not trace:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    tensor_log = copy.deepcopy(model.tensor_log)

    model.reset_hooks()
    return tensor_log