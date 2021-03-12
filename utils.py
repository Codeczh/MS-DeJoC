import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
# from model import *
from model_layer import *
from Resnet import *
# from Vgg import *
# from resnet_origine import ResNet_Origine
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def load_model(model_name, pretrain=True, require_grad=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net,50, 512, 200)
    elif model_name == 'resnet18_pmg':
        net = resnet18(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net,18, 512, 200)
    return net

def load_model_layer(model_name,classes_nums, pretrain=True, require_grad=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG_layer(net,50, 512, classes_nums)
    elif model_name == 'resnet18_pmg':
        net = resnet18(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG_layer(net,18, 512, classes_nums)
    elif model_name == 'vgg16_pmg':
        net = VGG16(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG_layer_16(net,16, 512, classes_nums)
        torchvision.models.vgg16()
    return net

def load_resnet_layer(model_name, classes_nums,pretrain=True, require_grad=True):
    net = ResNet_Origine(model_name,n_classes = classes_nums,pretrained=pretrain)
    for param in net.parameters():
        param.requires_grad = require_grad
    return net

def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)                # n*n , shuffle
    jigsaws = images.clone()
    for i in range(rounds):          # exchange [0,0]-----[x,y]
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws


def test(net, criterion, batch_size,device):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0

    transform_test = transforms.Compose([
        transforms.Scale((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    testset = torchvision.datasets.ImageFolder(root='./data/web-bird/val',
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)
    with torch.no_grad():
        # tqdm_gen = tqdm .tqdm(testloader)
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # if batch_idx >1 :
            #     break
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            output_1, output_2, output_3, output_concat= net(inputs)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

            # tqdm_gen.set_description(
            #     'Iteration: %d | Loss: %.3f |  Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
            #         batch_idx,  test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_com = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_com, test_loss


