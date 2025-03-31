import torch
from ogb.nodeproppred import Evaluator
from torch.optim.lr_scheduler import  _LRScheduler
from torch_geometric.utils import index_to_mask, subgraph
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class WarmupExpLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, gamma=0.1, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.gamma = gamma
        super(WarmupExpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]
    
    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** self.last_epoch
                for base_lr in self.base_lrs]

    



def get_optimizer(args, model):
    if args.model_name == 'LP':
        return None, None
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
        scheduler = None 
    elif args.optim == 'radam':
        optimizer = torch.optim.RAdam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay) #优化器（根据计算得到的梯度信息来更新模型的参数，以最小化损失函数），PyTorch里有：SGD、SGD+Momentum、AdaGrad、RMSProp、Adam
        scheduler = WarmupExpLR(optimizer, args.warmup, total_epochs=args.epochs, gamma=args.lr_gamma)  #调度器（调整优化过程中使用的特定参数，通常是学习率）
    return optimizer, scheduler


def s_train(model, data, optimizer, loss_fn, train_mask, val_mask, no_val, noise_ada):
    optimizer.zero_grad()
    preds = model(data)
    if len(data.y.shape) != 1:
        y = data.y.squeeze(1)
    else:
        y = data.y
    pred = F.softmax(preds,dim=1)
    eps = 1e-8
    score = noise_ada(preds).clamp(eps,1-eps)
    loss_train = F.cross_entropy(torch.log(score[train_mask]), y[train_mask])
    loss_train.backward()
    optimizer.step()
    train_acc, _ = test(model, data, False, train_mask)
    if not no_val:
        val_loss = loss_fn(preds[val_mask], y[val_mask])
        val_acc, _ = test(model, data, False, val_mask)
    else:
        val_loss = 0
        val_acc = 0
    return loss_train, val_loss, val_acc, train_acc


def train(model, data, optimizer, loss_fn, labeled_weight, labeled_mask, train_mask, val_mask, no_val, reliability_list = None):
    optimizer.zero_grad()         #【一】将梯度归零
    preds = model(data)           #【二】调用模型预测
    if len(data.y.shape) != 1:
        y = data.y.squeeze(1)     #移除原张量中大小为1的维度，data.y为真实标签
    else:
        y = data.y
    if len(data.yss.shape) != 1:
        ys = data.yss.squeeze(1)  #data.ys为节点伪标签
    else:
        ys = data.yss
    confidence = reliability_list
    unlabeled_weight = 1.0 - labeled_weight  # 和为 1：简化调参过程，避免因总损失规模变化带来的干扰. 若无标签数据质量高（伪标签可靠），可增大unlabeled_weight（labeled_weight动态减小）
    # print("preds: ", len(preds), preds, "\ntruth: ", len(y[train_mask]),y[train_mask])
    # n_classes = preds.size(1)
    # min_label = torch.min(y[labeled_mask]).item()
    # max_label = torch.max(y[labeled_mask]).item()
    # if min_label < 0 or max_label >= n_classes:
    #     print(f"目标标签范围错误：最小值为 {min_label}，最大值为 {max_label}，类别数为 {n_classes}")

    labeled_loss = loss_fn(preds[labeled_mask], y[labeled_mask])
    if loss_fn.reduction != 'none' or confidence == None:
        unlabeled_loss = loss_fn(preds[train_mask], ys[train_mask])
        # print("l:", labeled_loss.item(), "ul:",train_loss.item())
    else:
        # Extract the values using the mask
        values_to_normalize = confidence[train_mask]   #筛选节点（即通过了主动选择和后过滤）的置信度
        # Compute min and max of these values
        min_val = torch.min(values_to_normalize)
        max_val = torch.max(values_to_normalize)
        if min_val == max_val:
            # normalized_values = confidence
            pass
        # Apply Min-Max scaling
        else:
            normalized_values = (values_to_normalize - min_val) / (max_val - min_val)
            # Replace original tensor values with the normalized values for nodes defined by the train mask     #归一化，向量操作
            confidence[train_mask] = normalized_values.clone()
        # train_loss = (loss_fn(preds[train_mask], ys[train_mask]) * confidence[train_mask]).mean()  #训练集损失为所有筛选节点的损失（预测和伪标签的交叉熵）和置信度乘积的平均值
        unlabeled_loss = (loss_fn(preds[train_mask], ys[train_mask]) * confidence[train_mask]).mean()
    train_loss = labeled_weight * labeled_loss + unlabeled_weight * unlabeled_loss
    train_loss.backward()         #【三】反向传播计算得到每个参数的梯度值
    optimizer.step()              #【四】通过梯度下降执行一步参数更新
    # train_acc, _ = test(model, data, False, train_mask)  #eval()，计算acc
    train_acc, _ = test(model, data, False, train_mask | labeled_mask)
    if not no_val:
        val_loss = loss_fn(preds[val_mask], ys[val_mask])
        val_acc, _ = test(model, data, False, val_mask)
    else:
        val_loss = 0
        val_acc = 0
    return train_loss, val_loss, val_acc, train_acc


def batch_train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch_size, n_id, edge_index = batch.batch_size, batch.n_id, batch.edge_index
        optimizer.zero_grad()
        batch.edge_index = batch.edge_index.to(device)
        out = model(batch)[:batch_size]
        y = batch.y[:batch_size].squeeze()
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def to_inductive(data, msk_index = 0):
    data = data.clone()
    mask = data.train_masks[msk_index]
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = mask[mask]
    data.test_masks = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data



@torch.no_grad()
def batch_test(model, data, evaluator, subgraph_loader, device, mask):
    model.eval()

    out = model.inference(data.x, subgraph_loader, device)

    y_pred = out.argmax(dim=-1, keepdim=True)

    # import ipdb; ipdb.set_trace()
    if len(data.y.shape) == 1:
        y_true = data.y.unsqueeze(dim=1)  # for non ogb datas
    else:
        y_true = data.y

    test_acc = evaluator.eval({
        'y_true': y_true[mask],
        'y_pred': y_pred[mask]
    })['acc']

    return test_acc



@torch.no_grad()
def topk_test(model, data, mask, topk = 3, need_batch = False, subgraph_loader = None, device='cuda'):
    model.eval()
    # model.model.initialized = False
    if not need_batch:
        out = model(data)
        y_pred = out.argmax(dim=-1, keepdim=True)
    else:
        out = model.inference(data.x, subgraph_loader, device)
        y_true = data.y
        y_pred = out.argmax(dim=-1, keepdim=True)
    r_y_pred = y_pred.reshape(-1)
    confidence = out.gather(1, r_y_pred.unsqueeze(1)).reshape(-1)
    data.confidence = confidence
    sorted_conf_idx = torch.argsort(data.confidence)
    full_length = data.x.shape[0]
    com_res = data.y.view(-1, 1).expand_as(out.topk(3,1).values).eq(out.topk(3,1).indices).sum(-1).to(torch.bool)
    low_confidence_sorted_conf_mask = index_to_mask(sorted_conf_idx[:full_length // 3], size=full_length)
    med_confidence_sorted_conf_mask = index_to_mask(sorted_conf_idx[full_length // 3 : full_length * 2 // 3], size=full_length)
    high_confidence_sorted_conf_mask = index_to_mask(sorted_conf_idx[full_length * 2 // 3:], size=full_length)

    y_1 = y_pred.reshape(-1)
    true_mask = (y_1 == data.y)
    false_mask = ~true_mask

    evaluator = Evaluator(name='ogbn-arxiv')
    top3_low_acc = torch.sum(com_res[mask & low_confidence_sorted_conf_mask]) / com_res[mask & low_confidence_sorted_conf_mask].shape[0]
    top3_med_acc = torch.sum(com_res[mask & med_confidence_sorted_conf_mask]) / com_res[mask & med_confidence_sorted_conf_mask].shape[0]
    top3_high_acc = torch.sum(com_res[mask & high_confidence_sorted_conf_mask]) / com_res[mask & high_confidence_sorted_conf_mask].shape[0]
    # true_acc = torch.sum(com_res[mask & true_mask]) / com_res[mask & true_mask].shape[0]
    
    res = data.y.view(-1).eq(r_y_pred)
    top1_low_acc = torch.sum(res[mask & low_confidence_sorted_conf_mask]) / res[mask & low_confidence_sorted_conf_mask].shape[0]
    top1_med_acc = torch.sum(res[mask & med_confidence_sorted_conf_mask]) / res[mask & med_confidence_sorted_conf_mask].shape[0]
    top1_high_acc = torch.sum(res[mask & high_confidence_sorted_conf_mask]) / res[mask & high_confidence_sorted_conf_mask].shape[0]
    # top1_low_acc = torch.sum()
    top3_false_acc = torch.sum(com_res[mask & false_mask]) / com_res[mask & false_mask].shape[0]
    total_acc = torch.sum(com_res[mask]) / com_res[mask].shape[0]
    print("Top3 Accuracy on low confidence nodes: {}\n".format(top3_low_acc.item()))
    print("Top3 Accuracy on medium confidence nodes: {}\n".format(top3_med_acc.item()))
    print("Top3 Accuracy on high confidence nodes: {}\n".format(top3_high_acc.item()))
    print("Top1 Accuracy on low confidence nodes: {}\n".format(top1_low_acc.item()))
    print("Top1 Accuracy on medium confidence nodes: {}\n".format(top1_med_acc.item()))
    print("Top1 Accuracy on high confidence nodes: {}\n".format(top1_high_acc.item()))
    print("Top3 Accuracy on gnn false nodes: {}\n".format(top3_false_acc.item()))
    return top3_low_acc.item(), top3_med_acc.item(), top3_high_acc.item(), total_acc.item()



@torch.no_grad()
def confidence_test(model, data, mask):
    model.eval()
    # model.model.initialized = False
    out = model(data)
    y_pred = out.argmax(dim=-1, keepdim=True)
    r_y_pred = y_pred.reshape(-1)
    confidence = out.gather(1, r_y_pred.unsqueeze(1)).reshape(-1)
    data.confidence = confidence
    sorted_conf_idx = torch.argsort(data.confidence)
    full_length = data.x.shape[0]
    low_confidence_sorted_conf_mask = index_to_mask(sorted_conf_idx[:full_length // 3], size=full_length)
    med_confidence_sorted_conf_mask = index_to_mask(sorted_conf_idx[full_length // 3 : full_length * 2 // 3], size=full_length)
    high_confidence_sorted_conf_mask = index_to_mask(sorted_conf_idx[full_length * 2 // 3:], size=full_length)
    # ground_truth = data.y.cpu()
    # true_mask = data.y.cpu() == y_pred.cpu()
    # false_mask = data.y.cpu() != y_pred.cpu()

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1)  # for non ogb datas
    else:
        y = data.y
    
    y_1 = y_pred.reshape(-1)
    true_mask = (y_1 == data.y)
    false_mask = ~true_mask

    evaluator = Evaluator(name='ogbn-arxiv')
    low_acc = evaluator.eval({
        'y_true': y[mask | low_confidence_sorted_conf_mask],
        'y_pred': y_pred[mask | low_confidence_sorted_conf_mask],
    })['acc']
    
    med_acc = evaluator.eval({
        'y_true': y[mask | med_confidence_sorted_conf_mask],
        'y_pred': y_pred[mask | med_confidence_sorted_conf_mask],
    })['acc']

    high_acc = evaluator.eval({
        'y_true': y[mask | high_confidence_sorted_conf_mask],
        'y_pred': y_pred[mask | high_confidence_sorted_conf_mask],
    })['acc']


    true_acc = evaluator.eval({
        'y_true': y[mask | true_mask],
        'y_pred': y_pred[mask | true_mask],
    })['acc']


    false_acc = evaluator.eval({
        'y_true': y[mask | false_mask],
        'y_pred': y_pred[mask | false_mask],
    })['acc']

    print(true_acc, false_acc)

    return low_acc, med_acc, high_acc

@torch.no_grad()
def test(model, data, return_embeds, mask, gt_y = None):
    model.eval()             #训练完 train 样本后，生成的模型 model 要用来测试样本了。在 model(test) 之前，需要加上model.eval()，否则的话，有输入数据，即使不训练，它也会改变权值。这是 model 中含有 BN 层和 Dropout 所带来的的性质。eval() 时，pytorch 会自动把 BN 和 DropOut 固定住，不会取平均，而是用训练好的值。eval() 在非训练的时候是需要加的，没有这句代码，一些网络层的值会发生变动，不会固定，你神经网络每一次生成的结果也是不固定的，生成质量可能好也可能不好。
    # model.model.initialized = False
    out = model(data)        #使用训练好模型对data里的所有节点预测，若data是train则是看拟合程度，若data是test则是看效果
    y_pred = out.argmax(dim=-1, keepdim=True)
    
    if gt_y != None:
        if len(gt_y.shape) == 1:
            y = gt_y.unsqueeze(dim=1)  # for non ogb datas
        else:
            y = gt_y
    else:
        if len(data.y.shape) == 1:
            y = data.y.unsqueeze(dim=1)
        else:
            y = data.y

    evaluator = Evaluator(name='ogbn-arxiv') #使用OGB开源库的模型性能评估工具
    acc = evaluator.eval({
        'y_true': y[mask],
        'y_pred': y_pred[mask],
    })['acc']

    
    if not return_embeds:
        return acc, None
    else:
        return acc, out  #返回acc、对data的预测


def loss_kd(all_out, teacher_all_out, outputs, labels, teacher_outputs,
            alpha, temperature):
    """
    loss function for Knowledge Distillation (KD)
    """

    T = temperature

    loss_CE = F.cross_entropy(outputs, labels)
    D_KL = nn.KLDivLoss()(F.log_softmax(all_out / T, dim=1),
                          F.softmax(teacher_all_out / T, dim=1)) * (T * T)
    KD_loss =  (1. - alpha) * loss_CE + alpha * D_KL

    return KD_loss

def loss_kd_only(all_out, teacher_all_out, temperature):
    T = temperature

    D_KL = nn.KLDivLoss()(F.log_softmax(all_out / T, dim=1),
                          F.softmax(teacher_all_out / T, dim=1)) * (T * T)

    return D_KL



def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    #这里使用了多种随机数生成器，python random模块、NumPy随机模块、PyTorch随机数种子
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)    #设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。使得每次运行该 .py 文件时生成的随机数相同。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def graph_consistency(neighbors, gt):
    single_label_consistency = {}
    for key, value in neighbors.items():
        consistency = 0
        total_nei = len(value)
        center_y = gt[key]
        ## key: nodes
        ## value: neighbors
        for nei in value:
            nei_y = gt[nei]
            if nei_y == center_y:
                consistency += 1
        if total_nei != 0:
            single_label_consistency[key] = consistency / total_nei
        else:
            single_label_consistency[key] = 0
    sorted_keys = sorted(single_label_consistency, key=single_label_consistency.get)
    return sorted_keys


    # single_label_consistency = torch.tensor(single_label_consistency)
    


            
def calibration_plot(predicted_probs, preds, true_labels, output_name, number_of_bins=20):
    # Create bins for x-axis
    bin_size = 1.0 / number_of_bins
    bins = np.linspace(0, 1, number_of_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    true_proportions = []
    
    # Bin the predicted probabilities
    bin_assignments = np.digitize(predicted_probs, bins) - 1

    bin_width = 1.0 / number_of_bins

    true_or_false = (preds == true_labels).float()

    # For each bin, compute the actual accuracy
    bin_true_probs = np.array([true_or_false[bin_assignments == i].mean().item() for i in range(number_of_bins)])
    bin_true_probs[np.isnan(bin_true_probs)] = 0
    # plt.bar(bin_centers, bin_true_probs, width=bin_width, align='center', alpha=0.6, label="Model Accuracy")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    # plt.plot(bin_centers, true_proportions, label="Model", marker='o')
    plt.bar(bin_centers, bin_true_probs, width=bin_width, align='center', alpha=0.6, label="Model Accuracy")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    # plt.legend()
    # plt.title("Calibration Plot")
    # plt.grid(True)
    plt.savefig(output_name + '.pdf')
    plt.savefig(output_name + '.png')








            