from helper.utils import load_yaml, pkl_and_write
from helper.args import get_command_line_args, replace_args_with_dict_values, get_command_line_args_datasets
from helper.noisy import NoiseAda
import torch
from helper.train_utils import train, test, get_optimizer, seed_everything, s_train, batch_train, batch_test
from models.nn import get_model
import numpy as np
import time
import logging
# print("OK")
import torch.nn.functional as F
from copy import deepcopy
from helper.data import get_dataset
import os.path as osp
import optuna
from helper.hyper_search import hyper_search
import sys
from tqdm import tqdm
# from helper.utils import delete_non_tensor_attributes
# from ogb.nodeproppred import Evaluator

def train_pipeline(seeds, args, epoch, data, need_train, need_save_logits, reliability_list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_result_acc = []
    early_stop_accum = 0
    val_result_acc = []
    out_res = []
    best_val = 0
    debug_accs = []
    train_accs = []
    num_of_classes = data.y.max().item() + 1                                                      #数据集的节点类别数量
    if args.model_name == 'S_model':
        noise_ada = NoiseAda(num_of_classes).to(device)                                           #添加噪声
    else:
        noise_ada = None
    for i, seed in enumerate(seeds):                                                              #一个seed是一次随机生成训练集、节点选择、伪标签、过滤、训练过程
        if len(reliability_list) > 0:                                                             #若有大模型置信度数据
            reliability = reliability_list[0].to(device)
        seed_everything(seed)
        model = get_model(args).to(device)                                                        #训练模型，GCN、GAT等
        optimizer, scheduler = get_optimizer(args, model)                                         #1优化器更新参数，adam、radam等，2调度器更新学习率
        if args.loss_type == 'ce':
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)             #交叉熵损失，判定实际的输出与期望的输出的接近程度
        else:
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction='none')     #参数reduction在[none, mean, sum]中选，string型。none表示不降维，返回和target相同形状；mean表示对一个batch的损失求均值；sum表示对一个batch的损失求和
        if args.normalize:
            data.x = F.normalize(data.x, dim = -1)
        data = data.to(device)
        data.train_mask = data.train_masks[i]              #这样改变了data？以后seed用的data都是同一个？
        data.val_mask = data.val_masks[i]
        data.test_mask = data.test_masks[i]
        debug_acc = []
        this_train_acc = []
        if 'ft' in args.data_format and 'no_ft' not in args.data_format:
            data.x = data.xs[i]
            data.train_mask = data.train_masks[i]
            data.val_mask = data.val_masks[i]
            data.test_mask = data.test_masks[i]
        if 'pl' in args.split or 'active' in args.split:   #所提伪标签pl和主动选择active，则进入分支
            data.train_mask = data.train_masks[i]
            data.val_mask = data.val_masks[i]
            data.test_mask = data.test_masks[i]
            data.backup_y = data.y.clone()
            if not args.debug_gt_label:                    #debug_gt_label=0，用伪标签监督训练
                data.y = data.ys[i]
            else:
                print("Using ground truth label")

            # import ipdb; ipdb.set_trace()
        for i in tqdm(range(epoch)):                                                                        #tqdm是一个快速、可扩展的Python进度条库，用于在Python长循环中添加一个进度提示信息，用户只需封装任何迭代器 tqdm(iterator)
            # ipdb.set_trace()
            train_mask = data.train_mask
            val_mask = data.val_mask
            if need_train:                                                                                  #训练，若训练模型是LP则不进入
                if 'rim' in args.strategy or 'iterative' in args.strategy or args.split == 'active_train':  #节点选择策略为rim或iterative，或数据集掩码操作采用所提的节点主动选择+置信度后过滤
                    train_loss, val_loss, val_acc, train_acc = train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val, reliability)
                else:
                    if args.model_name == 'S_model':
                        train_loss, val_loss, val_acc, train_acc = s_train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val, noise_ada)
                    else:
                        train_loss, val_loss, val_acc, train_acc = train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val)
                if scheduler:
                    scheduler.step()
                if args.output_intermediate and not args.no_val:
                    print(f"Epoch {i}: Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_acc[0]}")
                if args.debug:
                    if args.filter_strategy == 'none':
                        test_acc, res = test(model, data, 0, data.test_mask)
                    else:
                        test_acc, res = test(model, data, 0, data.test_mask, data.backup_y)   #测试，返回测试集acc指标值
                    # print(f"Epoch {i}: Test acc: {test_acc}")
                    debug_acc.append(test_acc)
                    this_train_acc.append(train_acc)
                if not args.no_val:
                    if val_acc > best_val:
                        best_val = val_acc
                        best_model = deepcopy(model)
                        early_stop_accum = 0
                    else:
                        if i >= args.early_stop_start:
                            early_stop_accum += 1
                        if early_stop_accum > args.early_stopping and i >= args.early_stop_start:
                            print(f"Early stopping at epoch {i}")
                            break
            else:
                best_model = model
        if 'pl' in args.split or 'active' in args.split:
            data.y = data.backup_y
        if args.no_val or best_model == None:  #若设置验证集，则可训练多个模型取最好的
            best_model = model
        test_acc, res = test(best_model, data, args.return_embeds, data.test_mask)
        test_result_acc.append(test_acc)      #保存各seed的最终测试集acc
        val_result_acc.append(best_val)       #保存各seed训练中所有epoch中的最好验证集acc
        out_res.append(res)                   #保存各seed的最终模型输出
        best_val = 0
        best_model = None
        if args.debug:
            debug_accs.append(debug_acc)      #保存各seed训练中所有epoch的测试集acc
            train_accs.append(this_train_acc) #保存各seed训练中所有epoch的训练集acc
        if need_save_logits:
            torch.save(out_res, f'../output/logits/{args.dataset}_{args.split}_{args.model_name}_{seed}_logits.pt')
    if not args.debug:
        return test_result_acc, val_result_acc, out_res
    else:
        return test_result_acc, val_result_acc, out_res, debug_accs, train_accs


def main(data_path, args = None, custom_args = None, save_best = False):
    seeds = [i for i in range(args.main_seed_num)]                                    #随机种子，随机到某次实验的训练过程
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')             #作用是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行
    if custom_args != None:
        args = replace_args_with_dict_values(args, custom_args)
    vars(args)['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    params_dict = load_yaml(args.yaml_path)
    logit_path = params_dict['LOGIT_PATH']
    reliability_list = []
    data = get_dataset(seeds, args.dataset, args.split, args.data_format, data_path, logit_path, args.pl_noise, args.no_val, args.budget, args.strategy, args.num_centers, args.compensation, args.save_data, args.filter_strategy, args.max_part, args.oracle, reliability_list, args.total_budget, args.second_filter, True, False, args.filter_all_wrong_labels, args.alpha, args.beta, args.gamma, args.ratio).to(device)
    print("_______\n",data,"\n___________\n\n")
    epoch = args.epochs                                                               #训练轮次
    vars(args)['input_dim'] = data.x.shape[1]                                         #输入的维数
    vars(args)['num_classes'] = data.y.max().item() + 1
    if args.model_name == 'LP':
        need_train = False                                                            #若被训练模型为LP，则
    else:
        need_train = True
    if not args.batchify and args.ensemble_string == "":                              #设置意义方便读者调试，默认参数下这句内容是执行的 - 即训练GCN
        data.x = data.x.to(torch.float32)
        if not args.debug:
            test_result_acc, _, _ = train_pipeline(seeds, args, epoch, data, need_train, args.save_logits, reliability_list)
        else:                                                                         #debug！=0，训练、调试、测试整套流程
            test_result_acc, _, _, debug_accs, train_accs = train_pipeline(seeds, args, epoch, data, need_train, args.save_logits, reliability_list)
        mean_test_acc = np.mean(test_result_acc) * 100   #对各seed的最终测试集acc取平均，【二】用于计算训练效果指标Test Accuracy ————————————————————
        std_test_acc = np.std(test_result_acc) * 100     #对各seed的最终测试集acc取标准差
        if args.debug:
            best_possible_test_acc = [np.max(res) for res in debug_accs]  #对各seed，取各自训练中所有epoch的测试集acc的最大值,【三】表示模型潜在能力Best possible accuracy ————————————————————
        res_train_accs = [x[-1] for x in train_accs]                      #对各seed，取各自训练中最后一个epoch的训练集acc，【一】用于计算训练效果指标Train Accuracy ————————————————————
        print(f"Train Accuracy: {np.mean(res_train_accs) * 100:.2f} ± {np.std(res_train_accs) * 100:.2f}")
        print(f"Test Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
        if args.debug:
            print(f"Best possible accuracy: {np.mean(best_possible_test_acc) * 100:.2f} ± {np.std(best_possible_test_acc) * 100:.2f}")
        print("Test acc: {}".format(test_result_acc))
    elif args.ensemble_string != "":
        pass
    else:
        pass
    if save_best:
        pkl_and_write(args, osp.join("./bestargs", f"{args.model_name}_{args.dataset}_{args.data_format}.pkl"))
    if args.debug:
        if args.debug_gt_label:
            pkl_and_write(debug_accs, osp.join("./debug", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_gt.pkl"))
            pkl_and_write(train_accs, osp.join("./debug_train", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_train_accs_gt.pkl"))
        elif args.filter_all_wrong_labels:
            pkl_and_write(debug_accs, osp.join("./debug", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_filtered.pkl"))
            pkl_and_write(train_accs, osp.join("./debug_train", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_train_accs_filtered.pkl"))
        else:
            pkl_and_write(debug_accs, osp.join("./debug", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}.pkl"))
            pkl_and_write(train_accs, osp.join("./debug_train", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_train_accs.pkl"))


def max_trial_callback(study, trial, max_try):
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE or t.state == optuna.trial.TrialState.RUNNING])
    n_total_complete = len([t for t in study.trials])
    if n_complete >= max_try or n_total_complete >= 2 * max_try:
        study.stop()
        torch.cuda.empty_cache()


def sweep(data_path, args = None):
    # test_seeds = [i for i in range(args.seed_num)]
    sweep_seeds = [i for i in range(args.sweep_seed_num)]
    ## get default command line args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vars(args)['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = f"{args.dataset}_{args.model_name}_{args.data_format}_{args.split}"
    study = optuna.create_study(study_name=study_name, storage=None, direction='maximize', load_if_exists=True)
    param_f = hyper_search
    sweep_round = args.sweep_round
    study.optimize(lambda trial: sweep_run(trial, args, sweep_seeds, param_f, device, data_path), catch=(RuntimeError,), n_trials=sweep_round, callbacks=[lambda study, trial: max_trial_callback(study, trial, sweep_round)], show_progress_bar=True, gc_after_trial=True)
    main(args=args, custom_args = study.best_trial.params, save_best = True)
    print(study.best_trial.params)


def sweep_run(trial, args, sweep_seeds, param_f, device, data_path):
    params = param_f(trial, args.data_format, args.model_name, args.dataset)    
    args = replace_args_with_dict_values(args, params)
    params_dict = load_yaml(args.yaml_path)
    logit_path = params_dict['LOGIT_PATH']
    reliability_list = []
    data = get_dataset(sweep_seeds, args.dataset, args.split, args.data_format, data_path, logit_path, args.pl_noise, args.no_val, args.budget, args.strategy, args.num_centers, args.compensation, args.save_data, args.filter_strategy, args.max_part, args.oracle, reliability_list, args.total_budget, args.second_filter, True, False, args.filter_all_wrong_labels, args.alpha, args.beta, args.gamma, args.ratio).to(device)
    epoch = args.epochs
    vars(args)['input_dim'] = data.x.shape[1]
    vars(args)['num_classes'] = data.y.max().item() + 1
    if args.model_name == 'LP':
        need_train = False
    else:
        need_train = True
    if not args.batchify and args.ensemble_string == "":
        data.x = data.x.to(torch.float32)
        test_result_acc, _, _ = train_pipeline(sweep_seeds, args, epoch, data, need_train, args.save_logits, reliability_list)
    elif args.ensemble_string != "":
        pass
    else:
        pass
    mean_test_acc = np.mean(test_result_acc)
    std_test_acc = np.std(test_result_acc)
    print(f"Test Accuracy: {mean_test_acc} ± {std_test_acc}")
    return mean_test_acc

if __name__ == '__main__':
    current_time = int(time.time())
    # #logging.basicConfig(filename='../../logs/{}.log'.format(current_time),
    #                 filemode='a',
    #                 format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    #                 datefmt='%H:%M:%S',
    #                 level=logging.INFO)

    print("Start")

    # args = get_command_line_args()            #获取参数
    args = get_command_line_args_datasets()
    params_dict = load_yaml(args.yaml_path)
    data_path = params_dict['DATA_PATH']
    if args.mode == "main":
        main(data_path, args = args)          #训练
    else:
        sweep(data_path, args = args)


'''
运行cora例子 ：
python src/main.py --dataset cora --model_name GCN --data_format sbert --main_seed_num 3 --split active --output_intermediate 0 --no_val 1 --strategy pagerank2 --debug 1 --total_budget 140 --filter_strategy consistency --loss_type ce --second_filter conf+entropy --epochs 30 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0.2 --alpha 0.33 --beta 0.33
'''
