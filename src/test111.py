import transformers
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
import re
import tqdm
import json
from openail.utils import efficient_openai_text_api, set_endpoints, openai_text_api, openai_text_api_with_top_p, load_partial_openai_result, save_partial_openai_result, retrieve_dict, compute_ece, plot_calibration_curve, openai_text_api_with_backoff, num_tokens_from_string
from helper.data import get_dataset, inject_random_noise_y_level
from helper.args import get_command_line_args, get_command_line_args_datasets, replace_args_with_dict_values
from helper.active import train_lr, inference_lr
from helper.utils import load_yaml, pkl_and_write
from openail.config import configs
from config2 import configs2
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import ast
from openail.utils import load_mapping
import ipdb
import os.path as osp
import editdistance   #editdistance 是一个由 CSDN公司开发的InsCode AI大模型提及的高效实现编辑距离（Levenshtein距离）的开源项目，主要提供快速计算两个序列之间的Levenshtein距离的能力，这是一种衡量两个字符串相似度的方法，即通过插入、删除、替换操作将一个序列转换成另一个序列所需的最少操作数
from collections import Counter
import random
import re
import string
import numpy as np
from models.nn import LinearRegression
from helper.utils import noise_transition_matrix
import seaborn as sns
import ipdb
from helper.train_utils import calibration_plot
import pandas as pd
import sys
from helper.noisy import NoiseAda
import torch
from helper.train_utils import train, test, get_optimizer, seed_everything, s_train, batch_train, batch_test
from models.nn import get_model
import logging
import torch.nn.functional as F
from copy import deepcopy
from helper.data import get_dataset
import optuna
from helper.hyper_search import hyper_search
import sys
from tqdm import tqdm
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


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
        model = get_model(args).to(device)                                                        #训练模型，GCN、GAT、SAGE等
        optimizer, scheduler = get_optimizer(args, model)                                         #1优化器更新参数，adam、radam等，2调度器更新学习率
        if args.loss_type == 'ce':
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)             #交叉熵损失，判定实际的输出与期望的输出的接近程度
        else:
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction='none')     #参数reduction在[none, mean, sum]中选，string型。none表示不降维，返回和target相同形状；mean表示对一个batch的损失求均值；sum表示对一个batch的损失求和
        if args.normalize:
            data.x = F.normalize(data.x, dim = -1)
        data = data.to(device)
        data.labeled_mask = data.labeled_masks[i]
        data.train_mask = data.train_masks[i]             #取出第i个seed的数据集
        data.val_mask = data.val_masks[i]
        data.test_mask = data.test_masks[i]
        debug_acc = []
        this_train_acc = []
        if 'ft' in args.data_format and 'no_ft' not in args.data_format:
            data.x = data.xs[i]
            data.train_mask = data.train_masks[i]
            data.val_mask = data.val_masks[i]
            data.test_mask = data.test_masks[i]
        if 'pl' in args.split or 'active' in args.split:
            data.labeled_mask = data.labeled_masks[i]
            data.train_mask = data.train_masks[i]
            data.val_mask = data.val_masks[i]
            data.test_mask = data.test_masks[i]
            # twoset = data.train_mask == data.train_masks[i]  #分支上面这三句没必要吧？
            # print("if dataset:", torch.sum(twoset).item())
            data.backup_y = data.y.clone()
            data.yss = data.ys[i]
            if not args.debug_gt_label:                    #debug_gt_label=0，用伪标签监督训练
                data.yss = data.ys[i]
                # print("Using pseudo label")
            else:
                print("Using ground truth label")
            # import ipdb; ipdb.set_trace()
        for ii in tqdm(range(epoch)):                                                                        #tqdm是一个快速、可扩展的Python进度条库，用于在Python长循环中添加一个进度提示信息，用户只需封装任何迭代器 tqdm(iterator)
            # ipdb.set_trace()
            labeled_mask = data.labeled_mask
            train_mask = data.train_mask
            val_mask = data.val_mask
            # print("labeled: ",len(labeled_mask),torch.sum(labeled_mask).item(),"unlabeled: ", len(train_mask),torch.sum(train_mask).item())
            # print("\ntruth: ", len(data.y), data.y)
            lab_los_weight = args.l_l_weight - 0.001 * ii                                                              #随epoch动态降低
            if need_train:                                                                                  #训练，若训练模型是LP则不进入
                if 'rim' in args.strategy or 'iterative' in args.strategy or args.split == 'active_train':  #节点选择策略为rim或iterative，或数据集掩码操作采用所提的节点主动选择+置信度后过滤
                    train_loss, val_loss, val_acc, train_acc = train(model, data, optimizer, loss_fn, lab_los_weight, labeled_mask, train_mask, val_mask, args.no_val, reliability)
                else:
                    if args.model_name == 'S_model':
                        train_loss, val_loss, val_acc, train_acc = s_train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val, noise_ada)
                    else:
                        train_loss, val_loss, val_acc, train_acc = train(model, data, optimizer, loss_fn, lab_los_weight, labeled_mask, train_mask, val_mask, args.no_val)
                        # print("epoch:", ii, ": ", train_loss, train_acc, val_loss, val_acc)
                if scheduler:                                                                               #采用adam，scheduler = None
                    scheduler.step()
                if args.output_intermediate and not args.no_val:
                    print(f"Epoch {ii}: Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_acc[0]}")
                if args.debug:
                    if args.filter_strategy == 'none':
                        test_acc, res = test(model, data, 0, data.test_mask)
                    else:
                        test_acc, res = test(model, data, 0, data.test_mask, data.backup_y)                 #测试，返回测试集acc指标值
                    # print(f"Epoch {ii}: Test acc: {test_acc}")
                    debug_acc.append(test_acc)
                    this_train_acc.append(train_acc)
                if not args.no_val:
                    if val_acc > best_val:
                        best_val = val_acc
                        best_model = deepcopy(model)
                        early_stop_accum = 0
                    else:
                        if ii >= args.early_stop_start:
                            early_stop_accum += 1
                        if early_stop_accum > args.early_stopping and ii >= args.early_stop_start:
                            print(f"Early stopping at epoch {ii}")
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
        # trainmax_index, trainmax_value = max(enumerate(this_train_acc), key=lambda x: x[1])
        # testmax_index, testmax_value = max(enumerate(debug_acc), key=lambda x: x[1])
        # print(seed,"\ntrain:\ntra_best: ",trainmax_index, trainmax_value,"\ntra_sets: ", this_train_acc,"\nval_best: ", testmax_index, testmax_value,"\nval_set: ", debug_acc)
        # print("test:", test_acc,"\n\n")
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
    data = get_dataset(seeds, args.dataset, args.split, args.data_format, args.label_pc, data_path, logit_path, args.pl_noise, args.no_val, args.budget, args.strategy, args.num_centers, args.compensation, args.save_data, args.filter_strategy, args.max_part, args.oracle, reliability_list, args.total_budget, args.second_filter, True, False, args.filter_all_wrong_labels, args.alpha, args.beta, args.gamma, args.ratio).to(device)
    # print("_______\n",data,"\n___________\n\n")
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
    data = get_dataset(sweep_seeds, args.dataset, args.split, args.data_format, args.label_pc, data_path, logit_path, args.pl_noise, args.no_val, args.budget, args.strategy, args.num_centers, args.compensation, args.save_data, args.filter_strategy, args.max_part, args.oracle, reliability_list, args.total_budget, args.second_filter, True, False, args.filter_all_wrong_labels, args.alpha, args.beta, args.gamma, args.ratio).to(device)
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

def QA_llm_for_pre_and_conf(model_path, data_path, args = None):
    seeds = [i for i in range(args.main_seed_num)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')             #作用是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行

    vars(args)['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    params_dict = load_yaml(args.yaml_path)
    logit_path = params_dict['LOGIT_PATH']
    reliability_list = []
    data = get_dataset(seeds, args.dataset, args.split, args.data_format, args.label_pc, data_path, logit_path, args.pl_noise, args.no_val, args.budget, args.strategy, args.num_centers, args.compensation, args.save_data, args.filter_strategy, args.max_part, args.oracle, reliability_list, args.total_budget, args.second_filter, True, False, args.filter_all_wrong_labels, args.alpha, args.beta, args.gamma, args.ratio)
    full_mapping = load_mapping()
    data.label_names = [full_mapping[args.dataset][x] for x in data.label_names]
    data.label_names = [x.lower() for x in data.label_names]


    question_text = "What is the category of this paper?"
    content_text = "Stochastic pro-positionalization of non-determinate background knowledge. : It is a well-known fact that propositional learning algorithms require \"good\" features to perform well in practice. So a major step in data engineering for inductive learning is the construction of good features by domain experts. These features often represent properties of structured objects, where a property typically is the occurrence of a certain substructure having certain properties. To partly automate the process of \"feature engineering\", we devised an algorithm that searches for features which are defined by such substructures. The algorithm stochastically conducts a top-down search for first-order clauses, where each clause represents a binary feature. It differs from existing algorithms in that its search is not class-blind, and that it is capable of considering clauses (\"context\") of almost arbitrary length (size). Preliminary experiments are favorable, and support the view that this approach is promising."
    content_text2 = "Stochastic pro-positionalization of non-determinate background knowledge: Both propositional and relational learning algorithms require a good representation to perform well in practice. Usually such a representation is either engineered manually by domain experts or derived automatically by means of so-called constructive induction. Inductive Logic Programming (ILP) algorithms put a somewhat less burden on the data engineering effort as they allow for a structured, relational representation of background knowledge. In chemical and engineering domains, a common representational device for graph-like structures are so-called non-determinate relations. Manually engineered features in such domains typically test for or count occurrences of specific substructures having specific properties. However, representations containing non-determinate relations pose a serious efficiency problem for most standard ILP algorithms. Therefore, we have devised a stochastic algorithm to automatically derive features from non-determinate background knowledge. The algorithm conducts a top-down search for first-order clauses, where each clause represents a binary feature. These features are used instead of the non-determinate relations in a subsequent induction step. In contrast to comparable algorithms search is not class-blind and there are no arbitrary size restrictions imposed on candidate clauses. An empirical investigation in three chemical domains supports the validity and usefulness of the proposed algorithm."
    content_text3 = "Title: Design by interactive exploration using memory-based techniques. Abstract: One of the characteristics of design is that designers rely extensively on past experience in order to create new designs. Because of this, memory-based techniques from artificial intelligence which help store, organise, retrieve, and reuse experiential knowledge held in memory are good candidates for aiding designers. Another characteristic of design is the phenomenon of exploration in the early stages of design configuration. A designer begins with an ill structured, partially defined, problem specification, and through a process of exploration gradually refines and modifies it as his/her understanding of the problem improves. The paper describes DEMEX, an interactive computer-aided design system that employs memory-based techniques to help its users explore the design problems they pose to the system, so that they can acquire a better understanding of the requirements of the problems. DEMEX has been applied in the domain of the structural design of buildings."
    # input_text = content_text + " " + question_text  # 将上下文和问题拼接在一起，形成一个完整的输入序列

    object_cat = configs[args.dataset]['zero-shot']['object-cat']
    question = configs[args.dataset]['zero-shot']['question']
    # answer_format = configs[args.dataset]['zero-shot']['answer-format']
    answer_format = configs2[args.dataset]['zero-shot']['answer-format']
    prompt = "{}: \n".format(object_cat)
    prompt += (content_text3 + "\n")
    prompt += "Task: \n"
    prompt += "There are following categories: \n"
    prompt += "[" + ", ".join(full_mapping['cora']) + "]" + "\n"
    # prompt += question + "\n"
    # prompt += answer_format
    prompt += top_k_question(question, 3)
    print("prompt:\n",prompt,"\n--------------\n\n")

    # answer_text = generate_llama32_1b(prompt, model_path, device)
    # answer_text = QA_llama32_3b(prompt, model_path, device)
    # answer_text = QA_llama32_3b_Ins(prompt, model_path, device)
    # answer_text = QA_llama31_8b_Ins(prompt, model_path, device)
    # answer_text = QA_Qwen25_3b_Ins(prompt, model_path, device)
    answer_text = QA_Deep_R1_Qwen25(prompt, model_path, device)

    print(answer_text[-1])
def top_k_question(question, k):
    str = "and then give reasons of each one like (((1: <reason1>)), ...)"
    # return "Question: {}. Provide your {} best guesses and a confidence number, in the form of a list of python dicts like [{{\"answer\": <your_answer>, \"confidence\": <your_confidence>}}, ...], \
    #     each confidence is correct (0.00 to 100.00) for the following question from most probable to least, and the sum of all confidence should be 100.00. \
    #     ".format(question, k)

    return "Question: {}. Give your {} best guesses, with confidence scores representing their probability from most likely to least likely. Each confidence ranges from 0.0 to 100.0, and their sum should be 100.0. Please in the form of a list of python dicts like \
        [{{\"answer\": <answer_here>, \"confidence\": <confidence_here>}}, ...]. \
    ".format(question, k)

def generate_llama32_1b_origin(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # 分词器
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 use_cache=True,
                                                 low_cpu_mem_usage=True).cuda()  # 模型
    model.eval()

    question_text = "What's the category of this paper"
    content_text = "The title of one paper is \"Stochastic pro-positionalization of non-determinate background knowledge\". And its abstract is: It is a well-known fact that propositional learning algorithms require \"good\" features to perform well in practice. So a major step in data engineering for inductive learning is the construction of good features by domain experts. These features often represent properties of structured objects, where a property typically is the occurrence of a certain substructure having certain properties. To partly automate the process of \"feature engineering\", we devised an algorithm that searches for features which are defined by such substructures."  # 输入文本
    input_text = content_text + " " + question_text  # 将上下文和问题拼接在一起，形成一个完整的输入序列
    # inputs = tokenizer(input_text, return_tensors='pt').cuda()  # 使用分词器对输入文本进行编码

    input_ids = tokenizer.encode(input_text, return_tensors='pt').cuda()
    # attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    attention_mask = input_ids[0].ne(tokenizer.pad_token_id).long()

    # 生成回答
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=200,
            do_sample=True,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id,
        )  # 生成文本
    # top_k=50,
    # no_repeat_ngram_size=2,  # 防止生成重复 n-grams
    # num_beams=10,
    # early_stopping=True,

    # output_ids = outputs[0][len(input_ids[0])+1:]
    generated_text = tokenizer.decode(outputs, skip_special_tokens=True)  # 将生成的id转换为文本
    print(generated_text)
def generate_llama32_1b(prompt, model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # 分词器
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 use_cache=True,
                                                 low_cpu_mem_usage=True).to(device)  # 模型
    model.eval()
    inputs = tokenizer(prompt, return_tensors='pt').to(device)  # 使用分词器对输入文本进行编码
    # input_ids = tokenizer.encode(input_text, return_tensors='pt').cuda()
    # attention_mask = torch.ones(inputs['input_ids'].shape, dtype=torch.bfloat16).to(device)
    attention_mask = inputs['input_ids'].ne(tokenizer.pad_token_id).long()
    # print('input_ids\n',inputs['input_ids'],"\n--------------\n\n")

    with torch.inference_mode():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
        ).to(device)  # 生成文本
    # max_new_tokens = 20,
    # do_sample = True,
    # temperature = 0.2,
    # top_k = 30

    # print('output\n',outputs,"\n--------------\n\n")
    output_ids = [outputid[len(inputid):] for inputid, outputid in zip(inputs['input_ids'], outputs)]
    generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)  # 将生成的id转换为文本
    return generated_text

def QA_llama32_3b(prompt, model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # 分词器
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 use_cache=True,
                                                 low_cpu_mem_usage=True).to(device)  # 模型
    model.eval()
    inputs = tokenizer(prompt, return_tensors='pt').to(device)  # 使用分词器对输入文本进行编码
    # input_ids = tokenizer.encode(input_text, return_tensors='pt').cuda()
    # attention_mask = torch.ones(inputs['input_ids'].shape, dtype=torch.bfloat16).to(device)
    attention_mask = inputs['input_ids'].ne(tokenizer.pad_token_id).long()
    # print('input_ids\n',inputs['input_ids'],"\n--------------\n\n")

    with torch.inference_mode():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=attention_mask,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
        ).to(device)  # 生成文本
    # print('output\n',outputs,"\n--------------\n\n")

    output_ids = [outputid[len(inputid):] for inputid, outputid in zip(inputs['input_ids'], outputs)]
    generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)  # 将生成的id转换为文本
    return generated_text
def QA_llama32_3b_Ins(prompt, model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # 分词器
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 use_cache=True,
                                                 low_cpu_mem_usage=True).to(device)  # 模型
    model.eval()

    pipeline = transformers.pipeline(
        "text-generation",  #pipeline中支持的任务类型包括"text-generation", "question-answering", "summarization"
        model=model,
        tokenizer=tokenizer,
        device_map=device,
        batch_size=16,
    )

    # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    messages = [
        {"role": "user", "content": prompt},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    # print(outputs[0]["generated_text"][-1])
    return outputs[0]["generated_text"]
def QA_llama31_8b_Ins(prompt, model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # 分词器
    tokenizer.pad_token = tokenizer.eos_token

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        device_map=device,
        batch_size=32,
    )

    # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    messages = [
        {"role": "user", "content": prompt},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])
    return outputs[0]["generated_text"]
def QA_llama32_1b_Ins(prompt, model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # 分词器
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 use_cache=True,
                                                 low_cpu_mem_usage=True).to(device)  # 模型
    model.eval()

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map='auto',
    )

    # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    messages = [
        {"role": "user", "content": prompt},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    # print(outputs[0]["generated_text"][-1])
    return outputs[0]["generated_text"]

def QA_Qwen25_3b_Ins(prompt, model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # 分词器
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 use_cache=True,
                                                 low_cpu_mem_usage=True).to(device)  # 模型
    model.eval()

    pipeline = transformers.pipeline(
        "text-generation",  #pipeline中支持的任务类型包括"text-generation", "question-answering", "summarization"
        model=model,
        tokenizer=tokenizer,
        device_map=device,
        batch_size=16,
    )

    # {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    messages = [
        {"role": "user", "content": prompt},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    # print(outputs[0]["generated_text"][-1])
    return outputs[0]["generated_text"]
def QA_Deep_R1_Qwen25(prompt, model_path, device):
    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()
    print(initial_allocated, ",", initial_reserved) # 记录初始已分配的 GPU 内存

    tokenizer = AutoTokenizer.from_pretrained(model_path)  # 分词器
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 use_cache=True,
                                                 low_cpu_mem_usage=True).to(device)  # 模型
    model.eval()

    pipeline = transformers.pipeline(
        "text-generation",  #pipeline中支持的任务类型包括"text-generation", "question-answering", "summarization"
        model=model,
        tokenizer=tokenizer,
        device_map=device,
        batch_size=16,
    )
    # 记录创建变量后已分配的 GPU 内存
    allocated_after_creation = torch.cuda.memory_allocated()
    reserved_after_creation = torch.cuda.memory_reserved()
    # 计算中间变量占用的内存
    memory_allocated_by_x = allocated_after_creation - initial_allocated
    memory_reserved_by_x = reserved_after_creation - initial_reserved
    print("  after: ", allocated_after_creation, ",", reserved_after_creation)
    print(f"中间变量 占用的已分配内存: {memory_allocated_by_x} 字节")
    print(f"中间变量 占用的预留内存: {memory_reserved_by_x} 字节")

    # {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    messages = [
        {"role": "user", "content": prompt},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=1024,
    )
    # print(outputs[0]["generated_text"][-1])
    return outputs[0]["generated_text"]


if __name__ == '__main__':
    current_time = int(time.time())
    # model_path1 = 'E:/DL_python/LLM4Graph_Modules/Llama_3.2_1B'             #模型
    # model_path11 = 'E:/DL_python/LLM4Graph_Modules/Llama_3.2_1B_Instruct'
    # model_path12 = 'E:/DL_python/LLM4Graph_Modules/Llama_3.2_3B_Instruct'
    model_path13 = 'E:/DL_python/LLM4Graph_Modules/Llama_3.1_8B_Instruct'
    # model_path21 = 'E:/DL_python/LLM4Graph_Modules/Qwen_2.5_1.5B_Instruct'
    # model_path22 = 'E:/DL_python/LLM4Graph_Modules/Qwen_2.5_3B_Instruct'
    model_path23 = 'E:/DL_python/LLM4Graph_Modules/Qwen_2.5_7B_Instruct'
    model_path24 = 'E:/DL_python/LLM4Graph_Modules/DeepSeek_R1_Distill_Qwen_1.5B'
    model_path25 = 'E:/DL_python/LLM4Graph_Modules/DeepSeek_R1_Distill_Qwen_7B'

    # tokenizer_path = 'D:/FSGL-APL/DeepSeek_R1_Distill_Qwen/tokenizer_config.json'    #分词器
    output_path = 'D:/FSGL-APL/DeepSeek_R1_Distill_Qwen_tune/'
    # run1(model_path,output_path)
    # run2(model_path)

    print("—————— Start ——————")
    args = get_command_line_args_datasets()  #参数
    params_dict = load_yaml(args.yaml_path)  #地址
    data_path = params_dict['DATA_PATH']
    # seeds = [i for i in range(args.main_seed_num)]


    # QA_llm_for_pre_and_conf(model_path24, data_path, args = args)
    # generate_llama32_3b(model_path2)

    if args.mode == "main":
        main(data_path, args = args)         #训练
    else:
        sweep(data_path, args = args)
    print("—————— END ——————")

#运行：python src/test111.py
'''
运行cora例子：
python src/test111.py --dataset cora --model_name GCN --data_format sbert --main_seed_num 3 --split active --output_intermediate 0 --no_val 1 --strategy pagerank2 --debug 1 --total_budget 140 --filter_strategy consistency --loss_type ce --second_filter conf+entropy --epochs 30 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0.2 --alpha 0.33 --beta 0.33

运行PubMed例子：
python src/test111.py --dataset pubmed --model_name GCN --data_format sbert --main_seed_num 3 --split active --output_intermediate 0 --no_val 1 --strategy pagerank2 --debug 1 --total_budget 120 --filter_strategy consistency --loss_type ce --second_filter conf+entropy --epochs 30 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0.2 --alpha 0.33 --beta 0.33
'''