import transformers
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
import re
import tqdm
import json
from openail.utils import efficient_openai_text_api, set_endpoints, openai_text_api, openai_text_api_with_top_p, load_partial_openai_result, save_partial_openai_result, retrieve_dict, compute_ece, plot_calibration_curve, openai_text_api_with_backoff, num_tokens_from_string
from helper.data import get_dataset, inject_random_noise_y_level
from helper.args import get_command_line_args, get_command_line_args_datasets
from helper.active import train_lr, inference_lr
from helper.utils import load_yaml
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
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def QA_llm_for_pre_and_conf(model_path, data_path, args = None):
    seeds = [i for i in range(args.main_seed_num)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')             #作用是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行

    vars(args)['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    params_dict = load_yaml(args.yaml_path)
    logit_path = params_dict['LOGIT_PATH']
    reliability_list = []
    data = get_dataset(seeds, args.dataset, args.split, args.data_format, data_path, logit_path, args.pl_noise, args.no_val, args.budget, args.strategy, args.num_centers, args.compensation, args.save_data, args.filter_strategy, args.max_part, args.oracle, reliability_list, args.total_budget, args.second_filter, True, False, args.filter_all_wrong_labels, args.alpha, args.beta, args.gamma, args.ratio)
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
    answer_text = QA_Deep_R1_Qwen25_7b(prompt, model_path, device)

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
def QA_Deep_R1_Qwen25_7b(prompt, model_path, device):
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
        max_new_tokens=1500,
    )
    # print(outputs[0]["generated_text"][-1])
    return outputs[0]["generated_text"]


if __name__ == '__main__':
    print("AAAAAAAAAAAAAAAAAAAAAAAA")
    # model_path1 = 'E:/LLM4Graph_Modules/Llama_3.2_1B'             #模型
    model_path11 = 'E:/LLM4Graph_Modules/Llama_3.2_1B_Instruct'
    model_path12 = 'E:/LLM4Graph_Modules/Llama_3.2_3B_Instruct'
    model_path13 = 'E:/LLM4Graph_Modules/Llama_3.1_8B_Instruct'
    model_path21 = 'E:/LLM4Graph_Modules/Qwen_2.5_1.5B_Instruct'
    model_path22 = 'E:/LLM4Graph_Modules/Qwen_2.5_3B_Instruct'
    model_path23 = 'E:/LLM4Graph_Modules/Qwen_2.5_7B_Instruct'
    model_path24 = 'E:/LLM4Graph_Modules/DeepSeek_R1_Distill_Qwen_1.5B'
    model_path25 = 'E:/LLM4Graph_Modules/DeepSeek_R1_Distill_Qwen_7B'

    # tokenizer_path = 'D:/FSGL-APL/DeepSeek_R1_Distill_Qwen_7B/tokenizer_config.json'    #分词器
    output_path = 'D:/FSGL-APL/DeepSeek_R1_Distill_Qwen_7B_tune/'
    # run1(model_path,output_path)
    # run2(model_path)

    args = get_command_line_args_datasets()
    params_dict = load_yaml(args.yaml_path)  # 数据集地址
    data_path = params_dict['DATA_PATH']
    # seeds = [i for i in range(args.main_seed_num)]


    QA_llm_for_pre_and_conf(model_path23, data_path, args = args)

    # generate_llama32_8b(model_path2)

    print("—————— END ——————")


#运行：python src/test111.py
