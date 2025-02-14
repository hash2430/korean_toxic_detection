# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import csv

import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm
import pandas as pd
from pandas import DataFrame

from models.modeling_bert import SequenceClassificationMultitask, Config
from utils.tokenization import BertTokenizer
from utils.classifier_utils import KorNLIProcessor, KorSTSProcessor, KorTDTestProcessor, convert_examples_to_features, compute_metrics


logger = logging.getLogger(__name__)


processors = {
    "kortd": KorTDTestProcessor,
    "kornli": KorNLIProcessor,
    "korsts": KorSTSProcessor
}

output_modes = {
    "kortd": "classification_multitask",
    "kornli": "classification",
    "korsts": "regression"
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--config_file",
                        type=str,
                        required=True,
                        help="model configuration file")
    parser.add_argument("--vocab_file",
                        type=str,
                        required=True,
                        help="tokenizer vocab file")
    parser.add_argument("--checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="fine-tuned model checkpoint")
    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    args.n_gpu = n_gpu

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

    logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    set_seed(args)

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer(args.vocab_file, max_len=args.max_seq_length)

    test_examples = processor.get_test_examples(args.data_dir)
    texts = [test_example.text_a for test_example in test_examples]

    # Prepare model
    config = Config(args.config_file)
    model = SequenceClassificationMultitask(config)
    model.load_state_dict(torch.load(args.checkpoint))
    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)
    model.to(device)
    if args.fp16:
        model = model.half()

    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    logger.info("***** Running Evaluation *****")
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)

    if output_mode == "classification_multitask":
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.float)

    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    model.eval()
    preds = []
    preds0 = []
    preds1 = []
    preds2 = []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        if output_mode == "classification_multitask":
            if len(preds0) == 0:
                preds0.append(logits[0].detach().cpu().numpy())
                preds1.append(logits[1].detach().cpu().numpy())
                preds2.append(logits[2].detach().cpu().numpy())
            else:
                preds0[0] = np.append(preds0[0], logits[0].detach().cpu().numpy(), axis=0)
                preds1[0] = np.append(preds1[0], logits[1].detach().cpu().numpy(), axis=0)
                preds2[0] = np.append(preds2[0], logits[2].detach().cpu().numpy(), axis=0)
        else:
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

    if output_mode == "classification_multitask":
        preds0 = preds0[0]
        preds1 = preds1[0]
        preds2 = preds2[0]

        preds0 = np.argmax(preds0, axis=1)
        preds1 = np.argmax(preds1, axis=1)
        preds2 = np.argmax(preds2, axis=1)
        preds = [preds0, preds1, preds2]
        label=[]
        for p in preds0:
            if p==0:
                label.append('True')
            elif p==1:
                label.append('False')

    elif output_mode == "classification":
        preds = preds[0]
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = preds[0]
        preds = np.squeeze(preds)

    #TODO: write the csv
    #result = compute_metrics(task_name, preds, all_label_ids.numpy())
    data = {'comments':texts,
            'label':label}
    data_df = DataFrame(data)
    csv_write_path='/mnt/sdc1/korLM/by_me/out/by_me_9_gender_0608.csv'
    # data_df.to_csv(csv_write_path, sep=',',index=False)
    with open(csv_write_path, 'w', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['comments','label'])
        for text, lab in zip(texts, label):
            wr.writerow([text, lab])


    logger.info("***** Eval results *****")
    # for key in sorted(result.keys()):
    #     logger.info("  %s = %s", key, str(result[key]))


if __name__ == "__main__":
    main()
