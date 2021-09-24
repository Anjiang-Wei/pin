# -*- coding: utf-8 -*-
# https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/dynamic_quantization_bert_tutorial.ipynb#scrollTo=wKRaaq8FIm3o

from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import os
import random
import sys
import time
import torch

from argparse import Namespace
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

logging.getLogger("transformers.modeling_utils").setLevel(
   logging.WARN)  # Reduce logging

print(torch.__version__)

"""We set the number of threads to compare the single thread performance between FP32 and INT8 performance. In the end of the tutorial, the user can set other number of threads by building PyTorch with right parallel backend."""

torch.set_num_threads(1)
print(torch.__config__.parallel_info())

configs = Namespace()

# The output directory for the fine-tuned model.
configs.output_dir = "/pool0/anjiang/pin/pin/model/MRPC/"
# configs.output_dir = "./MRPC/"

# The data directory for the MRPC task in the GLUE benchmark.
configs.data_dir = "/pool0/anjiang/pin/pin/model/glue_data/MRPC"

# The model name or path for the pre-trained model.
configs.model_name_or_path = "bert-base-uncased"
# The maximum length of an input sequence
configs.max_seq_length = 128

# Prepare GLUE task.
configs.task_name = "MRPC".lower()
configs.processor = processors[configs.task_name]()
configs.output_mode = output_modes[configs.task_name]
configs.label_list = configs.processor.get_labels()
configs.model_type = "bert".lower()
configs.do_lower_case = True

# Set the device, batch size, topology, and caching flags.
configs.device = "cpu"
configs.per_gpu_eval_batch_size = 8
configs.n_gpu = 0
configs.local_rank = -1
configs.overwrite_cache = False


# Set random seed for reproducibility.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)

"""## 2.2 Load the fine-tuned BERT model

We load the tokenizer and fine-tuned BERT sequence classifier model (FP32) from the `configs.output_dir`.
"""

tokenizer = BertTokenizer.from_pretrained(
    configs.output_dir, do_lower_case=configs.do_lower_case)

model = BertForSequenceClassification.from_pretrained(configs.output_dir)
model.to(configs.device)

"""## 2.3 Define the tokenize and evaluation function
We reuse the tokenize and evaluation function from [HuggingFace](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py).
"""

def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1] 
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

"""# 3. Apply the dynamic quantization

We call `torch.quantization.quantize_dynamic` on the model to apply the dynamic quantization on the HuggingFace BERT model. Specifically,

- We specify that we want the torch.nn.Linear modules in our model to be quantized;
- We specify that we want weights to be converted to quantized int8 values.
"""

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
# print(quantized_model)

"""## 3.1 Check the model size
Let's first check the model size. We can observe a significant reduction in model size:
"""

# def print_size_of_model(model, file_to_save):
#     torch.save(model.state_dict(), file_to_save)
#     print('Size (MB):', os.path.getsize(file_to_save)/1e6)
#     # os.remove('temp.p')

# print_size_of_model(model, "fp32")
# print_size_of_model(quantized_model, "int8")

def param(model):
    for name, param in model.named_parameters():
        print(name, param)

# name_param(model)
param(quantized_model)


"""The BERT model used in this tutorial (bert-base-uncased) has a vocabulary size V of 30522. With the embedding size of 768, the total size of the word embedding table is ~ 4 (Bytes/FP32) * 30522 * 768 = 90 MB. So with the help of quantization, the model size of the non-embedding table part is reduced from 350 MB (FP32 model) to 90 MB (INT8 model).

## 3.2 Evaluate the inference accuracy and time

Next, let's compare the inference time as well as the evaluation accuracy between the original FP32 model and the INT8 model after the dynamic quantization.
"""

# def time_model_evaluation(model, configs, tokenizer):
#     eval_start_time = time.time()
#     result = evaluate(configs, model, tokenizer, prefix="")
#     eval_end_time = time.time()
#     eval_duration_time = eval_end_time - eval_start_time
#     print(result)
#     print("Evaluate total time (seconds): {0:.1f}".format(eval_duration_time))

# # Evaluate the original FP32 BERT model
# time_model_evaluation(model, configs, tokenizer)

# # Evaluate the INT8 BERT model after the dynamic quantization
# time_model_evaluation(quantized_model, configs, tokenizer)

# """Running this locally on a MacBook Pro, without quantization, inference (for all 408 examples in MRPC dataset) takes about 160 seconds, and with quantization it takes just about 90 seconds. We summarize the results for running the quantized BERT model inference on a Macbook Pro as the follows:

# ```
# | Prec | F1 score | Model Size | 1 thread | 4 threads | 
# | FP32 |  0.9019  |   438 MB   | 160 sec  | 85 sec    |
# | INT8 |  0.8953  |   181 MB   |  90 sec  | 46 sec    |
# ```

# We have 0.6% F1 score accuracy after applying the post-training dynamic quantization on the fine-tuned BERT model on the MRPC task. As a comparison, in a [recent paper](https://arxiv.org/pdf/1910.06188.pdf) (Table 1), it achieved 0.8788 by applying the post-training dynamic quantization and 0.8956 by applying the quantization-aware training. The main difference is that we support the asymmetric quantization in PyTorch while that paper supports the symmetric quantization only.
# """

# quantized_output_dir = configs.output_dir + "quantized/"
# if not os.path.exists(quantized_output_dir):
#     os.makedirs(quantized_output_dir)
#     quantized_model.save_pretrained(quantized_output_dir)



