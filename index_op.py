from pathlib import Path
import sys
sys.path.append("/home/tunghoang/hieunn/pyvene")

from data import generate_sample
import torch
import pandas as pd
import numpy as np
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from vllm import LLM, SamplingParams
from data import create_prompt
from utils.data import apply_chat_template
import argparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0')
    parser.add_argument('--model_alias', type=str, required=True)
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--no-task-description', action="store_true")
    parser.add_argument('--psize', type=int, default=1)
    parser.add_argument('--is-sft-model', action="store_true")
    # parser.add_argument('--data_frac', type=int, default=0)
    # parser.add_argument('--frac_len', type=int, default=0)
    # parser.add_argument('--data_path', type=str, default='HuggingFaceH4/ultrachat_200k')
    return parser.parse_args()
args = parse_arguments()


dataset_alias = "value_at_index_v0_1"
# ds = load_from_disk("datasets/value_at_index/v0.1")
# ds = ds.filter(lambda e: e['query_array_length'] < 4)

raw_dataset = load_dataset("nnheui/understanding-index-operation-v0.1")['train']


revision = args.revision
if args.no_task_description:
    output_path = Path("outputs/index-op-v0.1-wotd/")
else:
    output_path = Path("outputs/index-op-v0.1/")
output_path.mkdir(exist_ok=True, parents=True)

model_path = args.model
output_path /= args.model_alias

tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# ds = ds.filter(lambda e: e['num_few_shots'] == 0)
# ds = ds.map(create_prompt,
#             fn_kwargs={"include_task_description": not args.no_task_description},
#             )
if args.is_sft_model:
    def create_messages(e):
        e = create_prompt(e)
        e['messages'] = [
            {'role': 'user', 'content': e['prompt']},
            # {'role': 'assistant', 'content': str(e['target'])}
        ]
        # apply_chat_template(e, tokenizer=tokenizer, task='sft', auto_insert_empty_system_msg=True)
        return e
    raw_dataset = raw_dataset.map(create_messages)
    # raw_dataset = raw_dataset.shuffle(42)
    # raw_datasets = raw_dataset.train_test_split(test_size=0.5)
    column_names = list(raw_dataset.features)
    raw_dataset = raw_dataset.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "generation",
            "auto_insert_empty_system_msg": True,
        },
        num_proc=2,
        remove_columns=column_names,
        desc="Applying chat template",
    )
    # ds = raw_datasets['train']
    prompt_data = raw_dataset['text']
else:
    ds = raw_dataset
    # ds = ds.filter(lambda e: e['num_few_shots'] == 0)
    ds = ds.map(create_prompt,
                fn_kwargs={"include_task_description": not args.no_task_description},
                )
    prompt_data = ds['prompt']


print("----- Example prompt -----")
print(prompt_data[1000])
print("----- End of Example prompt -----")

llm = LLM(
    model=model_path,
    tensor_parallel_size=args.psize,
    revision=revision,
    # gpu_memory_utilization=0.4, 
)
sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1, logprobs=1)
results_gathered = llm.generate(prompt_data, sampling_params)

# # _, pred_toks,_ = forward(all_prompts)
# # print(query, pred_tok, answer)
all_preds = []
logprobs = []
for i, result in enumerate(results_gathered):
    pred = result.outputs[0].text.strip()
    all_preds.append(result.outputs[0].text)
    logprobs.append({k:(v.logprob, v.rank, v.decoded_token) for k,v in result.outputs[0].logprobs[0].items()})
    # d += (pred == ds[i]['target'])
    if i % 1000 == 0:
        print('-----')
        print(prompt_data[i])
        print(result.outputs[0].text)
        print(pred)
    # pred_tok = tokenizer.(result.outputs[0].text)
# acc[num_shot, q_len - qlen_min] = correct / (num_prompt * num_query)
torch.save({"all_preds": all_preds, "logprobs": logprobs}, output_path)