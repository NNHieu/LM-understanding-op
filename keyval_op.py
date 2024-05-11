from pathlib import Path
import sys
sys.path.append("/home/tunghoang/hieunn/pyvene")

from data import generate_sample
import torch
import pandas as pd
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

from vllm import LLM, SamplingParams
from data import create_prompt_dictionary
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
    # parser.add_argument('--data_frac', type=int, default=0)
    # parser.add_argument('--frac_len', type=int, default=0)
    # parser.add_argument('--data_path', type=str, default='HuggingFaceH4/ultrachat_200k')
    return parser.parse_args()
args = parse_arguments()


dataset_alias = "value_at_index_v0_1"
ds = load_from_disk("datasets/value_at_index/v0.1")
# ds = ds.filter(lambda e: e['query_array_length'] < 4)

revision = args.revision
if args.no_task_description:
    output_path = Path("outputs/keyval-op-v0.1_wotd")
else:
    output_path = Path("outputs/keyval-op-v0.1/")
output_path.mkdir(exist_ok=True, parents=True)

model_path = args.model
output_path /= args.model_alias

tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# ds = ds.filter(lambda e: e['num_few_shots'] == 0)
ds = ds.map(create_prompt_dictionary,
            fn_kwargs={"include_task_description": not args.no_task_description},
            )

print("----- Example prompt -----")
print(ds[1000]['prompt'])
print("----- End of Example prompt -----")

llm = LLM(
    model=model_path,
    tensor_parallel_size=args.psize,
    revision=revision,
    # gpu_memory_utilization=0.4, 
)
sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1, logprobs=1)
results_gathered = llm.generate(ds['prompt'], sampling_params)

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
        print(ds[i]['prompt'])
        print(result.outputs[0].text)
        print(pred)
    # pred_tok = tokenizer.(result.outputs[0].text)
# acc[num_shot, q_len - qlen_min] = correct / (num_prompt * num_query)
torch.save({"all_preds": all_preds, "logprobs": logprobs}, output_path)