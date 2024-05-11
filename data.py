import numpy as np
import datasets
from tqdm import tqdm
import pandas as pd
import random
from copy import deepcopy

def generate_sample(x_length, high=10, is_example=False):
    # a = np.random.randint(size=(x_length,), low=0, high=high)
    a = list(range(10))
    random.shuffle(a)
    a = a[:x_length]
    ind = np.random.randint(low=0, high=x_length)
    # s = f'a={a.tolist()}. a[{ind}]='
    e = {
        "array": a,
        "query": ind,
        'target': a[ind]
    }
    # ans = f'{a[ind]}'
    # if is_example:
    #     return s + ans
    # return s, ans
    return e

def create_dataset(qlen_min=3, qlen_max=10, max_num_shot=5, num_prompt=10, num_query=100):
    q_len_range = range(qlen_min, qlen_max + 1)
    num_shot_range = range(0, max_num_shot + 1)
    acc = np.zeros((len(num_shot_range), len(q_len_range)))

    # Create prompts
    data = []
    # all_prompts = []
    # all_answers = []
    task_prompt = """Output the value of an array at a given index."""
    for num_shot in tqdm(num_shot_range):
        # task_prompt = ""
        # task_prompt = task_prompt0
        for q_len in q_len_range:
            correct = 0
            for i in range(num_prompt):
                few_shots = []
                for shot in range(num_shot):
                    length = np.random.randint(3, 6)
                    few_shots.append(generate_sample(length, is_example=True))
                # few_shot_text += generate_sample(length, is_example=True) + "\n"
                # print(task_prompt)
                answers = []
                queries = []
                for _ in range(num_query):
                    e = generate_sample(q_len)
                    # queries.append(query)
                    # answers.append(answer)
                    e['few_shots'] = few_shots
                    e['query_array_length'] = q_len
                    e['num_few_shots'] = num_shot
                    e['task_prompt'] = task_prompt
                    data.append(e)
                # all_prompts += [(task_prompt + q) for q in queries]
                # all_answers += answers
    
    ds = datasets.Dataset.from_pandas(pd.DataFrame(data=data))
    return ds

def create_prompt(e, include_task_description=True):
    task_description = "Output the value of an array at a given index.\n"

    task_prompt = ""
    if include_task_description:
        task_prompt = task_description


    def to_qtext(s, include_target=False):
        # template = f"a = {s['array']}\nQ: a[{s['query']}]\nA: "
        # template = f"Let a={s['array']}. The value of the array at index {s['query']} is "
        template = f"a={s['array']}. a[{s['query']}]="
        # template = f"a={s['array']}\nQ: a[{s['query']}]\nA: "
        if not include_target:
            return template
        return template + f"{s['target']}"
    
    if len(e['few_shots']) == 0:
        task_prompt += to_qtext(e)
        e['prompt'] = task_prompt
    else:
        for s in e['few_shots']:
            task_prompt += to_qtext(s, True) + "\n"
        task_prompt += to_qtext(e)
        e['prompt'] = task_prompt
    return e

def create_prompt_dictionary(e, include_task_description=True):
    task_description = "Output the value of a dictionary at a given key.\n"

    task_prompt = ""
    if include_task_description:
        task_prompt = task_description

    def to_qtext(s, include_target=False):
        # template = f"a = {s['array']}\nQ: a[{s['query']}]\nA: "
        # template = f"Let a={s['array']}. The value of the array at index {s['query']} is "
        keys = [
            'a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
        ][:len(s['array'])]
        random.shuffle(keys)
        d = dict(zip(keys, s['array']))
        s_arr = "{" + ", ".join([f"{k}: {v}" for k, v in d.items()]) + "}"
        template = f"data={s_arr}. data[{keys[s['query']]}]="
        # template = f"a={s['array']}\nQ: a[{s['query']}]\nA: "
        if not include_target:
            return template
        return template + f"{s['target']}"
    
    if len(e['few_shots']) == 0:
        task_prompt += to_qtext(e)
        e['prompt'] = task_prompt
    else:
        for s in e['few_shots']:
            task_prompt += to_qtext(s, True) + "\n"
        task_prompt += to_qtext(e)
        e['prompt'] = task_prompt
    return e

if __name__ == '__main__':
    from pathlib import Path
    print(generate_sample(10))
    output_dir = Path("datasets")
    ds = create_dataset()
    # ds.save_to_disk(str(output_dir / f"value_at_index_train/v0.1"))
    # ds.push_to_hub("nnheui/understanding-index-operation-v0.1", private=True)