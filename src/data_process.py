from tqdm import tqdm
from glob import glob

import argparse
import json
import random
import pickle
import os
import copy


def make_data(args, ont):
    random.seed(args.seed)
    
    jsons = glob(f"{args.multiwoz_dir}/*.json")
    
    class_dict = {
        "pass": 0,
        "non-pass": 1
    }

    data = {
        "train": {i: {"dialogue_ids": [], "src_states": [], "src_utters": [], "trg_utters": []} for i in range(len(class_dict))},
        "valid": {i: {"dialogue_ids": [], "src_states": [], "src_utters": [], "trg_utters": []} for i in range(len(class_dict))},
        "test": {i: {"dialogue_ids": [], "src_states": [], "src_utters": [], "trg_utters": []} for i in range(len(class_dict))}
    }
    for file in jsons:
        print(f"Processing {file}...")
        
        if file.split('/')[-1].startswith("train"):
            data_type = "train"
        elif file.split('/')[-1].startswith("dev"):
            data_type = "valid"
        elif file.split('/')[-1].startswith("test"):
            data_type = "test"
        
        with open(file, 'r') as f:
            dials = json.load(f)
            
        for obj in tqdm(dials):
            dialogue = obj['dialogue']
            dialogue_idx = obj['dialogue_idx']
            for turn in dialogue:
                utter = turn['transcript']
                turn_label = turn['turn_label']
                
                if len(turn_label) == 0:
                    continue
                
                # class 0
                data[data_type][0]["dialogue_ids"].append(dialogue_idx)
                data[data_type][0]["src_states"].append(turn_label)
                data[data_type][0]["src_utters"].append(utter)
                data[data_type][0]["trg_utters"].append(utter)
         
                slot_spans = []
                for pair in turn_label:
                    slot_type = pair[0]
                    slot_value = pair[1]
                    
                    if "|" in slot_value:
                        slot_values = slot_value.split("|")
                        for slot_value in slot_values:
                            for i in range(len(utter)):
                                if utter[i:i+len(slot_value)] == slot_value:
                                    slot_spans.append((i, i+len(slot_value), slot_type))
                                    break
                    else:
                        for i in range(len(utter)):
                            if utter[i:i+len(slot_value)] == slot_value:
                                slot_spans.append((i, i+len(slot_value), slot_type))
                                break                    
                
                if len(slot_spans) > 0:
                    # Slot value substitution
                    data[data_type][1]["dialogue_ids"].append(dialogue_idx)
                    data[data_type][1]["src_states"].append(turn_label)
                    data[data_type][1]["trg_utters"].append(utter)
                    
                    corrupted_utter = value_substitute(utter, slot_spans, args.slot_change_rate, ont)

                    data[data_type][1]["src_utters"].append(corrupted_utter)
                    
                    # Truncate slot part
                    data[data_type][1]["dialogue_ids"].append(dialogue_idx)
                    data[data_type][1]["src_states"].append(turn_label)
                    data[data_type][1]["trg_utters"].append(utter)
                    
                    corrupted_utter = trunc_slot_part(utter, slot_spans, args.cut_rate, args.max_window_size)
                    data[data_type][1]["src_utters"].append(corrupted_utter)
                    
                # Random truncation
                else:
                    data[data_type][1]["dialogue_ids"].append(dialogue_idx)
                    data[data_type][1]["src_states"].append(turn_label)
                    data[data_type][1]["trg_utters"].append(utter)
                    
                    corrupted_utter = trunc_random_part(utter, args.cut_rate)

                    data[data_type][1]["src_utters"].append(corrupted_utter)
                    
    
    return data, class_dict


def count_data(data):
    for i, obj in data.items():
        print(f"Class label: {i}")
        src_states = obj["src_states"]
        src_utters = obj["src_utters"]
        trg_utters = obj["trg_utters"]
        
        assert len(src_states) == len(src_utters)
        assert len(src_utters) == len(trg_utters)
        
        print(f"# of samples: {len(src_utters)}")


def find_new_value(slot_type, slot_value, ont):
    value_list = ont[slot_type]
    if slot_value in value_list:
        value_list = [value for value in value_list if value != slot_value]
    new_value = random.choice(value_list)
    
    return new_value


def value_substitute(utter, slot_spans, slot_change_rate, ont):
    sub_slot_spans = random.sample(slot_spans, max(1, int(len(slot_spans) * slot_change_rate)))
    for slot_span in sub_slot_spans:
        new_value = find_new_value(slot_span[2], utter[slot_span[0]:slot_span[1]], ont)
        corrupted_utter = utter[:slot_span[0]] + new_value + utter[slot_span[1]:]
        
    return corrupted_utter


def trunc_slot_part(utter, slot_spans, cut_rate, max_window_size):
    num_cut_spans = max(1, int(len(slot_spans) * cut_rate))
    cut_candids = random.sample(slot_spans, num_cut_spans)
    
    copied = utter
    for span in cut_candids:
        start, end = span[0], span[1]
        copied = copied[:start] + "[MASKED]" + copied[end:]
    
    copied_words = copied.split(' ')
    words = copy.deepcopy(copied_words)
    assert len(copied_words) == len(words)
    
    if len(words) <= 4 * max_window_size:
        window_size = max_window_size // 2
    else:
        window_size = max_window_size
    
    for w, word in enumerate(words):
        if word == "[MASKED]":
            if w-window_size >= 0:
                copied_words[w-window_size:w] = ["[MASKED]"] * window_size
            else:
                copied_words[:w] = ["[MASKED]"] * (w+1)
                
            copied_words[w+1:w+window_size+1] = ["[MASKED]"] * window_size
            
    removed = [word for word in copied_words if word != "[MASKED]"]
    
    return ' '.join(removed).strip()


def trunc_random_part(utter, cut_rate):
    words = utter.split(' ')
    cut_len = max(1, int(len(words) * cut_rate))
    start = random.sample(list(range(len(words))), 1)[0]
    removed = words[:start] + words[start+cut_len:]
    
    return ' '.join(removed)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help="The random seed.")
    parser.add_argument('--multiwoz_dir', type=str, default="trade-dst/data", help="The directory path for multiwoz data files.")
    parser.add_argument('--data_dir', type=str, default="data", help="The directory path to save pickle files.")                            
    parser.add_argument('--slot_change_rate', type=float, default=0.8, help="The ratio of changed slot part.")
    parser.add_argument('--cut_rate', type=float, default=0.3, help="The ratio of truncation.")
    parser.add_argument('--max_window_size', type=int, default=2, help="The maximum size of window when truncated.")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
    
    print("Loading ontology...")
    with open(f"{args.multiwoz_dir}/MultiWOZ_2.1/ontology.json", 'r') as f:
        ont = json.load(f)
    
    new_ont = {}
    for k, value_list in ont.items():
        domain = k.split('-')[0]
        slot_type = k.split('-')[-1].lower()
        
        new_value_list = []
        for value in value_list:
            if "|" in value:
                new_value_list += value.split("|")
            else:
                new_value_list.append(value)
        new_value_list = list(set(new_value_list))
        
        if 'book' in k:
            new_ont[f"{domain}-book {slot_type}"] = new_value_list
        else:
            new_ont[f"{domain}-{slot_type}"] = new_value_list
        
    print(new_ont.keys())
        
    data, class_dict = make_data(args, new_ont)
    
    train_data = data["train"]
    valid_data = data["valid"]
    test_data = data["test"]
    
    print("*"*50 + " Train Data Statistics " + "*"*50)
    count_data(train_data)
    print("*"*50 + " Validation Data Statistics " + "*"*50)
    count_data(valid_data)
    print("*"*50 + " Test Data Statistics " + "*"*50)
    count_data(test_data)
    
    print("Saving data files into pickle...")
    with open(f"{args.data_dir}/train.pickle", 'wb') as f:
        pickle.dump(train_data, f)
    with open(f"{args.data_dir}/valid.pickle", 'wb') as f:
        pickle.dump(valid_data, f)
    with open(f"{args.data_dir}/test.pickle", 'wb') as f:
        pickle.dump(test_data, f)
    with open(f"{args.data_dir}/class_dict.json", 'w') as f:
        json.dump(class_dict, f)
        
    print("Done.")
    