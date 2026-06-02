import os
import csv
import torch
import numpy as np
from tqdm import tqdm

def save_processed_sequences_csv_split(users_train, users_valid, users_test, out_dir, prefix="processed_sequences"):
    """
    分别把train, valid, test保存为不同csv
    """
    os.makedirs(out_dir, exist_ok=True)
    path_train = os.path.join(out_dir, f"{prefix}_train.csv")
    path_valid = os.path.join(out_dir, f"{prefix}_valid.csv")
    path_test  = os.path.join(out_dir, f"{prefix}_test.csv")

    # train
    with open(path_train, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "item_sequence"])
        for user_id, seq in users_train.items():
            writer.writerow([user_id, " ".join(str(i) for i in seq)])
    print(f"train写入: {path_train}")

    # valid
    with open(path_valid, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "item_sequence"])
        for user_id, seq in users_valid.items():
            writer.writerow([user_id, " ".join(str(i) for i in seq)])
    print(f"valid写入: {path_valid}")

    # test
    with open(path_test, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "item_sequence"])
        for user_id, seq in users_test.items():
            writer.writerow([user_id, " ".join(str(i) for i in seq)])
    print(f"test写入: {path_test}")

def read_items(args):
    item_id_to_keys = {}
    item_name_to_id = {}
    for i in range(args.min_video_no, args.max_video_no + 1):
        image_name = str(i)
        item_id = i
        item_name_to_id[image_name] = item_id
        item_id_to_keys[item_id] = image_name
    return item_id_to_keys, item_name_to_id

def read_behaviors(before_item_id_to_keys, before_item_name_to_id, Log_file, args, local_rank=0):
    behaviors_path = os.path.join(args.root_data_dir, args.dataset, args.behaviors)
    max_seq_len, min_seq_len = args.max_seq_len, args.min_seq_len

    # Log_file.info('##### item number {}'.format(len(before_item_id_to_keys)))
    # Log_file.info('##### min seq len {}, max seq len {}#####'.format(min_seq_len, max_seq_len))

    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    before_seq_num = 0
    pairs_num = 0
    # Log_file.info('rebuild user seqs...')
    with open(behaviors_path, 'r') as f:
        for line in f:
            before_seq_num += 1
            splited = line.strip('\n').split('\t')
            user_id = splited[0]
            history_item_name = str(splited[1]).strip().split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[str(i)] for i in history_item_name]
            user_seq_dic[user_id] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1

    # Log_file.info("##### pairs_num {}".format(pairs_num))
    # Log_file.info('##### user seqs before {}'.format(before_seq_num))

    item_id = 1
    item_id_to_keys = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_keys[item_id] = before_item_id_to_keys[before_item_id]
            item_id += 1

    item_num = len(item_id_before_to_now)
    Log_file.info('##### items after clearing {}, {}, {}, {}#####'.format(item_num, item_id - 1, len(item_id_to_keys), len(item_id_before_to_now)))
    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    new_user_seq_dic = {}

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        new_user_seq_dic[user_name] = user_seq

        train = user_seq[:-2]
        valid = user_seq[-(max_seq_len+2):-1]
        test = user_seq[-(max_seq_len+1):]

        users_train[user_id] = train
        users_valid[user_id] = valid
        users_test[user_id] = test

        users_history_for_valid[user_id] = torch.LongTensor(np.array(train))
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1

    return item_num, item_id_to_keys, users_train, users_valid, users_history_for_valid, users_test, users_history_for_test
