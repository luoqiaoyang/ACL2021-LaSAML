import os
import itertools
import collections
import json
from collections import defaultdict
import pandas as pd

import numpy as np
import torch
from torchtext.vocab import Vocab, Vectors

from embedding.avg import AVG
from embedding.cxtebd import CXTEBD
from embedding.wordebd import WORDEBD
import dataset.stats as stats
from dataset.utils import tprint, InputExample

from transformers import BertTokenizer

from nltk.tokenize import word_tokenize


def _get_20newsgroup_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
            'talk.politics.mideast': 0,
            'sci.space': 1,
            'misc.forsale': 2,
            'talk.politics.misc': 3,
            'comp.graphics': 4,
            'sci.crypt': 5,
            'comp.windows.x': 6,
            'comp.os.ms-windows.misc': 7,
            'talk.politics.guns': 8,
            'talk.religion.misc': 9,
            'rec.autos': 10,
            'sci.med': 11,
            'comp.sys.mac.hardware': 12,
            'sci.electronics': 13,
            'rec.sport.hockey': 14,
            'alt.atheism': 15,
            'rec.motorcycles': 16,
            'comp.sys.ibm.pc.hardware': 17,
            'rec.sport.baseball': 18,
            'soc.religion.christian': 19,
        }

    train_classes = []
    for key in label_dict.keys():
        if key[:key.find('.')] in ['sci', 'rec']:
            train_classes.append(label_dict[key])

    val_classes = []
    for key in label_dict.keys():
        if key[:key.find('.')] in ['comp']:
            val_classes.append(label_dict[key])

    test_classes = []
    for key in label_dict.keys():
        if key[:key.find('.')] not in ['comp', 'sci', 'rec']:
            test_classes.append(label_dict[key])

    return train_classes, val_classes, test_classes


def _get_amazon_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'Amazon_Instant_Video': 0,
        'Apps_for_Android': 1,
        'Automotive': 2,
        'Baby': 3,
        'Beauty': 4,
        'Books': 5,
        'CDs_and_Vinyl': 6,
        'Cell_Phones_and_Accessories': 7,
        'Clothing_Shoes_and_Jewelry': 8,
        'Digital_Music': 9,
        'Electronics': 10,
        'Grocery_and_Gourmet_Food': 11,
        'Health_and_Personal_Care': 12,
        'Home_and_Kitchen': 13,
        'Kindle_Store': 14,
        'Movies_and_TV': 15,
        'Musical_Instruments': 16,
        'Office_Products': 17,
        'Patio_Lawn_and_Garden': 18,
        'Pet_Supplies': 19,
        'Sports_and_Outdoors': 20,
        'Tools_and_Home_Improvement': 21,
        'Toys_and_Games': 22,
        'Video_Games': 23
    }

    train_classes = [2, 3, 4, 7, 11, 12, 13, 18, 19, 20]
    val_classes = [1, 22, 23, 6, 9]
    test_classes = [0, 5, 14, 15, 8, 10, 16, 17, 21]

    return train_classes, val_classes, test_classes


def _get_rcv1_classes(args):
    '''
        @return list of classes associated with each split
    '''

    train_classes = [1, 2, 12, 15, 18, 20, 22, 25, 27, 32, 33, 34, 38, 39,
                     40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                     54, 55, 56, 57, 58, 59, 60, 61, 66]
    val_classes = [5, 24, 26, 28, 29, 31, 35, 23, 67, 36]
    test_classes = [0, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 19, 21, 30, 37,
                    62, 63, 64, 65, 68, 69, 70]

    return train_classes, val_classes, test_classes


def _get_fewrel_classes(args):
    '''
        @return list of classes associated with each split
    '''
    # head=WORK_OF_ART validation/test split
    train_classes = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 19, 21,
                     22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                     39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 52, 53, 56, 57, 58,
                     59, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                     76, 77, 78]

    val_classes = [7, 9, 17, 18, 20]
    test_classes = [23, 29, 42, 47, 51, 54, 55, 60, 65, 79]

    return train_classes, val_classes, test_classes


def _get_huffpost_classes(args):
    '''
        @return list of classes associated with each split
    '''

    train_classes = list(range(20))
    val_classes = list(range(20,25))
    test_classes = list(range(25,41))

    return train_classes, val_classes, test_classes


def _get_reuters_classes(args):
    '''
        @return list of classes associated with each split
    '''

    train_classes = list(range(15))
    val_classes = list(range(15,20))
    test_classes = list(range(20,31))

    return train_classes, val_classes, test_classes

def _get_banking77_classes(args):
    all_class_randm_idx = np.random.permutation(list(range(77)))
    train_classes = all_class_randm_idx[:30]
    val_classes = all_class_randm_idx[30:45]
    test_classes = all_class_randm_idx[45:]
    print("train classes: " ,train_classes)
    print("val classes: " ,val_classes)
    print("test classes: " ,test_classes)
    return train_classes, val_classes, test_classes


def _get_clinc150_classes(args):
    if args.cross_domain:
        train_classes = list(range(15))
        val_classes = list(range(15))
        test_classes = list(range(15))
    else:
        all_class_randm_idx = np.random.permutation(list(range(150)))
        train_classes = all_class_randm_idx[:60]
        val_classes = all_class_randm_idx[60:75]
        test_classes = all_class_randm_idx[75:]

    return train_classes, val_classes, test_classes


def _get_clinc150_domains(args):
    if args.cross_domain:
        all_domain_randm_idx = np.random.permutation(list(range(10)))
        train_domains = all_domain_randm_idx[:4]
        val_domains = all_domain_randm_idx[4:5]
        test_domains = all_domain_randm_idx[5:]
    else:
        train_domains = [0]
        val_domains = [0]
        test_domains = [0]

    return train_domains, val_domains, test_domains

def _load_json(path):
    '''
        load data file
        @param path: str, path to the data file
        @return data: list of examples
    '''
    label = {}
    text_len = []
    with open(path, 'r', errors='ignore') as f:
        data = []
        for line in f:
            row = json.loads(line)

            # count the number of examples per label
            if int(row['label']) not in label:
                label[int(row['label'])] = 1
            else:
                label[int(row['label'])] += 1

            item = {
                'label': int(row['label']),
                'text': row['text'][:500]  # truncate the text to 500 tokens
            }

            text_len.append(len(row['text']))

            keys = ['head', 'tail', 'ebd_id']
            for k in keys:
                if k in row:
                    item[k] = row[k]

            data.append(item)

        tprint('Class balance:')

        print(label)

        tprint('Avg len: {}'.format(sum(text_len) / (len(text_len))))

        return data, label


def _read_words(data):
    '''
        Count the occurrences of all words
        @param data: list of examples
        @return words: list of words (with duplicates)
    '''
    words = []
    for example in data:
        words += example['text']
    return words


def _meta_split(args, all_data, train_classes, val_classes, test_classes):
    '''
        Split the dataset according to the specified train_classes, val_classes
        and test_classes

        @param all_data: list of examples (dictionaries)
        @param train_classes: list of int
        @param val_classes: list of int
        @param test_classes: list of int

        @return train_data: list of examples
        @return val_data: list of examples
        @return test_data: list of examples
    '''
    train_data, val_data, test_data = [], [], []

    if args.dataset == 'clinc150' and args.cross_domain:
        for example in all_data:
            if example['domain'] in args.train_domains:
                train_data.append(example)
            if example['domain'] in args.val_domains:
                val_data.append(example)
            if example['domain'] in args.test_domains:
                test_data.append(example)
    else:
        for example in all_data:
            if example['label'] in train_classes:
                train_data.append(example)
            if example['label'] in val_classes:
                val_data.append(example)
            if example['label'] in test_classes:
                test_data.append(example)

    return train_data, val_data, test_data


def _del_by_idx(array_list, idx, axis):
    '''
        Delete the specified index for each array in the array_lists

        @params: array_list: list of np arrays
        @params: idx: list of int
        @params: axis: int

        @return: res: tuple of pruned np arrays
    '''
    if type(array_list) is not list:
        array_list = [array_list]

    # modified to perform operations in place
    for i, array in enumerate(array_list):
        array_list[i] = np.delete(array, idx, axis)

    if len(array_list) == 1:
        return array_list[0]
    else:
        return array_list


def _data_to_nparray(data, vocab, args):
    '''
        Convert the data into a dictionary of np arrays for speed.
    '''
    doc_label = np.array([x['label'] for x in data], dtype=np.int64)

    if args.dataset == 'clinc150' and args.cross_domain: 
        doc_domain = np.array([x['domain'] for x in data], dtype=np.int64)

    raw = np.array([e['text'] for e in data], dtype=object)

    if args.bert:
        tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased', do_lower_case=True)

        # convert to wpe
        vocab_size = 0  # record the maximum token id for computing idf
        for e in data:
            e['bert_id'] = tokenizer.encode(" ".join(e['text']),
                                            add_special_tokens=True)
                                            # max_length=80)
            vocab_size = max(max(e['bert_id'])+1, vocab_size)

        text_len = np.array([len(e['bert_id']) for e in data])
        max_text_len = max(text_len)

        text = np.zeros([len(data), max_text_len], dtype=np.int64)

        del_idx = []
        # convert each token to its corresponding id
        for i in range(len(data)):
            text[i, :len(data[i]['bert_id'])] = data[i]['bert_id']

            # filter out document with only special tokens
            # unk (100), cls (101), sep (102), pad (0)
            if np.max(text[i]) < 103:
                del_idx.append(i)

        text_len = text_len

    else:
        # compute the max text length
        text_len = np.array([len(e['text']) for e in data])
        max_text_len = max(text_len)

        # initialize the big numpy array by <pad>
        text = vocab.stoi['<pad>'] * np.ones([len(data), max_text_len],
                                         dtype=np.int64)

        del_idx = []
        # convert each token to its corresponding id
        for i in range(len(data)):
            text[i, :len(data[i]['text'])] = [
                    vocab.stoi[x] if x in vocab.stoi else vocab.stoi['<unk>']
                    for x in data[i]['text']]

            # filter out document with only unk and pad
            if np.max(text[i]) < 2:
                del_idx.append(i)

        vocab_size = vocab.vectors.size()[0]

    text_len, text, doc_label, raw, doc_domain = _del_by_idx(
            [text_len, text, doc_label, raw, doc_domain], del_idx, 0)

    new_data = {
        'text': text,
        'text_len': text_len,
        'label': doc_label,
        'raw': raw,
        'vocab_size': vocab_size,
        'domain': doc_domain
    }

    if 'pos' in args.auxiliary:
        # use positional information in fewrel
        head = np.vstack([e['head'] for e in data])
        tail = np.vstack([e['tail'] for e in data])

        new_data['head'], new_data['tail'] = _del_by_idx(
            [head, tail], del_idx, 0)

    return new_data


def _split_dataset(data, finetune_split):
    """
        split the data into train and val (maintain the balance between classes)
        @return data_train, data_val
    """

    # separate train and val data
    # used for fine tune
    data_train, data_val = defaultdict(list), defaultdict(list)

    # sort each matrix by ascending label order for each searching
    idx = np.argsort(data['label'], kind="stable")

    non_idx_keys = ['vocab_size', 'classes2id', 'is_train']
    for k, v in data.items():
        if k not in non_idx_keys:
            data[k] = v[idx]

    # loop through classes in ascending order
    classes, counts = np.unique(data['label'], return_counts=True)
    start = 0
    for label, n in zip(classes, counts):
        mid = start + int(finetune_split * n)  # split between train/val
        end = start + n  # split between this/next class

        for k, v in data.items():
            if k not in non_idx_keys:
                data_train[k].append(v[start:mid])
                data_val[k].append(v[mid:end])

        start = end  # advance to next class

    # convert back to np arrays
    for k, v in data.items():
        if k not in non_idx_keys:
            data_train[k] = np.concatenate(data_train[k], axis=0)
            data_val[k] = np.concatenate(data_val[k], axis=0)

    return data_train, data_val

def _load_banking77_categories_json(path):
    with open(path, 'r', errors='ignor') as infile:
        categories = []
        all_data = json.load(infile)
        for line in all_data:
            categories.append(line)
            
    return categories

def _load_banking77_data_csv(args, categories, data_path1, data_path2):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    all_data1 = pd.read_csv(data_path1, delimiter=",")
    all_data2 = pd.read_csv(data_path2, delimiter=",")
    all_data = []
    domain_id = 0
    max_text_len = 0
    max_text_limit = args.max_text_len_limits
    for i in range(len(categories)):
        example = []
        for j in range(len(all_data1)):
            if all_data1['category'][j] == categories[i]:
                text_raw = all_data1['text'][j]
                text_token_id = tokenizer.encode(text_raw)
                label_raw = all_data1['category'][j].replace("?","").split("_")
                label_token_id = tokenizer.encode(label_raw, add_special_tokens=False)
                if len(text_token_id) > max_text_limit:
                    text_token_id = text_token_id[:max_text_limit]
                    text_token_id.append(102)
                max_text_len = len(text_token_id) + len(label_token_id) if len(text_token_id) + len(label_token_id) > max_text_len else max_text_len
                example.append(InputExample(
                    text_token_id = text_token_id,
                    text_raw = all_data1['text'][j], 
                    label_id = i,
                    label_token_id = label_token_id,
                    label_raw = label_raw,
                    domain_id = domain_id
                    ))
        for j in range(len(all_data2)):
            if all_data2['category'][j] == categories[i]:
                text_raw = all_data2['text'][j]
                text_token_id = tokenizer.encode(text_raw)
                label_raw = all_data2['category'][j].replace("?","").split("_")
                label_token_id = tokenizer.encode(label_raw, add_special_tokens=False)
                if len(text_token_id) > max_text_limit:
                    text_token_id = text_token_id[:max_text_limit]
                    text_token_id.append(102)
                max_text_len = len(text_token_id) + len(label_token_id)  if len(text_token_id) + len(label_token_id)  > max_text_len else max_text_len
                example.append(InputExample(
                    text_token_id = text_token_id,
                    text_raw = all_data1['text'][j], 
                    label_id = i,
                    label_token_id = label_token_id,
                    label_raw = label_raw,
                    domain_id = domain_id
                ))
        all_data.append(example)
    return all_data, max_text_len

def _load_clinc150_data_json(args, path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    max_text_limit = args.max_text_len_limits
    max_text_len = 0
    all_data = []
    all_labels = []
    #[3 2 4 7 0][1]
    banking = ["transfer","transactions","balance","freeze_account","pay_bill",
           "bill_balance","bill_due","interest_rate","routing","min_payment",
          "order_checks","pin_change","report_fraud","account_blocked","spending_history"]
    credit_cards = ["credit_score", "report_lost_card", "credit_limit", "rewards_balance", "new_card",
                    "application_status", "card_declined", "international_fees", "apr", "redeem_rewards",
                "credit_limit_change", "damaged_card", "replacement_card_duration","improve_credit_score", "expiration_date"]
    kitchen_dining = ["recipe", "restaurant_reviews", "calories", "nutrition_info", "restaurant_suggestion",
                    "ingredients_list", "ingredient_substitution", "cook_time", "food_last", "meal_suggestion",
                    "restaurant_reservation", "confirm_reservation", "how_busy", "cancel_reservation", "accept_reservations"]
    home = ["shopping_list", "shopping_list_update", "next_song", "play_music", "update_playlist",
        "todo_list", "todo_list_update", "calendar", "calendar_update", "what_song",
        "order", "order_status", "reminder", "reminder_update", "smart_home"]
    auto_commute = ["traffic", "directions", "gas", "gas_type", "distance",
                "current_location", "mpg", "oil_change_when", "oil_change_how", "jump_start",
                "uber", "schedule_maintenance", "last_maintenance", "tire_pressure", "tire_change"]
    travel = ["book_flight", "book_hotel", "car_rental", "travel_suggestion", "travel_alert",
            "travel_notification", "carry_on", "timezone", "vaccines", "translate",
            "flight_status", "international_visa", "lost_luggage", "plug_type", "exchange_rate"]
    utility = ["time", "alarm", "share_location", "find_phone", "weather",
            "text", "spelling", "make_call", "timer", "date",
            "calculator", "measurement_conversion", "flip_coin", "roll_dice", "definition"]
    work = ["direct_deposit", "pto_request", "taxes", "payday", "w2",
        "pto_balance", "pto_request_status", "next_holiday", "insurance", "insurance_change",
        "schedule_meeting", "pto_used", "meeting_schedule", "rollover_401k", "income"]
    small_talk = ["greeting", "goodbye", "tell_joke", "where_are_you_from", "how_old_are_you",
                "what_is_your_name", "who_made_you", "thank_you", "what_can_i_ask_you", "what_are_your_hobbies",
                "do_you_have_pets", "are_you_a_bot", "meaning_of_life", "who_do_you_work_for", "fun_fact"]
    meta = ["change_ai_name", "change_user_name", "cancel", "user_name", "reset_settings",
            "whisper_mode", "repeat", "no", "yes", "maybe",
        "change_language", "change_accent", "change_volume", "change_speed", "sync_device"]

    all_labels.extend(banking)
    all_labels.extend(credit_cards)
    all_labels.extend(kitchen_dining)
    all_labels.extend(home)
    all_labels.extend(auto_commute)
    all_labels.extend(travel)
    all_labels.extend(utility)
    all_labels.extend(work)
    all_labels.extend(small_talk)
    all_labels.extend(meta)

    with open(path, 'r', errors='ignor') as infile:
        all_data = json.load(infile)
        domain_list = []
        new_data = []
        for d in all_data:
            for _, data in enumerate(all_data[d]):                
                if d=="test" or d=="val" or d=="train":
                    if data[1] in banking:
                        domain_list.append("banking")
                        domain_id = 0
                    elif data[1] in credit_cards:
                        domain_list.append("credit cards")
                        domain_id = 1
                    elif data[1] in kitchen_dining:
                        domain_list.append("kitchen dining")
                        domain_id = 2
                    elif data[1] in home:
                        domain_list.append("home")
                        domain_id = 3
                    elif data[1] in auto_commute:
                        domain_list.append("auto commute")
                        domain_id = 4
                    elif data[1] in travel:
                        domain_list.append("travel")
                        domain_id = 5
                    elif data[1] in utility:
                        domain_list.append("utility")
                        domain_id = 6
                    elif data[1] in work:
                        domain_list.append("work")
                        domain_id = 7
                    elif data[1] in small_talk:
                        domain_list.append("small talk")
                        domain_id = 8
                    elif data[1] in meta:
                        domain_list.append("meta")
                        domain_id = 9
                    else:
                        print(data[1])
                    text_raw = data[0]
                    text_token_id = tokenizer.encode(text_raw)
                    if len(text_token_id) > max_text_limit:
                        text_token_id = text_token_id[:max_text_limit]
                        text_token_id.append(102)
                    label_raw = data[1].split("_")
                    label_token_id = tokenizer.encode(label_raw, add_special_tokens=False)
                    max_text_len = len(text_token_id)+len(label_token_id) if len(text_token_id)+len(label_token_id) > max_text_len else max_text_len
                    if args.embedding == "meta":
                        label_id = all_labels.index(data[1]) 
                        text = word_tokenize(text_raw)
                        data_dict = {
                        'label': label_id,
                        'domain': domain_id,
                        'text': text  # truncate the text to 500 tokens
                        }
                    else:
                        data_dict = {
                            "text_raw": text_raw,
                            "text_token_id": text_token_id,
                            "label_raw": label_raw,
                            "label_token_id": label_token_id,
                            "domain_id": domain_id
                        }
                    new_data.append(data_dict)

    return new_data, all_labels, max_text_len

def _clinc150_data_to_json(all_data):
    all_example_data = []
    tmp_example = []
    sorted_all_data = sorted(all_data, key = lambda i: i['label_raw'])

    tmp_id = 0
    tmp_label_id = 0
    for i in range(len(sorted_all_data)):
        if tmp_id < 150:
            tmp_id += 1
            tmp_example.append(InputExample(
                text_raw = sorted_all_data[i]['text_raw'],
                text_token_id = sorted_all_data[i]['text_token_id'],
                label_id = tmp_label_id,
                label_raw = sorted_all_data[i]['label_raw'],
                label_token_id = sorted_all_data[i]['label_token_id'],
                domain_id = sorted_all_data[i]['domain_id']
            ))
        if tmp_id == 150:
            tmp_id = 0
            tmp_label_id += 1
            all_example_data.append(tmp_example)
            tmp_example = []

    for i in range(150):
        for j in range(150):
            assert(all_example_data[i][0].label_id == all_example_data[i][j].label_id)
            assert(all_example_data[i][0].domain_id == all_example_data[i][j].domain_id)

    return all_example_data

def _load_raw_data(data_dir, args):
    alldata_InputExample = []
    # Step 1. Load raw data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    max_text_len = 0
    if args.dataset == "huffpost":
        class_label = ["POLITICS","WELLNESS","ENTERTAINMENT", "TRAVEL", "STYLE & BEAUTY","PARENTING", 
                "HEALTHY LIVING", "QUEER VOICES", "FOOD & DRINK", "BUSINESS", "COMEDY", "SPORTS", "BLACK VOICES", 
                "HOME & LIVING", "PARENTS", "THE WORLDPOST", "WEDDINGS", "WOMEN", "IMPACT", "DIVORCE", "CRIME", 
                "MEDIA","WEIRD NEWS","GREEN","WORLDPOST", "RELIGION","STYLE","SCIENCE","WORLD NEWS", "TASTE", 
                "TECH","MONEY","ARTS","FIFTY","GOOD NEWS","ARTS & CULTURE","ENVIRONMENT","COLLEGE", "LATINO VOICES", 
                "CULTURE & ARTS", "EDUCATION"]
        
        # Step 1. load all data from json
        all_data, all_labels = _load_json(data_dir)
        
        # Step 2. initialize an array list
        for i in range(len(all_labels)):
            empty_array = []
            alldata_InputExample.append(empty_array)
        
        # Step 3. construct InputExample data and divide data by domain and label
        # Huffpost data has one domain
        for i in range(len(all_data)):
            text_token_id = tokenizer.encode(all_data[i]['text'])
            text_raw = all_data[i]['text']
            label_id = all_data[i]['label']
            label_raw = class_label[label_id]
            label_token_id = tokenizer.encode(label_raw, add_special_tokens=False)
            max_text_len = len(text_token_id) if len(text_token_id) > max_text_len else max_text_len
            domain_id = 0 # if there are 1 domain only
            alldata_InputExample[label_id].append(InputExample(
                text_token_id = text_token_id,
                text_raw = text_raw,
                label_id = label_id,
                label_token_id = label_token_id,
                label_raw = label_raw,
                domain_id = domain_id
            ))    
    elif args.dataset == "banking77":
        all_labels = _load_banking77_categories_json(os.path.join(data_dir, 'categories.json'))
        alldata_InputExample, max_text_len = _load_banking77_data_csv(args, all_labels, os.path.join(data_dir, 'train.csv'), os.path.join(data_dir, 'test.csv'))
    elif args.dataset == "clinc150":
        all_data, all_labels, max_text_len = _load_clinc150_data_json(args, data_dir)
        alldata_InputExample = _clinc150_data_to_json(all_data)
    
    print("approximate max len of input: ", max_text_len)
    assert(args.num_classes == len(all_labels))   
    return alldata_InputExample

def _split_example_data(example_data,args):
    # train_data, val_data, test_data include: data, all_domain, all_label 
    train_data = []
    val_data = []
    test_data = []
    domain_train_data = []
    domain_val_data = []
    domain_test_data = []
    # whether split by domains or by classes
    if args.cross_domain:
        for i in range(len(args.train_domains)):
            empty_train_list =[]
            domain_train_data.append(empty_train_list)
        for i in range(len(args.val_domains)):
            empty_val_list = []
            domain_val_data.append(empty_val_list)
        for i in range(len(args.test_domains)):
            empty_test_list = []
            domain_test_data.append(empty_test_list)
        for i in range(len(example_data)):
            tmp_domain_id = example_data[i][0].domain_id
            if tmp_domain_id in args.train_domains:
                idx = np.where(args.train_domains == tmp_domain_id)[0].item()
                domain_train_data[idx].append(example_data[i])
            elif tmp_domain_id in args.val_domains:
                idx = np.where(args.val_domains == tmp_domain_id)[0].item()
                domain_val_data[idx].append(example_data[i])
            else:
                idx = np.where(args.test_domains == tmp_domain_id)[0].item()
                domain_test_data[idx].append(example_data[i])
        train_data.extend(domain_train_data)
        val_data.extend(domain_val_data)
        test_data.extend(domain_test_data)
        assert (len(train_data) != 1)
        assert (len(test_data) != 1)
    else:
        for i in range(len(example_data)):
            if i in args.train_classes:
                domain_train_data.append(example_data[i])
            elif i in args.val_classes:
                domain_val_data.append(example_data[i])
            else:
                domain_test_data.append(example_data[i])
        train_data.append(domain_train_data)
        val_data.append(domain_val_data)
        test_data.append(domain_test_data)
    # import pdb; pdb.set_trace()
    return train_data, val_data, test_data

def _load_data_to_alldata(data_dir, args):
    all_data = []
    if args.dataset == 'banking77':
        categories = _load_banking77_categories_json(os.path.join(data_dir, 'categories.json'))
        data_path1 = os.path.join(data_dir, 'train.csv')
        data_path2 = os.path.join(data_dir, 'test.csv')
        all_data1 = pd.read_csv(data_path1, delimiter=",")
        all_data2 = pd.read_csv(data_path2, delimiter=",")
        all_data = []
        for i in range(len(categories)):
            for j in range(len(all_data1)):
                if all_data1['category'][j] == categories[i]:
                    label = i
                    text = word_tokenize(all_data1['text'][j])
                    item = {
                    'label': label,
                    'text': text  # truncate the text to 500 tokens
                    }
                    all_data.append(item)
            for j in range(len(all_data2)):
                if all_data2['category'][j] == categories[i]:
                    label = i
                    text = word_tokenize(all_data2['text'][j])
                    item = {
                    'label': label,
                    'text': text  # truncate the text to 500 tokens
                    }
                    all_data.append(item)
    elif args.dataset == 'clinc150':
        all_data, _, _ = _load_clinc150_data_json(args, data_dir)
        return all_data
    return all_data

def load_dataset(args):
    if args.dataset == '20newsgroup':
        train_classes, val_classes, test_classes = _get_20newsgroup_classes(args)
    elif args.dataset == 'amazon':
        train_classes, val_classes, test_classes = _get_amazon_classes(args)
    elif args.dataset == 'fewrel':
        train_classes, val_classes, test_classes = _get_fewrel_classes(args)
    elif args.dataset == 'huffpost':
        train_classes, val_classes, test_classes = _get_huffpost_classes(args)
    elif args.dataset == 'reuters':
        train_classes, val_classes, test_classes = _get_reuters_classes(args)
    elif args.dataset == 'rcv1':
        train_classes, val_classes, test_classes = _get_rcv1_classes(args)
    elif args.dataset == 'banking77':
        train_classes, val_classes, test_classes = _get_banking77_classes(args)
    elif args.dataset == 'clinc150':
        train_classes, val_classes, test_classes = _get_clinc150_classes(args)
    else:
        raise ValueError(
            'args.dataset should be one of'
            '[20newsgroup, amazon, fewrel, huffpost, reuters, rcv1]')

    # assert(len(train_classes) == args.n_train_class)
    # assert(len(val_classes) == args.n_val_class)
    # assert(len(test_classes) == args.n_test_class)

    if args.dataset == "clinc150":
        train_domains, val_domains, test_domains = _get_clinc150_domains(args)
    else:
        train_domains = [0]
        val_domains = [0]
        test_domains = [0]

    args.train_classes = train_classes
    args.val_classes = val_classes
    args.test_classes = test_classes
    args.num_classes = args.n_train_class + args.n_val_class + args.n_test_class
    args.num_domain = 1 if args.n_train_domain == 1 else args.n_train_domain + args.n_val_domain + args.n_test_domain
    args.train_domains = train_domains
    args.val_domains = val_domains
    args.test_domains = test_domains
    print("train_domains: ", train_domains)
    print("val_domains: ", val_domains)
    print("test_domains: ", test_domains)


    if args.mode == 'finetune':
        # in finetune, we combine train and val for training the base classifier
        train_classes = train_classes + val_classes
        args.n_train_class = args.n_train_class + args.n_val_class
        args.n_val_class = args.n_train_class

    tprint('Loading data')
    
    if args.meta_w_target==False:
        vocab = None
        example_data = _load_raw_data(args.data_path, args)
        train_data, val_data, test_data = _split_example_data(example_data, args)
        # import pdb; pdb.set_trace()
        return train_data, val_data, test_data, vocab

    if args.dataset == "huffpost":
        all_data, _ = _load_json(args.data_path)
    else:
        all_data = _load_data_to_alldata(args.data_path, args)

    tprint('Loading word vectors')
    path = os.path.join(args.wv_path, args.word_vector)
    if not os.path.exists(path):
        # Download the word vector and save it locally:
        tprint('Downloading word vectors')
        import urllib.request
        urllib.request.urlretrieve(
            'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec',
            path)

    vectors = Vectors(args.word_vector, cache=args.wv_path)
    vocab = Vocab(collections.Counter(_read_words(all_data)), vectors=vectors,
                  specials=['<pad>', '<unk>'], min_freq=5)

    # print word embedding statistics
    wv_size = vocab.vectors.size()
    tprint('Total num. of words: {}, word vector dimension: {}'.format(
        wv_size[0],
        wv_size[1]))

    num_oov = wv_size[0] - torch.nonzero(
            torch.sum(torch.abs(vocab.vectors), dim=1)).size()[0]
    tprint(('Num. of out-of-vocabulary words'
           '(they are initialized to zeros): {}').format( num_oov))
    

    # Split into meta-train, meta-val, meta-test data
    train_data, val_data, test_data = _meta_split(
            args, all_data, train_classes, val_classes, test_classes)
    tprint('#train {}, #val {}, #test {}'.format(
        len(train_data), len(val_data), len(test_data)))
    # import pdb; pdb.set_trace()

    # Convert everything into np array for fast data loading
    train_data = _data_to_nparray(train_data, vocab, args)
    val_data = _data_to_nparray(val_data, vocab, args)
    test_data = _data_to_nparray(test_data, vocab, args)
    # import pdb; pdb.set_trace()
    train_data['is_train'] = True
    # this tag is used for distinguishing train/val/test when creating source pool

    stats.precompute_stats(train_data, val_data, test_data, args)

    if args.meta_w_target:
        # augment meta model by the support features
        if args.bert:
            ebd = CXTEBD(args.pretrained_bert,
                         cache_dir=args.bert_cache_dir,
                         finetune_ebd=False,
                         return_seq=True)
        else:
            ebd = WORDEBD(vocab, finetune_ebd=False)

        train_data['avg_ebd'] = AVG(ebd, args)
        if args.cuda != -1:
            train_data['avg_ebd'] = train_data['avg_ebd'].cuda(args.cuda)

        val_data['avg_ebd'] = train_data['avg_ebd']
        test_data['avg_ebd'] = train_data['avg_ebd']

    # if finetune, train_classes = val_classes and we sample train and val data
    # from train_data
    if args.mode == 'finetune':
        train_data, val_data = _split_dataset(train_data, args.finetune_split)

    return train_data, val_data, test_data, vocab
