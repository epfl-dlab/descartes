import os
import re
import gzip
import shutil
import argparse
import pickle
import numpy as np

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--languages", type=str, help="list of language codes")
parser.add_argument("--samples_train", type=int, help="number of samples in train set")
parser.add_argument("--samples_val", type=int, help="number of samples in validation set")
parser.add_argument("--samples_test", type=int, help="number of samples in test set")
parser.add_argument("--path", type=str, help="path to the main folder containing data")
parser.add_argument("--data_dir", type=str, help="directory to store the data")
parser.add_argument("--embd_path", type=str, help="path to pretrained graph embeddings")
parser.add_argument("--graph_path", type=str, help="path to wikidata graph relations")

args, uknown = parser.parse_known_args()

if os.path.exists(args.data_dir) and len(os.listdir(args.data_dir) != 0):
    print("Data already loaded")
    quit()

lang_dict = {}
languages = args.languages.strip().split(',')
for lang in languages:
    lang_dict[lang[0:2]] = lang

file = open(args.embd_path,'rb')
embd_file = pickle.load(file)
file.close()
cnt=0
mean_embd = np.zeros((128,))
for key,value in embd_file.items():
    mean_embd += value
    cnt += 1
avg_embd = mean_embd/cnt

f = open(args.graph_path, 'r')
lines = f.readlines()
f.close()

type_dict={}
for i in range(len(lines)):
    line = lines[i].split()
    qid_source = line[0].strip(">").split("/")[-1]
    relation = line[1].strip(">").split("/")[-1]
    qid_dest = line[2].strip(">").split("/")[-1]
    if relation=="P31" or relation=="P279":
        try:
            curr_list = type_dict[qid_source]
            curr_list.append(qid_dest)
            type_dict[qid_source] = curr_list
        except:
            type_dict[qid_source] = [qid_dest]
    else:
        continue
del lines
# calculate the embeddings
embd_dict = {}
for key, value in type_dict.items():
    embds = []
    for qid in value:
        try:
            embd = embd_file[qid]
            embds.append(embd)
        except:
            continue
    if len(embds) == 0:
        embd_dict[key] = avg_embd
    else:
        embedding = np.mean(np.array(embds), axis=0)
        embd_dict[key] = embedding

del type_dict

articles = {}
descriptions = {}
qids = {}
qid_dicts = {}

for lang, lang_code in lang_dict.items():
    path = os.path.join(args.path, lang_code)
    for file in os.listdir(path):
        if not file.endswith(".gz"):
            continue
        with gzip.open(os.path.join(path, file), 'rb') as s_file:
            for article in s_file:
                article = article.decode('utf-8')
                extract = re.search('"extract":"(.+?)","extract_html"', article)
                description = re.search('"description":"(.+?)","description_source"', article)
                qid = re.search('"wikibase_item":"(.+?)"', article)
                if qid is None:
                    continue
                #if description is None:
                #    continue
                if extract is None:
                    continue
                qid = qid.group(1)
                extract = extract.group(1)
                if description is not None:
                    description = description.group(1)
                try:
                    articles_lang = articles[lang]
                    articles_lang.append(extract)
                    articles[lang] = articles_lang
                    descriptions[lang].append(description)
                    qids[lang].append(qid)
                except:
                    articles[lang] = [extract]
                    descriptions[lang] = [description]
                    qids[lang] = [qid]
                
                    
    qid_dict = {}
    for i in range(len(qids[lang])):
        qid_dict[qids[lang][i]] = i
    qid_dicts[lang] = qid_dict

articles_fin = {}
descriptions_fin = {}
embds_fin = []
qids_fin = []

all_qids = []
for lang, qids_lang in qids.items():
    all_qids.extend(qids_lang)

np.random.shuffle(all_qids)

for qid in all_qids:
    desc_flag = True
    for lang, lang_code in lang_dict.items():
        try:
            index = qid_dicts[lang][qid]
        except:
            index = None
        if index is not None:
            extract = articles[lang][index]
            if descriptions[lang][index] is not None:
                description = descriptions[lang][index]
                desc_flag = False
            else:
                description = "no article"
        else:
            extract = "no article"
            description = "no article"
        
        try:
            articles_lang_list = articles_fin[lang]
            articles_lang_list.append(extract)
            articles_fin[lang] = articles_lang_list
            descriptions_fin[lang].append(description)
        except:
            articles_fin[lang] = [extract]
            descriptions_fin[lang] = [description]
    if desc_flag:
        # remove the sample with no descriptions
        for lang, lang_code in lang_dict.items():
            articles_fin[lang].pop()
            descriptions_fin[lang].pop()
        continue
    qids_fin.append(qid)

    try:
        embd = embd_dict[qid]
    except:
        embd = avg_embd
    if (embd==avg_embd).all():
        embd = [0]
    embds_fin.append(embd)

if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir)

for lang, lang_code in lang_dict.items():
    f1 = open(os.path.join(args.data_dir, "train.source" + lang), 'w', encoding='utf-8')
    f2 = open(os.path.join(args.data_dir, "train.target" + lang), 'w', encoding='utf-8')
    f3 = open(os.path.join(args.data_dir, "val.source" + lang), 'w', encoding='utf-8')
    f4 = open(os.path.join(args.data_dir, "val.target" + lang), 'w', encoding='utf-8')
    f5 = open(os.path.join(args.data_dir, "test.source" + lang), 'w', encoding='utf-8')
    f6 = open(os.path.join(args.data_dir, "test.target" + lang), 'w', encoding='utf-8')

    for i in range(args.samples_train):
	    f1.write(articles_fin[lang][i] + "\n")
	    f2.write(descriptions_fin[lang][i] + "\n")

    for i in range(args.samples_train, args.samples_train + args.samples_val):
        f3.write(articles_fin[lang][i]+ "\n")
        f4.write(descriptions_fin[lang][i] + "\n")

    for i in range(args.samples_train + args.samples_val, args.samples_train + args.samples_val + args.samples_test):
        f5.write(articles_fin[lang][i]+ "\n")
        f6.write(descriptions_fin[lang][i] + "\n")

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f6.close()

f = open(os.path.join(args.data_dir, "train.embd"), 'w', encoding='utf-8')
f1 = open(os.path.join(args.data_dir, "train.qid"), 'w', encoding='utf-8')
for i in range(args.samples_train):
    f.write(" ".join(str(item) for item in embds_fin[i]) + "\n")
    f1.write(qids_fin[i]  +"\n")
f.close()
f1.close()

f = open(os.path.join(args.data_dir, "val.embd"), 'w', encoding='utf-8')
f1 = open(os.path.join(args.data_dir, "val.qid"), 'w', encoding='utf-8')
for i in range(args.samples_train, args.samples_train + args.samples_val):
    f.write(" ".join(str(item) for item in embds_fin[i]) + "\n")
    f1.write(qids_fin[i] + "\n")
f.close()
f1.close()

f = open(os.path.join(args.data_dir, "test.embd"), 'w', encoding='utf-8')
f1 = open(os.path.join(args.data_dir, "test.qid"), 'w', encoding='utf-8')
for i in range(args.samples_train + args.samples_val, args.samples_train + args.samples_val + args.samples_test):
    f.write(" ".join(str(item) for item in embds_fin[i]) + "\n")
    f1.write(qids_fin[i] + "\n")
f.close()
f1.close()