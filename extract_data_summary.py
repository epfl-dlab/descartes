import os
import argparse
import linecache

parser = argparse.ArgumentParser()
parser.add_argument("--languages", type=str, help="list of language codes")
parser.add_argument("--old_data", type=str, help="path to the full data directory")
parser.add_argument("--data_dir", type=str, help="path to save data")
parser.add_argument("--bert_path", type=str, default=None, help="whether to use graph embeddings")

args, uknown = parser.parse_known_args()
print(args)

if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir)
    
lang_dict = {}
languages = args.languages.strip().split(',')
for lang in languages:
    lang_dict[lang[0:2]] = lang

datasets = ["train","val","test"]

for dataset in datasets:
    qids_f = open(os.path.join(args.old_data, dataset+".qid"),'r')
    embd_f = os.path.join(args.old_data, dataset+".embd")
    langs = list(lang_dict.keys())
    src_files = {}
    tgt_files = {}
    source_files = {}
    target_files = {}

    for lang in langs:
        src_files[lang] = os.path.join(args.old_data, dataset + ".source" + lang)
        tgt_files[lang] = os.path.join(args.old_data, dataset + ".target" + lang)
        source_files[lang] = open(os.path.join(args.data_dir, dataset + ".source" + lang), 'w')
        target_files[lang] = open(os.path.join(args.data_dir, dataset + ".target" + lang), 'w')

    qids = qids_f.readlines()
    qids_f.close()

    f_embd = open(os.path.join(args.data_dir, dataset+".embd"), 'w')
    f_qid = open(os.path.join(args.data_dir, dataset+".qid"), 'w')

    for i in range(1,len(qids)+1):
        num_langs = 0
        for lang in langs:
            tgt = linecache.getline(str(tgt_files[lang]), i).rstrip("\n")
            if tgt != "no article":
                num_langs += 1
        if num_langs > 1:
            if args.bert_path is None:
                continue
        else:
            if args.bert_path is not None:
                continue
        for lang in langs:
            src = linecache.getline(str(src_files[lang]), i)
            tgt = linecache.getline(str(tgt_files[lang]), i)
            source_files[lang].write(src)
            target_files[lang].write(tgt)
        embd = linecache.getline(str(embd_f), i).rstrip("\n")
        f_embd.write(embd + "\n")
        f_qid.write(qids[i-1])
        
    
    for lang in langs:
        source_files[lang].close()
        target_files[lang].close()
    f_embd.close()
    f_qid.close()

    
        