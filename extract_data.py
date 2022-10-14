import os
import argparse
import linecache

parser = argparse.ArgumentParser()
parser.add_argument("--languages", type=str, help="list of language codes")
parser.add_argument("--old_data", type=str, help="path to the full data directory")
parser.add_argument("--data_dir", type=str, help="path to save data")
parser.add_argument("--use_graph_embds", help="whether to use graph embeddings", action="store_true")

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
        embd = linecache.getline(str(embd_f), i).rstrip("\n")
        if embd == "0":
            if args.use_graph_embds:
                continue
        else:
            if not args.use_graph_embds:
                continue
            else:
                f_embd.write(embd + "\n")
        f_qid.write(qids[i-1])
        for lang in langs:
            src = linecache.getline(str(src_files[lang]), i)
            tgt = linecache.getline(str(tgt_files[lang]), i)
            source_files[lang].write(src)
            target_files[lang].write(tgt)
    
    for lang in langs:
        source_files[lang].close()
        target_files[lang].close()
    f_embd.close()
    f_qid.close()

    
        
