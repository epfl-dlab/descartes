from transformers import AutoTokenizer, AutoModelWithLMHead, AutoConfig
from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers.models.mbart.modeling_mbart import MBartForConditionalGenerationBaseline, MBartFourDecodersConditional
from transformers import BertModel, BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding
import torch
import numpy as np
import logging
import os
import argparse
from pathlib import Path
logging.basicConfig(level=logging.ERROR)
np.random.seed(0)

def prepare_inputs(inputs, device):
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
            elif isinstance(v, dict):
                for key, val in v.items():
                    if isinstance(val, torch.Tensor):
                        v[key] = val.to(device)
                    elif isinstance(val, BatchEncoding) or isinstance(val, dict):
                        for k1,v1 in val.items():
                            if isinstance(v1, torch.Tensor):
                                val[k1] = v1.to(device)
        return inputs

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, help="directory where model is stored")
parser.add_argument("--bert_path", default=None, type=str, help="path to folder with bert model (for summary embeddings)")
parser.add_argument("--use_graph_embds", help="whether to use graph embeddings", action="store_true")
parser.add_argument("--languages", type=str, help="list of language codes")
parser.add_argument("--data_dir", type=str, help="directory to store the data")
parser.add_argument("--output_folder", type=str, help="path to the folder where to save outputs")
parser.add_argument("--baseline", help="whether to use baseline model", action="store_true")
parser.add_argument("--fourdecoders", help="whether to use four decoders model", action="store_true")
parser.add_argument("--randomization", help="whether to use randomize the choice of query in attention layer", action="store_true")
parser.add_argument("--graph_embd_length", type=int, default=128, help="length of graph embeddings")
parser.add_argument("--lang_to_test", type=str, default=None)
parser.add_argument("--mask_text", help="whether to mask input text", action="store_true")

args, uknown = parser.parse_known_args()

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)
config = AutoConfig.from_pretrained(args.output_dir)
config.graph_embd_length = args.graph_embd_length
if args.baseline:
    model = MBartForConditionalGenerationBaseline.from_pretrained(args.output_dir,config=config)
elif args.fourdecoders:
    model = MBartFourDecodersConditional.from_pretrained(args.output_dir,config=config)
else:
    model = MBartForConditionalGeneration.from_pretrained(args.output_dir, config=config)
tokenizer = MBartTokenizer.from_pretrained(args.output_dir)
if args.bert_path is not None:
    tokenizer_bert = BertTokenizer.from_pretrained(args.bert_path)
    bert_model = BertModel.from_pretrained(args.bert_path)
    model.model_bert = bert_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

outputs = []

sources = {}
lang_dict = {}
languages = args.languages.strip().split(',')
for lang in languages:
    lang_dict[lang[0:2]] = lang

sources = {}
targets = {}
    
#for lang, lang_code in lang_dict.items():
#    f = open(Path(args.data_dir).joinpath("test" + ".source" + lang), 'r', encoding='utf-8')
#    p = open(Path(args.data_dir).joinpath("test" + ".target" + lang), 'r', encoding='utf-8')
#    lines = f.readlines()
#    t_lines = p.readlines()
#    sources[lang] = lines
#    targets[lang] = t_lines
#    f.close()

f = open(Path(args.data_dir).joinpath("test" + ".embd"), 'r', encoding='utf-8')
embds = f.readlines()
print("length of embeddings: " + str(len(embds[0])))
f.close()

for lang, lang_code in lang_dict.items():
    try:
        f = open(Path(args.data_dir).joinpath("test" + ".source" + lang), 'r', encoding='utf-8')
        lines = f.readlines()
        sources[lang] = lines
        f.close()
    except:
        print("Missing source file " + str(lang) + ", creating an empty list")
        sources[lang] = ["no article\n"] * len(embds)
    try:
        p = open(Path(args.data_dir).joinpath("test" + ".target" + lang), 'r', encoding='utf-8')
        t_lines = p.readlines()
        targets[lang] = t_lines
        p.close()
    except:
        print("Missing target file " + str(lang) + ", creating an empty list")
        targets[lang] = ["no article\n"] * len(embds)

outputs = open(Path(args.output_folder).joinpath("outputs.txt"), 'w', encoding='utf-8')
target_file = open(Path(args.output_folder).joinpath("mod_targets.txt"), 'w', encoding='utf-8')
lang_file = open(Path(args.output_folder).joinpath("lang_list.txt"), 'w', encoding='utf-8')

for i in range(len(embds)):
    batch = {}
    batch_encodings = {}
    remaining_langs = []
    available_langs = []
    target_langs = []
    for lang, lang_code in lang_dict.items():
        txt = sources[lang][i].strip()

        if targets[lang][i].strip() != "no article":
            target_langs.append(lang)

        if txt == "no article":
            remaining_langs.append(lang)
            continue
        src_lang = lang_code
        tokenizer.src_lang = src_lang
        batch_enc = tokenizer([txt], padding=True, truncation=True)
        batch_encodings[lang] = batch_enc
        available_langs.append(lang)    
    input_ids = {}
    attention_mask = {}
    for key, val in batch_encodings.items():
        inputs = val["input_ids"]
        masks = val["attention_mask"]
        inputs = torch.tensor(inputs)
        masks = torch.tensor(masks)
        input_ids[key] = inputs
        attention_mask[key] = masks
    
    for lang in remaining_langs:
        input_ids[lang] = None
        attention_mask[lang] = None

    main_lang = None
    if args.randomization:
        main_lang = np.random.choice(available_langs,1)[0]

    #graph embeddings
    if args.use_graph_embds:
        embd = embds[i].strip()
        if embd == "0":
            embds_line = None
        else:
            embds_line = np.array([float(x) for x in embd.split()])
            embds_line = embds_line.astype(np.float32)
            embds_line = torch.tensor([embds_line])
    else:
        embds_line = None

    #summary embeddings
    if args.bert_path is not None:
        bert_inputs = {}
        for lang in target_langs:
            lang = lang[0:2]
            bert_outs = tokenizer_bert(
                [targets[lang][i].strip()],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            bert_inputs[lang] = bert_outs
    else:
        bert_inputs = None
                
    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_mask
    batch["graph_embeddings"] = embds_line
    batch["bert_inputs"] = bert_inputs

    batch = prepare_inputs(batch, device)
    if args.lang_to_test is None:
        for tgt_lang in target_langs:
            target_lang = lang_dict[tgt_lang]
            target = targets[tgt_lang][i]
            if args.bert_path is not None:
                bert_inputs_modified = bert_inputs.copy()
                bert_inputs_modified.pop(tgt_lang)
                batch["bert_inputs"] = bert_inputs_modified
            translated_tokens = model.generate(**batch, max_length=20, min_length=2, length_penalty=2.0, num_beams=4, early_stopping=True, target_lang = target_lang, decoder_start_token_id=tokenizer.lang_code_to_id[target_lang], baseline=args.baseline, mask_text=args.mask_text, main_lang=main_lang)
            output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            outputs.write(output+"\n")
            target_file.write(target)
            lang_file.write(target_lang+"\n")
    else:
        target_lang = args.lang_to_test
        tgt_lang = target_lang[0:2]
        target = targets[tgt_lang][i]
        if args.bert_path is not None:
            bert_inputs_modified = bert_inputs.copy()
            bert_inputs_modified.pop(tgt_lang)
            batch["bert_inputs"] = bert_inputs_modified
        translated_tokens = model.generate(**batch, max_length=20, min_length=2, length_penalty=2.0, num_beams=4, early_stopping=True, target_lang = target_lang, decoder_start_token_id=tokenizer.lang_code_to_id[target_lang], baseline=args.baseline, mask_text=args.mask_text,  main_lang=main_lang)
        output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        outputs.write(output+"\n")
        target_file.write(target)
        lang_file.write(target_lang+"\n")
    
outputs.close()
target_file.close()
lang_file.close()
