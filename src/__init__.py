# coding=utf-8

""" MONPA: Multi-Objective NER POS Annotator for Chinese
https://github.com/monpa-team/monpa

See our paper for details.
Hsieh, Y. L., Chang, Y. C., Huang, Y. J., Yeh, S. H., Chen, C. H., Hsu, W. L. (2017, November).
MONPA: Multi-objective Named-entity and Part-of-speech Annotator for Chinese using Recurrent Neural Network.
In Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 2: Short Papers) (pp. 80-85).
"""

import logging
import io
import os
import hashlib
from argparse import Namespace

try:
    import torch
except ImportError:
    print("monpa 需要 pytorch >= 1.0 才能使用，請確定您已安裝。\nmonpa requires pytorch >= 1.0 to work. Please make sure you have it.")
    raise

from .configuration_albert import AlbertConfig
from .modeling_albert import AlbertForMONPA
from .tokenization import FullTokenizer, convert_to_unicode

__version__ = "0.3.1"

print("+---------------------------------------------------------------------+")
print("  Welcome to MONPA: Multi-Objective NER POS Annotator for Chinese")
print("+---------------------------------------------------------------------+")

_userdict = []
_model = None
_in_tokenizer = None
_out_tokenizer = None
_args = None
_config = None

path_monpa_package = os.path.dirname(os.path.abspath(__file__))
path_albert_config_file = os.path.join(path_monpa_package, 'albert_config', 'albert_config_mo.json')
path_vocab_file = os.path.join(path_monpa_package, 'vocab_monpa.vocab')
path_output_vocab_file = os.path.join(path_monpa_package, 'pos.tgt.dict')
path_model = os.path.join(path_monpa_package, 'model', 'monpa_model_8000.pt')

is_initialized = False

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s', \
                        datefmt = '%m/%d/%Y %H:%M:%S', \
                        level = logging.INFO)
logger = logging.getLogger(__name__)

def initialize():
    ''' Create arguments, model, and load weights
    '''

    global is_initialized
    if is_initialized:
        return
    global _model
    global _args
    global _config
    global logger
    global _in_tokenizer
    global _out_tokenizer

    _args = Namespace(vocab_file=path_vocab_file, \
                     output_vocab_file=path_output_vocab_file, \
                     config_name=path_albert_config_file, \
                     do_lower_case=False, \
                     no_logs=True, \
                     max_seq_length=200, \
                     model=path_model, \
                     cuda=False, \
                     device=torch.device("cpu"), \
                     )
    if _args.no_logs:
        logger.setLevel(logging.WARNING)

    logger.info("running on device %s", _args.device)
    if _in_tokenizer == None:
        _in_tokenizer = FullTokenizer(vocab_file=_args.vocab_file, do_lower_case=_args.do_lower_case)
    if _out_tokenizer == None:
        _out_tokenizer = FullTokenizer(vocab_file=_args.output_vocab_file, do_lower_case=_args.do_lower_case)
    out_vocab = _out_tokenizer.vocab
    id_to_pos = dict([(i,w) for (w,i) in out_vocab.items()])

    label_list = out_vocab.keys()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    _config = AlbertConfig.from_pretrained(
        _args.config_name,
        num_labels=num_labels,
    )
    _config.__dict__["tag_id_to_name"] = id_to_pos
    _model = get_model(_config, _args)
    is_initialized = True
    return

def use_gpu(gpu_on):
    global _args
    if gpu_on == True:
        if torch.cuda.is_available():
            _args.cuda = True
            _args.device = torch.device("cuda")
        else:
            print("GPU unavailable.")
    if gpu_on == False:
        _args.cuda = False
        _args.device = torch.device("cpu")

def get_model(config, args):
    ''' Try to find model file and load weights
    '''

    correct_hash = '171a87641bafe3b1c83331be67f81ce0cb724b84d52c02465683da52b88c697a'
    found_model_file = False
    if os.path.exists(path_model):
        if correct_hash == get_file_hash(path_model):
            print("已找到 model檔。Found model file.")
        else:
            print("Warning: found model file with a different hashcode. It may be corrupted.")
        found_model_file = True

    if not found_model_file:
        raise ImportError("找不到必要的 model 檔。搜尋位置 {}\nCannot find model file at {}.".format(
                            path_model, path_model))

    model = AlbertForMONPA(config=config)
    model = load_checkpoint(model, args.model)

    logger.info("Model parameters %s", args)
    model.eval()
    return model

def load_checkpoint(module, checkpoint_filename):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    state_dict = torch.load(checkpoint_filename, map_location="cpu")
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")
    load(module)
    return module

def get_file_hash(filename):
    sha256_hash = hashlib.sha256()
    with io.open(filename, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    got_hash = sha256_hash.hexdigest()
    return got_hash

def to_CoNLL_format(org_input, predicted_pos):
    conll_formatted = []
    segmented_words = []
    pos_tags = []
    temp_word = ''
    for pos_id, pos_name in enumerate(predicted_pos):
        if pos_id >= len(org_input):
            break
        split_pos_names = pos_name.split('-')
        if len(split_pos_names) == 3:
            bound, _, detailed_pos = split_pos_names
        elif len(split_pos_names) == 2:
            bound, detailed_pos = split_pos_names
        else:
            break
        bound = bound.lower()
        if bound == 's':
            pos_tags.append(detailed_pos)
            segmented_words.append(org_input[pos_id])
            continue
        temp_word += org_input[pos_id]
        if bound == 'b':
            pos_tags.append(detailed_pos)
        if bound == 'e':
            segmented_words.append(temp_word)
            temp_word = ''
    if len(temp_word) > 0: # special case: no 'E-' in predicted_pos
        segmented_words.append(temp_word)
        temp_word = ''
    assert len(segmented_words) == len(pos_tags), \
           "lengths {} (words) and\n {} (pos tags) mismatch".format( \
                 segmented_words, pos_tags)
    for w, p in zip(segmented_words, pos_tags):
        conll_formatted.append((w, p))
    return conll_formatted, segmented_words, pos_tags

def query_model(model, input_text_tokens, args):
    global _in_tokenizer
    global _out_tokenizer

    # only support cutting one sentence now
    input_text_tokens = input_text_tokens[0]
    input_text_tokens = input_text_tokens.strip()
    input_text_tokens = _in_tokenizer.tokenize(input_text_tokens)
    if len(input_text_tokens) < 1:
        return [], [], []
    if len(input_text_tokens) > args.max_seq_length:
        input_text_tokens = input_text_tokens[:args.max_seq_length]

    input_text_tokens = ["[CLS]"] + input_text_tokens + ["[SEP]"]
    segment_ids = [0] * len(input_text_tokens)
    input_mask = [1] * len(input_text_tokens)
    # word to ID
    input_ids = _in_tokenizer.convert_tokens_to_ids(input_text_tokens)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.tensor([input_mask], dtype=torch.long)
    token_type_ids = torch.tensor([segment_ids], dtype=torch.long)
    # to device CPU or GPU
    input_ids = input_ids.to(args.device)
    attention_mask = attention_mask.to(args.device)
    token_type_ids = token_type_ids.to(args.device)
    model.to(args.device)

    with torch.no_grad():
        predictions = model(input_ids, attention_mask, token_type_ids)
    predictions = predictions[0][0]
    predictions = _out_tokenizer.convert_ids_to_tokens(predictions)
    conll_formatted, segmented_words, pos_tags = to_CoNLL_format(input_text_tokens[1:-1], predictions[1:-1])

    return conll_formatted, segmented_words, pos_tags

def load_userdict(pathtofile):
    ''' load user dictionary function
    '''

    global _userdict
    # empty previous userdict
    _userdict = []
    for input_item in io.open(pathtofile, 'r', encoding="utf-8").read().split("\n"):
        item = input_item.split(" ")
        if len(item[0].strip()) == 0:
            continue
        _userdict.append(item)

def findall(p, s):
    ''' Yields all the positions of the pattern p in the string s
    '''

    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i+1)

def cut_wo_userdict(text):
    global _args
    try:
        sentence = text
    except EOFError:
        return
    conll_formatted, segmented_words, pos_tags = query_model(_model, [sentence], _args)

    return segmented_words

def cut_w_userdict(text):
    ''' segment within user defined dict and return terms without POS
    '''
    global _args
    global _userdict
    from operator import itemgetter
    # find user defined dict terms' position within sentence or not
    userdict_in_sentence = []
    text_temp = text
    for term_dict in _userdict:
        for term_index in findall(term_dict[0], text_temp):
            text_temp = text_temp.replace(term_dict[0], "＃" * len(term_dict[0]))
            if term_dict[0] == '':break
            userdict_in_sentence.append((term_index, term_dict[0]))

    j = 0
    sentence_list = []
    userdict_in_sentence_sorted = sorted(userdict_in_sentence, key=itemgetter(0))

    for term in userdict_in_sentence_sorted:

        if term[0] == j:
            # print(term)
            sentence_list.append(term[1])
            j = j + len(term[1])
            if len(userdict_in_sentence_sorted) - (userdict_in_sentence_sorted.index(term) + 1) == 0:
                sentence = text[j:]
                conll_formatted, segmented_words, pos_tags = query_model(_model, [sentence], _args)
                out_str = segmented_words
                sentence_list.extend(out_str)
                break
        else:
            sentence = text[j: term[0]]

            conll_formatted, segmented_words, pos_tags = query_model(_model, [sentence], _args)
            out_str = segmented_words
            sentence_list.extend(out_str)
            sentence_list.append(term[1])
            j = term[0] + len(term[1])
            if len(userdict_in_sentence_sorted) - (userdict_in_sentence_sorted.index(term) + 1) == 0:
                sentence = text[j:]
                conll_formatted, segmented_words, pos_tags = query_model(_model, [sentence], _args)
                out_str = segmented_words
                sentence_list.extend(out_str)
                break
    return sentence_list


def pseg_wo_userdict(text):
    global _args
    try:
        sentence = text
    except EOFError:
        return

    conll_formatted, segmented_words, pos_tags = query_model(_model, [sentence], _args)

    return conll_formatted

def pseg_w_userdict(text):
    global _args
    global _userdict
    from operator import itemgetter
    # find user defined dict terms' position within sentence or not
    userdict_in_sentence = []
    text_temp = text
    for term_dict in _userdict:
        for term_index in findall(term_dict[0], text_temp):
            text_temp = text_temp.replace(term_dict[0], "＃" * len(term_dict[0]))
            if term_dict[0] == '':break
            userdict_in_sentence.append((term_index, term_dict[0], term_dict[2]))

    j = 0
    sentence_list = []
    userdict_in_sentence_sorted = sorted(userdict_in_sentence, key=itemgetter(0))
    # print(userdict_in_sentence_sorted)
    for term in userdict_in_sentence_sorted:

        if term[0] == j:
            # print(term)
            sentence_list.append((term[1], term[2]))
            j = j+ len(term[1])
            if len(userdict_in_sentence_sorted) - (userdict_in_sentence_sorted.index(term) + 1) == 0:
                sentence = text[j:]
                conll_formatted, segmented_words, pos_tags = query_model(_model, [sentence], _args)
                out_str = conll_formatted
                sentence_list.extend(out_str)
                break
        else:
            sentence = text[j: term[0]]
            conll_formatted, segmented_words, pos_tags = query_model(_model, [sentence], _args)
            out_str = conll_formatted
            sentence_list.extend(out_str)
            sentence_list.append((term[1], term[2]))
            j = term[0] + len(term[1])
            if len(userdict_in_sentence_sorted) - (userdict_in_sentence_sorted.index(term) + 1) == 0:
                sentence = text[j:]
                conll_formatted, segmented_words, pos_tags = query_model(_model, [sentence], _args)
                out_str = conll_formatted
                sentence_list.extend(out_str)
                break
    return sentence_list

def cut(text):
    text = convert_to_unicode(text)
    if _userdict == []:
        return cut_wo_userdict(text)
    else:
        if any(x[0] in text for x in _userdict):
            return cut_w_userdict(text)
        else:
            return cut_wo_userdict(text)

def pseg(text):
    text = convert_to_unicode(text)
    if _userdict == []:
        return pseg_wo_userdict(text)
    else:
        if any(x[0] in text for x in _userdict):
            return pseg_w_userdict(text)
        else:
            return pseg_wo_userdict(text)

def cut_batch(text_list):
    global _model
    global _args
    #TODO: add userdict support
    _, words, _ = query_model_batch(_model, text_list, _args)
    return words

def pseg_batch(text_list):
    global _model
    global _args
    #TODO: add userdict support
    conll_fmt, _, _ = query_model_batch(_model, text_list, _args)
    return conll_fmt

def query_model_batch(model, input_text_list, args):
    if not isinstance(input_text_list, list):
        input_text_list = [input_text_list]
    global _in_tokenizer
    global _out_tokenizer
    global _model
    input_text_list = [convert_to_unicode(t.strip()) for t in input_text_list]
    input_text_tokens = [_in_tokenizer.tokenize(t) for t in input_text_list]
    input_text_tokens = [t[:args.max_seq_length] for t in input_text_tokens]

    input_text_tokens = [["[CLS]"] + t + ["[SEP]"] for t in input_text_tokens]
    input_mask = [[1] * len(t) for t in input_text_tokens]
    # padding
    max_len = min(args.max_seq_length + 2, # add two special symbols \
                  max(len(t) for t in input_text_tokens))
    input_text_tokens = [t + ["[PAD]"] * (max_len - len(t)) for t in input_text_tokens]
    input_mask = [t + [0] * (max_len - len(t)) for t in input_mask]
    # word to ID
    input_ids = [_in_tokenizer.convert_tokens_to_ids(t) for t in input_text_tokens]
    # to tensor
    attention_mask = torch.tensor(input_mask, dtype=torch.long)
    token_type_ids = torch.tensor([[0] * max_len for _ in input_ids], dtype=torch.long)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    
    input_ids = input_ids.to(args.device)
    attention_mask = attention_mask.to(args.device)
    token_type_ids = token_type_ids.to(args.device)
    model.to(args.device)

    with torch.no_grad():
        predictions = model(input_ids, attention_mask, token_type_ids)
        predictions = predictions[0]
        predictions = [_out_tokenizer.convert_ids_to_tokens(pd) for pd in predictions]
    
    conll_formatted = []
    segmented_words = []
    pos_tags = []
    for tt, pp in zip(input_text_tokens, predictions):
        tt = tt[1:-1]
        pp = pp[1:-1]
        cf, sw, po = to_CoNLL_format(tt, pp)
        conll_formatted.append(cf)
        segmented_words.append(sw)
        pos_tags.append(po)

    return conll_formatted, segmented_words, pos_tags

if not is_initialized:
    initialize()
