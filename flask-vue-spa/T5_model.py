import collections.abc as container_abcs
import torch
from torch.utils.data import DataLoader, Dataset
import re
import argparse
from tqdm.auto import tqdm
import numpy as np

from Tokenizer import KeyDataset, MTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(data_line):
    data_src = data_line
    D = [re.sub('\s|\t|\n', '', data_src)]
    return D


def create_data(data, tokenizer, max_len):
    """调用tokenizer.encode编码正文/标题，每条样本用dict表示数据域
    """
    ret, title = [], None
    for content in data:
        if type(content) == tuple:
            title, content = content
        text_ids = tokenizer.encode(content, max_length=max_len,
                                    truncation='only_first')

        features = {'input_ids': text_ids,
                    'attention_mask': [1] * len(text_ids),
                    'raw_data': content}
        if title:
            features['title'] = title
        ret.append(features)
    return ret


def sequence_padding(inputs, length=None, padding=0):
    """
    Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')


def default_collate(batch):
    """组batch
    各个数据域分别转换为tensor，tensor第一个维度等于batch_size
    """
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out).to(device)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            batch = sequence_padding(batch)

        return default_collate([default_collate(elem) for elem in batch])

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def prepare_data(test,args, tokenizer):
    """准备batch数据
    """
    test_data = load_data(test)
    test_data = create_data(test_data, tokenizer, args.max_len)
    test_data = KeyDataset(test_data)
    test_data = DataLoader(test_data, batch_size=args.batch_size, collate_fn=default_collate)
    return test_data


# def compute_rouge(source, target):
#     """计算rouge-1、rouge-2、rouge-l
#     """
#
#     source, target = ' '.join(source), ' '.join(target)
#     try:
#         scores = rouge.Rouge().get_scores(hyps=source, refs=target)
#         return {
#             'rouge-1': scores[0]['rouge-1']['f'],
#             'rouge-2': scores[0]['rouge-2']['f'],
#             'rouge-l': scores[0]['rouge-l']['f'],
#         }
#     except ValueError:
#         return {
#             'rouge-1': 0.0,
#             'rouge-2': 0.0,
#             'rouge-l': 0.0,
#         }


# def compute_rouges(sources, targets):
#     scores = {
#         'rouge-1': 0.0,
#         'rouge-2': 0.0,
#         'rouge-l': 0.0,
#     }
#     for source, target in zip(sources, targets):
#         score = compute_rouge(source, target)
#         for k, v in scores.items():
#             scores[k] = v + score[k]
#
#     return {k: v / len(targets) for k, v in scores.items()}


def generate(test_data, model, tokenizer, args):
    gens, summaries = [], []
    model.eval()
    for feature in tqdm(test_data):
        raw_data = feature['raw_data']
        content = {k: v for k, v in feature.items() if k not in ['raw_data', 'title']}
        gen = model.generate(max_length=args.max_len_generate,
                             eos_token_id=tokenizer.sep_token_id,
                             decoder_start_token_id=tokenizer.cls_token_id,
                             **content)
        gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
        gen = [item.replace(' ', '') for item in gen]

        for title in gen:
            print("=============",title)
            return title
    # if 'title' in feature:
    #     summaries.extend(feature['title'])
    # if summaries:
    #     scores = compute_rouges(gens, summaries)
    #     print(scores)


def init_argument():
    parser = argparse.ArgumentParser(description='t5-pegasus-chinese')
    parser.add_argument('--pretrain_model', default='/root/Projects/Abstract/flask-vue-spa/model')
    parser.add_argument('--model', default='/root/Projects/Abstract/flask-vue-spa/saved_model/summary_model')

    parser.add_argument('--batch_size', default=16, help='batch size')
    parser.add_argument('--max_len', default=1024, help='max length of inputs')
    parser.add_argument('--max_len_generate', default=40, help='max length of generated text')
    parser.add_argument('--use_multiprocess', default=False, action='store_true')

    args, unknown = parser.parse_known_args()

    return args


# if __name__ == '__main__':
def get_T5_result(data_line):
    args = init_argument()

    tokenizer = MTokenizer.from_pretrained(args.pretrain_model)
   

    model = torch.load(args.model, map_location=device)

    test_data = prepare_data(data_line, args, tokenizer)
    result = generate(test_data, model, tokenizer, args)
    return result
# get_T5_result('hhahahahha')