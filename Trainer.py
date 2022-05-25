import os
import re
import rouge
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import collections.abc as container_abcs
from transformers import MT5ForConditionalGeneration
from data_loader_func import data_loader
from Tokenizer import MTokenizer, create_data,KeyDataset


def sequence_padding(inputs, length=None, padding=0):
    """
    通过numpy进行填充，确保输入长度一致
    :param inputs:
    :param length:
    :param padding:
    :return: array
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
    """
    分组batch
    :param batch:
    :return:tensor
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
        return torch.stack(batch, 0, out=out)
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


def prepare_data(args, data_path, tokenizer, term='train'):
    """
    数据分组，按先编码后进行分组，返回tensor形式
    :param args:
    :param data_path:
    :param tokenizer:
    :param term:
    :return:
    """
    data = data_loader(data_path)
    data = create_data(data, tokenizer, args.max_len, term)
    data = KeyDataset(data)
    data = DataLoader(data, batch_size=args.batch_size, collate_fn=default_collate)
    return data


def compute_rouge(source, target):
    """
    计算rouge-1、rouge-2、rouge-l
    :param source: 源
    :param target: 目标
    :return:
    """
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.Rouge().get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


def compute_rouges(sources, targets):
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_rouge(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]

    return {k: v / len(targets) for k, v in scores.items()}


def train_model(model, adam, train_data, dev_data, tokenizer, device, args):
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    best = 0
    for epoch in range(args.num_epoch):
        model.train()
        print(len(train_data.dataset))
        for i, cur in enumerate(tqdm(train_data, desc='Epoch {}:'.format(epoch))):
            cur = {k: v.to(device) for k, v in cur.items()}
            prob = model(**cur)[0]
            mask = cur['decoder_attention_mask'][:, 1:].reshape(-1).bool()
            prob = prob[:, :-1]
            prob = prob.reshape((-1, prob.size(-1)))[mask]
            labels = cur['decoder_input_ids'][:, 1:].reshape(-1)[mask]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prob, labels)
            if i % 100 == 0:
                print("Iter {}:  Training Loss: {}".format(i, loss.item()))
            loss.backward()
            adam.step()
            adam.zero_grad()

        # 验证
        model.eval()
        gens = []
        summaries = []
        for feature in tqdm(dev_data):
            title = feature['title']
            content = {k: v.to(device) for k, v in feature.items() if k != 'title'}
            if args.data_parallel and torch.cuda.is_available():
                gen = model.module.generate(max_length=args.max_len_generate,
                                            eos_token_id=tokenizer.sep_token_id,
                                            decoder_start_token_id=tokenizer.cls_token_id,
                                            **content)
            else:
                gen = model.generate(max_length=args.max_len_generate,
                                     eos_token_id=tokenizer.sep_token_id,
                                     decoder_start_token_id=tokenizer.cls_token_id,
                                     **content)
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            gen = [item.replace(' ', '') for item in gen]
            # print(title)
            # print(gen)
            gens.extend(gen)
            summaries.extend(title)
        scores = compute_rouges(gens, summaries)
        print("Validation Loss: {}".format(scores))
        rouge_l = scores['rouge-l']
        if rouge_l > best:
            best = rouge_l
            if args.data_parallel and torch.cuda.is_available():
                torch.save(model.module, os.path.join(args.model_dir, 'summary_model'))
            else:
                torch.save(model, os.path.join(args.model_dir, 'summary_model'))
        # torch.save(model, os.path.join(args.model_dir, 'summary_model_epoch_{}'.format(str(epoch))))


def init_argument():
    parser = argparse.ArgumentParser(description='t5-pegasus-chinese')
    parser.add_argument('--train_data', default='./src/train.tsv')
    parser.add_argument('--dev_data', default='./src/val.tsv')
    parser.add_argument('--pretrain_model', default='./model')
    parser.add_argument('--model_dir', default='./saved_model')

    parser.add_argument('--num_epoch', default=20, help='number of epoch')
    parser.add_argument('--batch_size', default=16, help='batch size')
    parser.add_argument('--lr', default=2e-4, help='learning rate')
    parser.add_argument('--data_parallel', default=False)
    parser.add_argument('--max_len', default=1024, help='max length of inputs')
    parser.add_argument('--max_len_generate', default=40, help='max length of outputs')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # step 1. init argument
    args = init_argument()

    # step 2. prepare training data and validation data
    tokenizer = MTokenizer.from_pretrained(args.pretrain_model)
    train_data = prepare_data(args, args.train_data, tokenizer, term='train')
    dev_data = prepare_data(args, args.dev_data, tokenizer, term='dev')

    # step 3. load pretrain model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MT5ForConditionalGeneration.from_pretrained(args.pretrain_model).to(device)
    if args.data_parallel and torch.cuda.is_available():
        device_ids = range(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # step 4. finetune
    adam = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_model(model, adam, train_data, dev_data, tokenizer, device, args)