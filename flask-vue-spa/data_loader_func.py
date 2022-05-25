import json
import numpy as np
import re
# from Tokenizer import create_data, KeyDataset
# from torch.utils.data import DataLoader
# from Trainer import default_collate


def data_out(filename: str):
    """
    数据预处理，生成csv格式
    :param filename:
    :return:
    """
    with open(r'./src/' + filename + '.json', 'r', encoding='utf-8') as file:
        load_dict = json.load(file)

        load_dict = np.array(load_dict)
        # print(load_dict)
    with open(r'./src/' + filename + '.tsv', 'w', encoding='utf-8') as fileout:
        for sets in load_dict:
            title, content = sets['title'], re.sub('\s|\t|\n', '', sets['content'])
            print(title + '\t' + content, file=fileout, end='\n')

    print(filename + '处理完成')


def data_loader(filename: str):
    """
    数据处理，以元组形式返回
    :param filename: 文件名
    :return: tuple (title, content)
    """
    ret = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            cur = line.strip().split('\t')
            title, content = cur[0], cur[1]
            ret.append((title, content))

    return ret



if __name__ == '__main__':
    data_out('train')
    data_out('val')
    # data_out('val')

    # print(data_loader('train'))
