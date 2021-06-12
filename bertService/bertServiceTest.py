# 导入bert客户端
import os
import sqlite3

from bert_serving.client import BertClient
import numpy as np
from pandas.tests.io.excel.test_xlrd import xlwt
import utils


def cal_similar(sen_a_vec, sen_b_vec):
    vector_a = np.mat(sen_a_vec)
    vector_b = np.mat(sen_b_vec)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return cos


# 计算预料句向量(单字段)
def bert_cal_vec(param):
    # 读取数据
    docs_raw = utils.readDb(['id', param])

    # 构建bertClient
    bert_client = BertClient()

    # 生成的向量数组
    docs_vec = []

    # 计算句向量
    for i in range(len(docs_raw)):

        doc_raw = docs_raw[i]
        doc_vec = bert_client.encode([doc_raw[1] + '。'])[0]
        docs_vec.append(doc_vec)
        print(param, ': ', i)
        # if i > 2:
        #     break

    # 关闭bertClient
    bert_client.close()

    # 转换为np数组，并保存到文件
    docs_vec = np.array(docs_vec)
    np.save('vecs/multi/' + param + '_vec', docs_vec)


# 计算预料句向量(三个字段混合)
def bert_cal_vec_mixed():
    # 读取数据
    docs_raw = utils.readDb(['id', 'title', 'subject', 'description'])
    # 构建bertClient
    bert_client = BertClient()

    # mixed_vec 合并但不加权
    # 生成的向量数组
    docs_vec = []
    # 计算句向量
    for i in range(len(docs_raw)):
        doc_raw = docs_raw[i]
        doc = doc_raw[1] + '。' + doc_raw[2] + '。' + doc_raw[3] + '。'
        doc_vec = bert_client.encode([doc])[0]
        docs_vec.append(doc_vec)
        print('mixed: ', i)
        # if i > 2:
        #     break
    # 关闭bertClient
    bert_client.close()
    # 转换为np数组，并保存到文件
    docs_vec = np.array(docs_vec)
    np.save('vecs/multi/' + 'mixed_vec', docs_vec)


# 计算预料句向量(三个字段加权混合)
def bert_cal_vec_mixed_weighting():
    # 读取数据
    docs_raw = utils.readDb(['id', 'title', 'subject', 'description'])
    # 构建bertClient
    bert_client = BertClient()
    # mixed_weighting_vec 合并 + 加权
    # 生成的向量数组
    docs_vec = []
    # 计算句向量
    for i in range(len(docs_raw)):
        doc_raw = docs_raw[i]
        doc = doc_raw[1] + '。' + doc_raw[1] + '。' + doc_raw[1] + '。' + doc_raw[1] + '。' + doc_raw[1] + '。' + doc_raw[2] + '。' + doc_raw[2] + '。' + doc_raw[2] + '。' + doc_raw[3] + '。'
        doc_vec = bert_client.encode([doc])[0]
        docs_vec.append(doc_vec)
        print('mixed_weighting: ', i)
        # if i > 2:
        #     break
    # 关闭bertClient
    bert_client.close()
    # 转换为np数组，并保存到文件
    docs_vec = np.array(docs_vec)
    np.save('vecs/multi/' + 'mixed_weighting_vec', docs_vec)


if __name__=='__main__':

    # title 句向量
    bert_cal_vec('title')
    title_vec = np.load('vecs/title_vec.npy')
    print('title:', len(title_vec))
    # subject 句向量
    bert_cal_vec('subject')
    subject_vec = np.load('vecs/subject_vec.npy')
    print('subject:', len(subject_vec))
    # description 句向量
    bert_cal_vec('description')
    description_vec = np.load('vecs/description_vec.npy')
    print('description:', len(description_vec))

    # 计算混合后的句向量
    bert_cal_vec_mixed()
    mixed_vec = np.load('vecs/mixed_vec.npy')
    print('mixed_vec', len(mixed_vec))
    # 计算加权混合后的句向量
    bert_cal_vec_mixed_weighting()
    mixed_weighting_vec = np.load('vecs/mixed_weighting_vec.npy')
    print('mixed_weighting_vec', len(mixed_weighting_vec))
