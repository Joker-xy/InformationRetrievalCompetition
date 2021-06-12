import math
import xlwt

import utils
import numpy as np


class BM25(object):

    def __init__(self, docs, docs_raw):
        self.docs_raw = docs_raw     # 未处理的文档列表，包含id、title、subject、description
        self.docs = docs             # 传入已经分好词的文档列表
        self.docs_num = len(docs)    # 文档总数量
        self.avgdl = sum([len(doc) + 0.0 for doc in docs]) / self.docs_num  # 文档词数平均
        self.docs_f = []             # 存储文档中每个词出现次数（每个文档为一个dict）
        self.df = {}                 # 存储所有词及出现了该词的文档数量
        self.idf = {}                # 存储所有词的idf值  log( (docs_num - df[word] + 0.5) / (df[word] + 0.5) )
        self.k1 = 1.5                # 调整参数
        self.b = 0.75                # 调整参数
        self.init()

    # 初始化：计算docs_f、df、idf
    def init(self):
        for doc in self.docs:
            # 计算该文档中每个词的出现次数
            doc_f = {}
            for word in doc:
                doc_f[word] = doc_f.get(word, 0) + 1
            self.docs_f.append(doc_f)
            # 计算df
            for word in doc_f.keys():
                self.df[word] = self.df.get(word, 0) + 1
        # 计算idf
        for word, word_df in self.df.items():
            self.idf[word] = math.log(self.docs_num - word_df + 0.5) - math.log(word_df + 0.5)

    # 计算查询式q与索引为index的文档的相似度得分（q为分词后的词列表）
    def bm25_score(self, q, index):
        # 该文档所包含的所有词及其词频
        doc_f = self.docs_f[index]
        # 该文档总词数
        dl = len(self.docs[index])
        # 相关度累计得分
        score = 0
        for word in q:
            if word not in doc_f:
                # 文档中不包含该词
                continue
            else:
                # 文当中包含该词，计算相关度得分并累计 (idf(word) * f(word) * (k1+1)) / (f(word) + k1 * (1-b+b*(dl/avgdl)))
                score += (self.idf[word] * doc_f[word] * (self.k1 + 1)
                          / (doc_f[word] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)))
        return score

    # 计算所有相似度得分，并输出结果前n项到excel文件
    def bm25_score_all(self, q_raw, file_name, n=100000):
        # 查询式预处理
        q = utils.pretreatment(q_raw)
        print('q_raw:', q_raw)
        print('q:', q)

        # 计算查询式与所有文档的相似度
        scores = []
        for index in range(self.docs_num):
            score = self.bm25_score(q, index)
            scores.append(score)
        scores = np.array(scores)

        # argsort数组排序，传入-scores是倒序排列。返回排序后的原数组索引 [最大的数的索引, 第二大索引, ...]
        scores_sort_index = np.argsort(-scores)
        # 保存到bm_result.xls
        workbook = xlwt.Workbook(encoding="utf-8")
        sheet = workbook.add_sheet("sim_result")
        # 第一列 id
        sheet.write(0, 0, 'id')
        # 第二列 相似度得分
        sheet.write(0, 1, 'sim_score')
        # 第三列 id
        sheet.write(0, 2, 'id')
        # 第四列 title
        sheet.write(0, 3, 'title')
        # 第五列 subject
        sheet.write(0, 4, 'subject')
        # 第六列 description
        sheet.write(0, 5, 'description')
        for i in range(len(scores)):
            if i < n and scores[scores_sort_index[i]] > 0:
                doc = bm25.docs_raw[scores_sort_index[i]]
                # 第一列 id
                sheet.write(i + 1, 0, str(scores_sort_index[i]+1))
                # 第二列 相似度得分
                sheet.write(i + 1, 1, scores[scores_sort_index[i]])
                # 第三列 id
                sheet.write(i + 1, 2, doc[0])
                # 第四列 title
                sheet.write(i + 1, 3, doc[1])
                # 第五列 subject
                sheet.write(i + 1, 4, doc[2])
                # 第六列 description
                sheet.write(i + 1, 5, doc[3])
            else:
                break
        workbook.save(file_name)


# 初始话bm25模型(类)，并存储到bm25.pkl文件
def bm25_init():
    # 读取数据库
    docs_raw = utils.readDb(["id", 'title', 'subject', 'description'])
    # 预处理
    docs = []
    for doc in docs_raw:
        doc_str = ''
        # 按权重合并title、subject、description
        # title
        for i in range(5):
            doc_str += doc[1]
        # subject
        for i in range(3):
            doc_str += doc[2]
        # description
        for i in range(1):
            doc_str += doc[3]
        # 预处理（分词、去停用词）
        doc_words = utils.pretreatment(doc_str)
        docs.append(doc_words)
        # if len(docs) > 1000:
        #     break
    bm25 = BM25(docs, docs_raw)
    # 保存bm25模型(类)到bm25.pkl文件
    utils.save_model('bm/bm25.pkl', bm25)


if __name__ == '__main__':

    # 初始话bm25模型(类)，并存储到bm25.pkl文件
    # bm25_init()

    # 从bm25.pkl模型中读取bm25模型类
    bm25 = utils.read_model('bm25.pkl')

    # 查询式
    q_raw = '农村居民人均可支配收入'

    # bm25计算相似度并输出
    bm25.bm25_score_all(q_raw, 'bm_result.xls', 1000)
