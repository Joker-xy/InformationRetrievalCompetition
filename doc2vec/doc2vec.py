import xlwt
import utils
from gensim.models import doc2vec

# 读取数据
docs_raw = utils.readDb(['id', 'title', 'subject', 'description'])


# 构建doc2vec，并保存到文件
def bulidDoc2vec(save_path='doc2vec.vec'):

    # 预处理
    docs = []
    for doc_raw in docs_raw:
        # TODO:这里将title、subject、description简单拼接,但是因为doc2vec需要考虑词顺序，感觉这样处理不是很好
        doc = utils.pretreatment(doc_raw[1] + doc_raw[1] + doc_raw[1] + doc_raw[1] + doc_raw[1]
                                 + doc_raw[2] + doc_raw[2] + doc_raw[2] + doc_raw[3])
        docs.append(doc)
        # if len(docs) > 5000:
        #     break
    # 标签处理
    docs_tagged = [doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
    # document: 经过标签化的语料
    # window:当前词与预测词之间的最大距离
    # vector_size:特征向量维度
    # min_count:忽略总频率低于此的所有单词
    # dm:训练算法  0=PV-DBOW  1=PV-DM(用dm算法偏差有点大...)
    # epochs:迭代次数
    # workers:多线程运算
    model = doc2vec.Doc2Vec(documents=docs_tagged, dm=1, vector_size=1000, window=2, min_count=1, epochs=8, workers=4)

    # 文件保存
    model.save(save_path)
    # 保存词向量和文档向量结果
    # model.wv.save(save_path + 'wv.kv')
    # model.dv.save(save_path + 'dv.kv')
    return model


# 计算查询式与语料库的相似度，并输出到文件
# model:构建好的doc2vec模型；
# q_raw:查询式；
# n:输出相似度最高的前n项
def cal_sim_score(model, q_raw, file_name, n=10000):
    # 查询式预处理
    q = utils.pretreatment(q_raw)
    # 查询式转换为doc2vec向量
    q_vec = model.infer_vector(q)

    # 计算余弦相似度（取值[-1,1]）；返回数组，里面为元组(文档序号,相似度)
    # vector:查询式向量
    # topn:获取相似度最大的n个结果；传入none则返回所有相似度(不排序)
    sim_result = model.dv.similar_by_vector(vector=q_vec, topn=n)

    # 保存到doc2vec_result.xls
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
    for i in range(len(sim_result)):
        index = sim_result[i][0]
        doc = docs_raw[index]
        # 第一列 id
        sheet.write(i + 1, 0, index+1)
        # 第二列 相似度得分
        sheet.write(i + 1, 1, sim_result[i][1])
        # 第三列 id
        sheet.write(i + 1, 2, doc[0])
        # 第四列 title
        sheet.write(i + 1, 3, doc[1])
        # 第五列 subject
        sheet.write(i + 1, 4, doc[2])
        # 第六列 description
        sheet.write(i + 1, 5, doc[3])
    workbook.save(file_name)

    return sim_result


# 自相似性测试（把训练语料当作query，计算最相似的是否为自身）
# 返回元组(得到自身的数量，百分比)
def self_sim_test(model):
    first_num = 0
    for i in range(len(docs_raw)):
        doc_raw = docs_raw[i]
        doc = utils.pretreatment(doc_raw[1] + doc_raw[1] + doc_raw[1] + doc_raw[1] + doc_raw[1]
                                 + doc_raw[2] + doc_raw[2] + doc_raw[2] + doc_raw[3])
        result = model.dv.similar_by_vector(vector=model.infer_vector(doc), topn=1)[0]
        if i == result[0]:
            first_num += 1
        # if i > 5000:
        #     break
    return first_num, first_num / len(docs_raw)


if __name__ == '__main__':

    # 构建doc2vec模型，并保存到文件
    model = bulidDoc2vec(save_path='doc2vec.vec')
    # 从文件中导入doc2vec模型
    model = doc2vec.Doc2Vec.load('doc2vec.vec')

    # 自相似性测试
    print(self_sim_test(model))

    # 词向量集合
    # wv = model.wv
    # 文档向量集合（KeyedVectors类型：https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors）
    # dv = model.dv
    # # 获取某个词的向量
    # print(model.wv['危化'])
    # # 获取某个文档的向量
    # print(model.dv[0])

    # 查询式
    q_raw = '农村居民人均可支配收入'

    # 计算余弦相似度并输出前1000条到文件
    sim_result = cal_sim_score(model, q_raw, 'doc2vec_result.xls', 1000)
