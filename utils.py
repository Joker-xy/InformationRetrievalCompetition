# encoding=utf-8
# 预处理
import os
import pickle
import re
import jieba
import sqlite3


# 读取数据库 参数：需要获取的字段名列表
# 数据库文件名：Humanities_Social_Sciences_Datasets_sqlite3.db
# 数据表名：dataset_metadata
# 相关字段：id,title,subject学科分类,description描述,date发布时间,creator作者,publisher出版平台,source获取来源,language语言
def readDb(params):
    # 创建数据库连接
    db_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'Humanities_Social_Sciences_Datasets_sqlite3.db')
    print(db_path)
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # sql读取数据
    sql = "select "
    for i in range(len(params)):
        if i == 0:
            sql += params[i]
        else:
            sql += "," + params[i]
    sql += " from dataset_metadata"
    print('sql: ', sql)
    cursor.execute(sql)
    result = cursor.fetchall()
    print('len:', len(result))
    # 关闭数据库连接
    connection.close()
    # 返回数据集列表
    return result


# 将数据库中的None值改为''
def dbDeleteNone(para):
    # 创建数据库连接
    db_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'Humanities_Social_Sciences_Datasets_sqlite3.db')
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("UPDATE dataset_metadata SET " + para + " = '' WHERE " + para + " is null ")
    connection.commit()
    print(connection.total_changes)
    connection.close()


# 从stopwords.txt中读取停用词
def get_stopwords():
    stopwords_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'stopwords.txt')
    stopwordsList = []
    with open(stopwords_file_path, "r", encoding="utf-8") as f:
        data = f.readlines()
        for item in data:
            stopwordsList.append(re.sub(r"\s", "", item))
    return stopwordsList


# 语料预处理 参数：文档字符串
def pretreatment(doc_raw):
    # 分词
    doc = jieba.lcut(doc_raw, cut_all=False, HMM=True)
    # 获取停用词表
    stopwords = get_stopwords()
    # 去停用词
    doc1 = []
    for word in doc:
        if not word.strip() in stopwords and not word.isspace():
            doc1.append(word)
    return doc1


# 保存模型(类)到文件
def save_model(file_path, model_class):
    # pickle库用来序列化数据并存储到文件
    file = open(file_path, 'wb')
    model_str = pickle.dumps(model_class)
    file.write(model_str)
    file.close()


# 读取文件中保存的模型(类)
def read_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.loads(file.read())
    return model

