# coding=utf-8
"""用户数据处理"""
"""只取训练集和测试集中出现的用户ID"""
"""用户描述信息在users.csv文件：共7维特征 
locale：地区，语言 
birthyear：出身年 
gender：性别 
joinedAt：用户加入APP的时间，ISO-8601 UTC time 
location：地点 
timezone：时区
"""

import pandas as pd
import scipy.sparse as ss
import scipy.io as sio

# 保存数据
import cPickle

# event的特征需要编码
from utils import FeatureEng
from sklearn.preprocessing import normalize
# 相似度/距离
import scipy.spatial.distance as ssd

"""总的用户数目超过训练集和测试集中的用户， 为节省处理时间和内存，先去处理train和test，得到竞赛需要用到的事件和用户 
然后对在训练集和测试集中出现过的事件和用户建立新的ID索引 先运行user_event.py, 得到事件列表文件：PE_userIndex.pkl"""

"""读取之前算好的测试集和训练集中出现过的用户"""

# 读取训练集和测试集中出现过的用户列表
userIndex = cPickle.load(open("PE_userIndex.pkl", 'rb'))
n_users = len(userIndex)

print("number of users in train & test :%d" % n_users)

# 处理users.csv --> 特征编码、用户之间的相似度
users = pd.read_csv("users.csv")

FE = FeatureEng()
# locale	birthyear	gender	joinedAt	location	timezone
# 去掉user_id列
n_cols = users.shape[1] - 1
cols = ['LocaleId', 'BirthYearInt', 'GenderId', 'JoinedYearMonth', 'CountryId', 'TimezoneInt']

# users编码后的特征
# userMatrix = np.zeros((n_users, n_cols), dtype=np.int)
userMatrix = ss.dok_matrix((n_users, n_cols))

for u in range(users.shape[0]):
    userId = str(users.loc[u, 'user_id'])

    if userIndex.has_key(userId):  # 在训练集或测试集中出现
        i = userIndex[userId]

        userMatrix[i, 0] = FE.getLocaleId(users.loc[u, 'locale'])
        userMatrix[i, 1] = FE.getBirthYearInt(users.loc[u, 'birthyear'])
        userMatrix[i, 2] = FE.getGenderId(users.loc[u, 'gender'])
        userMatrix[i, 3] = FE.getJoinedYearMonth(users.loc[u, 'joinedAt'])

        # 由于地点的写法不规范，该编码似乎不起作用（所有样本的特征都被编码成0了）
        userMatrix[i, 4] = FE.getCountryId(users.loc[u, 'location'])

        userMatrix[i, 5] = FE.getTimezoneInt(users.loc[u, 'timezone'])

# 归一化用户矩阵
userMatrix = normalize(userMatrix, norm="l2", axis=0, copy=False)
sio.mmwrite("US_userMatrix", userMatrix)

# 计算用户相似度矩阵，之后用户推荐系统
userSimMatrix = ss.dok_matrix((n_users, n_users))

# 读取在测试集和训练集中出现的用户对
uniqueUserPairs = cPickle.load(open("FE_uniqueUserPairs.pkl", 'rb'))

# 对角线元素
for i in range(0, n_users):
    userSimMatrix[i, i] = 1.0

# 对称
for u1, u2 in uniqueUserPairs:
    # i = userIndex[u1]
    # j = userIndex[u2]
    i = u1
    j = u2
    if not userSimMatrix.has_key((i, j)):
        # Person相关系数做为相似度度量
        # 特征：国家（locale、location）、年龄、性别、时区、地点
        # usim = ssd.correlation(userMatrix[i,:],
        # userMatrix[j,:])

        usim = ssd.correlation(userMatrix.getrow(i).todense(),
                               userMatrix.getrow(j).todense())
        userSimMatrix[i, j] = usim
        userSimMatrix[j, i] = usim

sio.mmwrite("US_userSimMatrix", userSimMatrix)
