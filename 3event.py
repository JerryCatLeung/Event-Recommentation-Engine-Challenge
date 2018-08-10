# coding=utf-8
"""对活动数据进行分析
活动描述信息在events.csv文件：共110维特征 前9列：event_id, user_id, start_time, city, state, zip, country, lat, and lng.
event_id：id of the event,
user_id：id of the user who created the event.
city, state, zip, and country： more details about the location of the venue (if known).
lat and lng： floats（latitude and longitude coordinates of the venue）
start_time： 字符串，ISO-8601 UTC time，表示活动开始时间
后101列为词频：count_1, count_2, ..., count_100，count_other count_N：活动描述出现第N个词的次数
count_other：除了最常用的100个词之外的其余词出现的次数
这里我们用count_1, count_2, ..., count_100，count_other属性做聚类，即事件用这些关键词来描述，可表示活动的类别"""

import scipy.sparse as ss
import scipy.io as sio

# 保存数据
import cPickle

# event的特征需要编码
from utils import FeatureEng
from sklearn.preprocessing import normalize
# 相似度/距离
import scipy.spatial.distance as ssd

"""统计活动数量"""
# 读取数据，并统计有多少不同的events
# 其实EDA.ipynb中用read_csv已经统计过了
lines = 0
fin = open("events.csv", 'rb')
# 字段：event_id, user_id,start_time, city, state, zip, country, lat, and lng， 101 columns of words count
fin.readline()  # skip header，列名行
for line in fin:
    cols = line.strip().split(",")
    lines += 1
fin.close()

print("number of records :%d" % lines)
"""活动数目太多（300w+），训练+测试集的活动没这么多，所有先去处理train和test，得到竞赛需要用到的活动和用户 
然后对在训练集和测试集中出现过的活动和用户建立新的ID索引 先运行user_event.py, 得到活动列表文件：PE_eventIndex.pkl"""

"""读取之前算好的测试集和训练集中出现过的活动"""
# 读取训练集和测试集中出现过的活动列表
eventIndex = cPickle.load(open("PE_eventIndex.pkl", 'rb'))
n_events = len(eventIndex)

print("number of events in train & test :%d" % n_events)

"""处理events.csv --> 特征编码、活动之间的相似度"""
FE = FeatureEng()

fin = open("events.csv", 'rb')

# 字段：event_id, user_id,start_time, city, state, zip, country, lat, and lng， 101 columns of words count
fin.readline()  # skip header

# start_time, city, state, zip, country, lat, and lng
eventPropMatrix = ss.dok_matrix((n_events, 7))

# 词频特征
eventContMatrix = ss.dok_matrix((n_events, 101))

for line in fin.readlines():
    cols = line.strip().split(",")
    eventId = str(cols[0])

    if eventIndex.has_key(eventId):  # 在训练集或测试集中出现
        i = eventIndex[eventId]

        # event的特征编码，这里只是简单处理，其实开始时间，地点等信息很重要
        eventPropMatrix[i, 0] = FE.getJoinedYearMonth(cols[2])  # start_time
        eventPropMatrix[i, 1] = FE.getFeatureHash(cols[3])  # city
        eventPropMatrix[i, 2] = FE.getFeatureHash(cols[4])  # state
        eventPropMatrix[i, 3] = FE.getFeatureHash(cols[5])  # zip
        eventPropMatrix[i, 4] = FE.getFeatureHash(cols[6])  # country
        eventPropMatrix[i, 5] = FE.getFloatValue(cols[7])  # lat
        eventPropMatrix[i, 6] = FE.getFloatValue(cols[8])  # lon

        # 词频
        for j in range(9, 110):
            eventContMatrix[i, j - 9] = cols[j]
fin.close()

# 用L2模归一化
eventPropMatrix = normalize(eventPropMatrix,
                            norm="l2", axis=0, copy=False)
sio.mmwrite("EV_eventPropMatrix", eventPropMatrix)

# 词频，可以考虑我们用这部分特征进行聚类，得到活动的genre
eventContMatrix = normalize(eventContMatrix,
                            norm="l2", axis=0, copy=False)
sio.mmwrite("EV_eventContMatrix", eventContMatrix)

# calculate similarity between event pairs based on the two matrices
eventPropSim = ss.dok_matrix((n_events, n_events))
eventContSim = ss.dok_matrix((n_events, n_events))

# 读取在测试集和训练集中出现的活动对
uniqueEventPairs = cPickle.load(open("PE_uniqueEventPairs.pkl", 'rb'))

for e1, e2 in uniqueEventPairs:
    # i = eventIndex[e1]
    # j = eventIndex[e2]
    i = e1
    j = e2

    # 非词频特征，采用Person相关系数作为相似度
    # 其实开始时间/国家/城市/经纬度等应该分别计算相似度
    if not eventPropSim.has_key((i, j)):
        epsim = ssd.correlation(eventPropMatrix.getrow(i).todense(),
                                eventPropMatrix.getrow(j).todense())

        eventPropSim[i, j] = epsim
        eventPropSim[j, i] = epsim

    # 对词频特征，采用余弦相似度，也可以用直方图交/Jacard相似度
    if not eventContSim.has_key((i, j)):
        ecsim = ssd.cosine(eventContMatrix.getrow(i).todense(),
                           eventContMatrix.getrow(j).todense())

        eventContSim[i, j] = epsim
        eventContSim[j, i] = epsim

sio.mmwrite("EV_eventPropSim", eventPropSim)
sio.mmwrite("EV_eventContSim", eventContSim)

print(eventPropSim.getrow(0).todense())
