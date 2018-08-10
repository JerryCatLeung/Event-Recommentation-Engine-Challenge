# coding=utf-8
"""只取训练集和测试集中出现的用户ID"""
"""
event_attendees.csv文件：共5维特征 
event_id：活动ID 
yes, maybe, invited, and no：以空格隔开的用户列表， 分别表示该活动参加的用户、可能参加的用户，被邀请的用户和不参加的用户.
"""

import pandas as pd

import numpy as np
import scipy.sparse as ss
import scipy.io as sio

# 保存数据
import cPickle

from sklearn.preprocessing import normalize

"""总的用户数目超过训练集和测试集中的用户， 为节省处理时间和内存，先去处理train和test，得到竞赛需要用到的事件和用户 
然后对在训练集和测试集中出现过的事件和用户建立新的ID索引 先运行user_event.py, 得到事件列表文件：PE_userIndex.pkl"""

"""读取之前算好的测试集和训练集中出现过的活动"""

# 读取训练集和测试集中出现过的事件列表
eventIndex = cPickle.load(open("PE_eventIndex.pkl", 'rb'))
n_events = len(eventIndex)

print("number of events in train & test :%d" % n_events)

# 读取数据
"""
  统计某个活动，参加和不参加的人数，计算活动热度
"""

# 活动活跃度
eventPopularity = ss.dok_matrix((n_events, 1))

f = open("event_attendees.csv", 'rb')

# 字段：event_id,yes, maybe, invited, and no
f.readline()  # skip header

for line in f:
    cols = line.strip().split(",")
    eventId = str(cols[0])  # event_id
    if eventIndex.has_key(eventId):
        i = eventIndex[eventId]  # 事件索引

        # yes - no
        eventPopularity[i, 0] = \
            len(cols[1].split(" ")) - len(cols[4].split(" "))

f.close()

eventPopularity = normalize(eventPopularity, norm="l1",
                            axis=0, copy=False)
sio.mmwrite("EA_eventPopularity", eventPopularity)

print(eventPopularity.todense())
