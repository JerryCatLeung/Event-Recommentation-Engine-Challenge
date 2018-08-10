# coding=utf-8
"""用户和活动关联关系处理
整个数据集中活动数目（events.csv）太多，所以下面的处理我们找出只在训练集和测试集中出现的活动和用户集合，并对他们重新编制索引
"""
# 保存数据
import cPickle

import itertools

# 处理事件字符串
import datetime

import numpy as np
import scipy.io as sio
import scipy.sparse as ss

# 相似度/距离
import scipy.spatial.distance as ssd

from collections import defaultdict
from sklearn.preprocessing import normalize

"""
我们只关心train和test中出现的user和event，因此重点处理这部分关联数据

train.csv 有6列：
user：用户ID
event：活动ID
invited：是否被邀请（0/1）
timestamp：ISO-8601 UTC格式时间字符串，表示用户看到该活动的时间
interested, and not_interested

Test.csv 除了没有interested, and not_interested，其余列与train相同
 """

# 统计训练集中有多少独立的用户的events
uniqueUsers = set()
uniqueEvents = set()

# 到排表
# 统计每个用户参加的活动   / 每个活动参加的用户
eventsForUser = defaultdict(set)
usersForEvent = defaultdict(set)

for filename in ["train.csv", "test.csv"]:
    f = open(filename, 'rb')

    # 忽略第一行（列名字）
    f.readline().strip().split(",")

    for line in f:  # 对每条记录
        cols = line.strip().split(",")
        uniqueUsers.add(cols[0])  # 第一列为用户ID
        uniqueEvents.add(cols[1])  # 第二列为活动ID

        # eventsForUser[cols[0]].add(cols[1])    #该用户参加了这个活动
        # usersForEvent[cols[1]].add(cols[0])    #该活动被用户参加
    f.close()

n_uniqueUsers = len(uniqueUsers)
n_uniqueEvents = len(uniqueEvents)

print("number of uniqueUsers :%d" % n_uniqueUsers)
print("number of uniqueEvents :%d" % n_uniqueEvents)

# 用户关系矩阵表，可用于后续LFM/SVD++处理的输入
# 这是一个稀疏矩阵，记录用户对事件感兴趣
userEventScores = ss.dok_matrix((n_uniqueUsers, n_uniqueEvents))
userIndex = dict()
eventIndex = dict()

# 重新编码用户索引字典
for i, u in enumerate(uniqueUsers):
    userIndex[u] = i

# 重新编码活动索引字典
for i, e in enumerate(uniqueEvents):
    eventIndex[e] = i

n_records = 0
n_zeros = 0
n_ones = 0
ftrain = open("train.csv", 'rb')
ftrain.readline()
for line in ftrain:
    cols = line.strip().split(",")
    i = userIndex[cols[0]]  # 用户
    j = eventIndex[cols[1]]  # 活动

    eventsForUser[i].add(j)  # 该用户参加了这个活动
    usersForEvent[j].add(i)  # 该活动被用户参加

    # userEventScores[i, j] = int(cols[4]) - int(cols[5])   #interested - not_interested
    score = int(cols[4])
    # if score == 0:  #0在稀疏矩阵中表示该元素不存在，因此借用-1表示interested=0
    # userEventScores[i, j] = -1
    #    n_zeros += 1
    # else:
    userEventScores[i, j] = score
    #    n_ones += 1
ftrain.close()

# temp= userEventScores.tocsr()()
# print np.transpose(np.nonzero(temp))
# nonzero_scores_index = np.transpose(np.nonzero(temp))
# n_nonzero_scores =nonzero_scores_index.shape[0]
# print "non zero scores is:", n_nonzero_scores

# 统计每个用户参加的事件，后续用于将用户朋友参加的活动影响到用户
cPickle.dump(eventsForUser, open("PE_eventsForUser.pkl", 'wb'))
# 统计事件参加的用户
cPickle.dump(usersForEvent, open("PE_usersForEvent.pkl", 'wb'))

# 保存用户-活动关系矩阵R，以备后用
sio.mmwrite("PE_userEventScores", userEventScores)

# 保存用户索引表
cPickle.dump(userIndex, open("PE_userIndex.pkl", 'wb'))
# 保存活动索引表
cPickle.dump(eventIndex, open("PE_eventIndex.pkl", 'wb'))

# 为了防止不必要的计算，我们找出来所有关联的用户 或者 关联的event
# 所谓的关联用户，指的是至少在同一个event上有行为的用户pair
# 关联的event指的是至少同一个user有行为的event pair
uniqueUserPairs = set()
uniqueEventPairs = set()
for event in uniqueEvents:
    i = eventIndex[event]
    users = usersForEvent[i]
    if len(users) > 2:
        uniqueUserPairs.update(itertools.combinations(users, 2))

for user in uniqueUsers:
    u = userIndex[user]
    events = eventsForUser[u]
    if len(events) > 2:
        uniqueEventPairs.update(itertools.combinations(events, 2))

# 保存用户-事件关系对索引表
cPickle.dump(uniqueUserPairs, open("FE_uniqueUserPairs.pkl", 'wb'))
cPickle.dump(uniqueEventPairs, open("PE_uniqueEventPairs.pkl", 'wb'))
