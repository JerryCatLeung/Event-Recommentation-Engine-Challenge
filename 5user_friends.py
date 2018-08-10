# coding=utf-8
"""用户社交数据（user_friends.csv）处理"""
"""user_friends.csv文件：共2维特征 user：用户ID friends：以空格隔开的用户好友ID列表"""
import pandas as pd

import numpy as np
import scipy.sparse as ss
import scipy.io as sio

# 保存数据
import cPickle

from sklearn.preprocessing import normalize

"""总的用户数目超过训练集和测试集中的用户， 为节省处理时间和内存，先去处理train和test，得到竞赛需要用到的活动和用户 
然后对在训练集和测试集中出现过的事件和用户建立新的ID索引 先运行user_event.py, 得到事件列表文件：PE_userIndex.pkl"""

"""读取之前算好的测试集和训练集中出现过的用户"""
# 读取训练集和测试集中出现过的事件列表
userIndex = cPickle.load(open("PE_userIndex.pkl", 'rb'))
n_users = len(userIndex)

print("number of users in train & test :%d" % n_users)

"""读取之前用户-活动分数矩阵，将朋友参加活动的影响扩展到用户"""
# 用户-事件关系矩阵
userEventScores = sio.mmread("PE_userEventScores")

# 后续用于将用户朋友参加的活动影响到用户
eventsForUser = cPickle.load(open("PE_eventsForUser.pkl", 'rb'))

# 读取数据

"""
  找出某用户的那些朋友，想法非常简单
  1)如果你有更多的朋友，可能你性格外向，更容易参加各种活动
  2)如果你朋友会参加某个活动，可能你也会跟随去参加一下
"""

# 用户有多少个朋友
numFriends = np.zeros((n_users))
userFriends = ss.dok_matrix((n_users, n_users))

fin = open("user_friends.csv", 'rb')
# 字段：user，friends
fin.readline()  # skip header

# ln = 0
for line in fin:  # 对每个用户
    cols = line.strip().split(",")
    user = str(cols[0])  # user

    if userIndex.has_key(user):  # 该用户在训练集和测试集的用户列表中
        friends = cols[1].split(" ")  # friends
        i = userIndex[user]  # 该用户的索引
        numFriends[i] = len(friends)
        for friend in friends:  # 该用户的每个朋友
            str_friend = str(friend)
            if userIndex.has_key(str_friend):  # 如果朋友也在训练集或测试集中出现
                j = userIndex[str_friend]  # 朋友的索引

                # the objective of this score is to infer the degree to
                # and direction in which this friend will influence the
                # user's decision, so we sum the user/event score for
                # this user across all training events.

                # userEventScores为用户对活动的打分（interested - not interseted）
                # 在Users-Events.ipynb中计算好了
                eventsForUser = userEventScores.getrow(j).todense()

                # 所有朋友参加活动的数量（平均频率）
                score = eventsForUser.sum() / np.shape(eventsForUser)[1]
                userFriends[i, j] += score
                userFriends[j, i] += score

fin.close()

# 用户的朋友数目
# 归一化数组
sumNumFriends = numFriends.sum(axis=0)
numFriends = numFriends / sumNumFriends
sio.mmwrite("UF_numFriends", np.matrix(numFriends))

#
userFriends = normalize(userFriends, norm="l2", axis=0, copy=False)
sio.mmwrite("UF_userFriends", userFriends)

print(numFriends)
