# coding=utf-8
"""
train.csv 有6列：
user：用户ID
event：事件ID
invited：是否被邀请（0/1）
timestamp：ISO-8601 UTC格式时间字符串，表示用户看到该事件的时间
interested, and not_interested

Test.csv 除了没有interested, and not_interested，其余列与train相同
 """
import pandas as pd

# 读取数据
train = pd.read_csv("train.csv")
print(train.head())
print(train.info())

"""
test同.csv 有6列：
user：用户ID
event：事件ID
invited：是否被邀请（0/1）
timestamp：ISO-8601 UTC格式时间字符串，表示用户看到该事件的时间
interested, and not_interested

Test.csv 除了没有interested, and not_interested，其余列与train相同
"""

# 读取数据
test = pd.read_csv("test.csv")
print(test.head())
print(test.info())

# ---------------------用户数据--------------------------
"""
用户描述信息在users.csv文件：共7维特征
user_id
locale：地区，语言
birthyear：出生年月
gender：性别
joinedAt：用户加入APP的时间，ISO-8601 UTC time
location：地点
timezone：时区
 """

# 读取数据
users = pd.read_csv("users.csv")
print(users.head())
print(users.info())

"""
gender、joinedAt、location、timezone这几个特征有缺失值 所以需要做缺失值处理
用户数比测试集和训练集中出现的用户多 为节省空间和时间，竞赛中可以只取出训练集和测试集中有的用户 
（猜测event也是一样，因为events.csv以gz压缩格式给出，记录数目应该更多）
"""

# ---------------------事件数据--------------------------
"""
事件描述信息在events.csv文件：共110维特征
前9列：event_id, user_id, start_time, city, state, zip, country, lat, and lng.
event_id：id of the event, 
user_id：id of the user who created the event.  
city, state, zip, and country： more details about the location of the venue (if known).
lat and lng： floats（latitude and longitude coordinates of the venue）
start_time： 字符串，ISO-8601 UTC time，表示事件开始时间

后101列为词频：count_1, count_2, ..., count_100，count_other
count_N：事件描述出现第N个词的次数
count_other：除了最常用的100个词之外的其余词出现的次数
 """

# 读取数据
events = pd.read_csv("events.csv")
print(events.head())
print(events.info())

# ----------------事件参加者数据---------------------
"""
event_attendees.csv文件：共5维特征
event_id：事件ID
yes, maybe, invited, and no：以空格隔开的用户列表，
分别表示该事件参加的用户、可能参加的用户，被邀请的用户和不参加的用户.
 """

# 读取数据
event_attendees = pd.read_csv("event_attendees.csv")
print(event_attendees.head())
print(event_attendees.info())

# -------------------用户好友数据---------------------
"""
user_friends.csv文件：共2维特征
user：用户ID
friends：以空格隔开的用户好友ID列表，
 """

# 读取数据
user_friends = pd.read_csv("user_friends.csv")
print(user_friends.head())
print(user_friends.info())
