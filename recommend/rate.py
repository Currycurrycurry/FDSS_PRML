import pandas as pd
import numpy as np
import tensorflow as tf
import csv

############################################几个常量###############################################################################
train_data_path = 'dataset/train_data.txt'
key_data_path = 'dataset/key_data.txt'
test_data_path = 'dataset/test_data.txt'
num_features = 5 #用户的特征数
iterate_time = 5000
train_column_names = ['userId','movieId','rating']
test_column_names = ['userId','movieId']
##############
REG_CONST = 0.001
LEARNING_RATE = 0.00005
MOMENTUM = 0.9
############################################几个有效函数: 标准化；计算RMSE###############################################################################
def normalizeRatings(rating,record):
    m,n = rating.shape
    rating_mean = np.zeros((m,1)) # m*1 电影平均分
    rating_norm = np.zeros((m,n)) # m*n 标准化后的电影得分
    for i in range(m):
        idx = record[i,:] != 0 # 难道是非0行的行数？
        #if idx==false :
        rating_mean[i] = np.mean(rating[i,idx])
        rating_norm[i,idx] -= rating_mean[i]
    rating_norm = np.nan_to_num(rating_norm)
    print(rating_norm)
    rating_mean = np.nan_to_num(rating_mean)
    print(rating_mean)
    return rating_norm,rating_mean

def calc_rmse(a,b,c):
    rmse = np.sqrt(np.sum(np.square(np.multiply(a-b,c))) / b.size)
    return rmse

def loss(users, user_embedding, movies, movie_embedding, ratings, reg_const):
    latent_users = tf.nn.embedding_lookup(user_embedding, users)
    latent_movies = tf.nn.embedding_lookup(movie_embedding, movies)
    pred_ratings = tf.reduce_sum(tf.multiply(latent_users, latent_movies), 1)
    reg = tf.reduce_sum(tf.square(latent_users), 1) + tf.reduce_sum(tf.square(latent_movies), 1)
    loss = tf.reduce_sum((tf.square(ratings - pred_ratings) + tf.multiply(reg_const, reg)), 0)

    return loss

def calc_rmse_mf(users, user_embedding, movies, movie_embedding, ratings):
    latent_users = tf.nn.embedding_lookup(user_embedding, users)
    latent_movies = tf.nn.embedding_lookup(movie_embedding, movies)
    pred_ratings = tf.reduce_sum(tf.multiply(latent_users, latent_movies), 1)
    rmse = tf.sqrt(tf.reduce_mean(tf.square(ratings - pred_ratings)))

    return rmse

def get_batch(data_arr, batch_size, batch_ix):
    return data_arr[batch_ix * batch_size: (batch_ix + 1) * batch_size]

############################################数据预处理###############################################################################

train_ratings_df = pd.read_csv(train_data_path,sep=',',names=train_column_names)
print(train_ratings_df.tail())

trained_ratings_df = train_ratings_df.sample(frac=0.8, random_state=0)
validated_ratings_df = train_ratings_df.drop(trained_ratings_df.index)

print("ratings_df for training: ",trained_ratings_df.shape[0])
print("ratings_df for validation: ",validated_ratings_df.shape[0])

userNo = train_ratings_df['userId'].max() + 1
movieNo = train_ratings_df['movieId'].max() + 1

rating = np.zeros((movieNo, userNo))

for index, row in trained_ratings_df.iterrows():
    rating[int(row['movieId']), int(row['userId'])] = row['rating']

record = rating > 0
record = np.array(record, dtype = int)

rating_norm, rating_mean = normalizeRatings(rating, record)
############################################训练模型###############################################################################
X_para = tf.Variable(tf.random_normal([movieNo,num_features],stddev = 0.35))
Theta_para = tf.Variable(tf.random_normal([userNo,num_features],stddev = 0.35))

loss = 1/2 * tf.reduce_sum(((tf.matmul(X_para,Theta_para,transpose_b=True) - rating_norm) * record) ** 2)+ 1/2 * (tf.reduce_sum(X_para ** 2) + tf.reduce_sum(Theta_para ** 2))
optimizer = tf.train.AdamOptimizer(1e-4)
train = optimizer.minimize(loss)

# 训练模型
print("training the model...")
tf.summary.scalar('loss',loss)

print("merging and saving...")
summaryMerged = tf.summary.merge_all() # merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。
filename = './movie_tensorborad'
writer = tf.summary.FileWriter(filename) # 指定一个文件用来保存图。
sess = tf.Session() # 定义一个session
init = tf.global_variables_initializer()
sess.run(init) # 运行session

# 递归iterate_time次直到收敛
print("recurring %s times.." %iterate_time)
for i in range(iterate_time):
    _, movie_summary = sess.run([train, summaryMerged])
    writer.add_summary(movie_summary, i)
    if i % 100 == 0:
        print("processed %d times recurring..." %i)

Current_X_parameters, Current_Theta_parameters = sess.run([X_para, Theta_para])
# Current_X_parameters为用户内容矩阵，Current_Theta_parameters用户喜好矩阵
predicts = np.dot(Current_X_parameters,Current_Theta_parameters.T) + rating_mean

train_error = calc_rmse(predicts,rating,record)
print("RMSE train error:",train_error)

# users = tf.placeholder(tf.int32)
# movies = tf.placeholder(tf.int32)
# ratings = tf.placeholder(tf.float32)
# reg_const = tf.constant(REG_CONST)
#
# user_embedding = tf.Variable(tf.multiply(0.1,tf.random_normal([userNo,num_features],0,1)),name="U")
# movie_embedding = tf.Variable(tf.multiply(0.1,tf.random_normal([movieNo,num_features],0,1)),name="V")
#
# # # 训练模型
# print("training the model...")
#
# loss = loss(users,user_embedding,movies,movie_embedding,ratings,reg_const)
# tf.summary.scalar('loss',loss)
# train_op = tf.train.MomentumOptimizer(LEARNING_RATE,MOMENTUM).minimize(loss)
#
#
# print("merging and saving...")
# summaryMerged = tf.summary.merge_all() # merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。
# filename = './movie_tensorborad'
# writer = tf.summary.FileWriter(filename) # 指定一个文件用来保存图。
#
# init_op = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init_op)
#
# # 递归iterate_time次直到收敛
# print("recurring %s times.." %iterate_time)
# for i in range(iterate_time):
#     _, movie_summary = sess.run([train_op, summaryMerged])
#     writer.add_summary(movie_summary, i)
#     if i % 100 == 0:
#         print("processed %d times recurring..." %i)
#
# Current_Users, Current_Movies = sess.run([user_embedding, movie_embedding])
# # Current_X_parameters为用户内容矩阵，Current_Theta_parameters用户喜好矩阵
# predicts = np.dot(Current_Users,Current_Movies.T) + rating_mean
#
# train_error = calc_rmse_mf(predicts,rating)
# print("RMSE train error:",train_error)


############################################验证集上测试###############################################################################

rating = np.zeros((movieNo, userNo))

for index, row in validated_ratings_df.iterrows():
    rating[int(row['movieId']), int(row['userId'])] = row['rating']

record = rating > 0
record = np.array(record, dtype = int)

validation_error = calc_rmse(predicts,rating,record)
print('RMSE validation error', validation_error)

############################################测试集上进行答题###############################################################################
test_data = pd.read_csv(test_data_path,sep=',',names=test_column_names)
print(test_data.tail())

pred = np.zeros(test_data.shape[0])

for row in test_data.index:
    pred[row] = predicts[int(test_data.loc[row]['movieId']), int(test_data.loc[row]['userId'])]


test_data['ratings'] = pred

test_data.to_csv(key_data_path,sep=',',header=None,index=None)

# testUseNo = test_data['userId'].max() + 1
#
# for userId in range(testUseNo):
#     sortedResult = predicts[:, int(userId)].argsort()[::-1]
#     idx = 0
#     print('为 %s 号用户推荐的评分最高的20部电影是：'.center(80, '=') %userId)
#     for i in sortedResult:
#         print('score: %.3f, movie name: %s' % (predicts[i, int(userId)], train_ratings_df.iloc[i]['movieId']))
#         idx += 1
#         if idx == 20: break
#  rating[int(row['movieId']), int(row['userId'])] = row['rating']
# out = open("dataset/key.txt",'a')
# csv_write = csv.writer(out)



# for index,row in test_data.iterrows():
    # list = []
    # list.append(predicts[int(row['movieId']),int(row['userId'])])
    # csv_write.writerow(list)
