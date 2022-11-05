from umap import UMAP
import umap.plot
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
import holoviews
import datashader
from bokeh.plotting import show, save, output_notebook, output_file
from bokeh.resources import INLINE
import numpy
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import normalize


# data_X = loadmat('cD_nlos')
# data_X_nlos = np.abs(data_X.get('cD'))
# print((data_X_nlos))
#
# data_X = loadmat('cD_los')
# data_X_los = np.abs(data_X.get('cD'))
# print((data_X_los))
#
# data_Y = loadmat('data_Y_freq_los')
# data_Y_los = np.abs(data_Y.get('data_Y_freq_los'))
#
# data_Y = loadmat('data_Y_freq_nlos')
# data_Y_nlos = np.abs(data_Y.get('data_Y_freq_nlos'))
#
# X_data = np.concatenate((data_X_los,data_X_los,data_X_los,data_X_los,data_X_los,data_X_los,data_X_los,data_X_los,data_X_los,data_X_los,data_X_nlos,
#                          data_X_nlos,data_X_nlos,data_X_nlos,data_X_nlos,data_X_nlos,data_X_nlos,data_X_nlos,data_X_nlos,data_X_nlos),axis=0)
# Y_data = np.concatenate((data_Y_los,data_Y_los,data_Y_los,data_Y_los,data_Y_los,data_Y_los,data_Y_los,data_Y_los,data_Y_los,data_Y_los,data_Y_nlos,
#                          data_Y_nlos,data_Y_nlos,data_Y_nlos,data_Y_nlos,data_Y_nlos,data_Y_nlos,data_Y_nlos,data_Y_nlos,data_Y_nlos),axis=0)
#
# for i in range(0,3):
#     X_data = np.concatenate((X_data,X_data),axis=0)
#     Y_data = np.concatenate((Y_data,Y_data),axis=0)
# X_data = normalize(X_data, axis=0, norm='max')
#
#
#
# print(Y_data)
# #
# Data=np.concatenate((X_data,Y_data),axis=1)
#
#
#
# print(Data)
#
# x_train = X_data[0:10000,:]
# x_test = X_data[10000:16000,:]
#
# y_train = Y_data[0:10000]
# y_test = Y_data[10000:16000]
#
# y_train=np.reshape(y_train,10000,)
# y_test=np.reshape(y_test,6000,)
#
#
#

def data_gen(test_percentage):
    data_X = loadmat('Data_x_new')
    data_X_1 = np.abs(data_X.get('Data_x'))
    print((data_X_1))



    data_Y = loadmat('Data_y_new')
    data_Y_1 = np.array(data_Y.get('Data_y')).astype('int')



    for i in range(0,6):
        data_X_1 = np.concatenate((data_X_1,data_X_1),axis=0)
        data_Y_1 = np.concatenate((data_Y_1, data_Y_1), axis=0)

    data_X_new = data_X_1[:,0:576]
    data_X_new = normalize(data_X_new, axis=0, norm='max')
    Data = np.concatenate((data_X_new,data_Y_1),axis=1)

    data_length = len(Data)
    col_length = len(Data[0])-1
    np.random.shuffle(Data)

    train_split = int(np.floor(data_length*test_percentage))
    x_train = Data[0:train_split, 0:col_length]
    x_test = Data[train_split:data_length, 0:col_length]

    y_train = Data[0:train_split,col_length]
    y_test = Data[train_split:data_length,col_length]

    y_train = np.reshape(y_train, train_split, )
    y_test = np.reshape(y_test, data_length-train_split, )

    return x_train,y_train,x_test,y_test


x_train,y_train,x_test,y_test=data_gen(.8)



#
#
# model = UMAP(n_neighbors = 50, min_dist = .1, n_components = 2, verbose = True)
# trans = model.fit_transform(x_train)
# fig = plt.figure(figsize=(6, 6))
# plt.scatter(trans[:, 0], trans[:, 1],c=y_train,cmap='Paired')
# plt.title('Embedding', fontsize=4);
# fig.savefig('final_t.png', dpi=fig.dpi)


#
# z_test = np.load('latent.npy')
# y_test = np.load('y_test.npy')

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(z_test[:, 0], z_test[:, 1], z_test[:,2], c=y_test, alpha = 1, s=3 ** 2, cmap='Paired')
#
#
# ax.view_init(60, 35)
# plt.show()
# plt.savefig('scatter3d.png')



model = UMAP(random_state=42,n_neighbors = 100, min_dist = 0, n_components = 2, verbose = True)
trans = model.fit_transform(x_train)
fig = plt.figure(figsize=(6, 6))
plt.scatter(trans[:, 0], trans[:, 1],c=y_train,cmap='Paired')
plt.title('Embedding', fontsize=24);
fig.savefig('final_channel_latent_train.png', dpi=fig.dpi)


model = UMAP(random_state=42,n_neighbors = 100, min_dist = 0, n_components = 2, verbose = True)
trans = model.fit_transform(x_test)
fig = plt.figure(figsize=(6, 6))
plt.scatter(trans[:, 0], trans[:, 1],c=y_test,cmap='Paired')
plt.title('Embedding', fontsize=24);
fig.savefig('final_channel_latent_test.png', dpi=fig.dpi)



# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(trans[:, 0], trans[:, 1], trans[:,2] ,c=y_test, alpha = 1, s=3 ** 2, cmap='Paired')
#
# ax.view_init(60, 35)
# plt.show()
# plt.savefig('scatter13d.png')