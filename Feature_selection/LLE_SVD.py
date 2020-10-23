


def lle(data,n_components=10):
    embedding = LocallyLinearEmbedding(n_components=n_components)
    X_transformed = embedding.fit_transform(data)
    return X_transformed
def nmf(data,n_components=10):
    model=NMF(n_components=n_components)
    X_new= model.fit_transform(data) 
    return X_new
def svd(data,n_components=10):
    SVD = TruncatedSVD(n_components=n_components)
    new_data=SVD.fit_transform(data)  
    return new_data


row=data.shape[0]
column=data.shape[1]
index = [i for i in range(row)]
np.random.shuffle(index)
index=np.array(index)
data_=data[index,:]
shu=data_[:,np.array(range(1,column))]


n_num=100 
data_9=lle(shu,n_components=n_num)
X1=data_9
data_11=svd(shu,n_components=n_num)
X3=data_11
label[label==-1]=0
y=label
