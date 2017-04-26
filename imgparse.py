import numpy as np
import pickle
def unpickle(filename):
    with open(filename, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic

def parse_data(data,num_pictures):
    for n in range(num_pictures):#iterate over each row being one image
        img=np.ones([32,32,3])
        for i in range(np.size(data,1)):#iterate over each term in the row
            channel=np.floor(i/1024)#returns [0,1,2] for [r,g,b]
            row=np.floor((i-channel*1024)/32)
            column=(i-1024*channel)%32
            img[int(row),int(column),int(channel)]=data[n,i]
        if n==0:
            imglist=[img]
        else:
            imglist.append(img)
    return np.array(imglist)

#numimgs=10000;
#d=unpickle('/users/justin/Documents/tensorflow/cifar-10/data_batch_1')
#data=d[b'data']
#masterimgs=parse_data(data,numimgs)
#print(masterimgs.shape)
#d=unpickle('/users/justin/Documents/tensorflow/cifar-10/data_batch_2')
#data=d[b'data']
#imgs=parse_data(data,numimgs)
#masterimgs=np.append(masterimgs,imgs,0);
#print(masterimgs.shape)
#d=unpickle('/users/justin/Documents/tensorflow/cifar-10/data_batch_3')
#data=d[b'data']
#imgs=parse_data(data,numimgs)
#masterimgs=np.append(masterimgs,imgs,0);
#print(masterimgs.shape)
#d=unpickle('/users/justin/Documents/tensorflow/cifar-10/data_batch_4')
#data=d[b'data']
#imgs=parse_data(data,numimgs)
#masterimgs=np.append(masterimgs,imgs,0);
#print(masterimgs.shape)
#d=unpickle('/users/justin/Documents/tensorflow/cifar-10/data_batch_5')
#data=d[b'data']
#imgs=parse_data(data,numimgs)
#masterimgs=np.append(masterimgs,imgs,0);
#print(masterimgs.shape)
#np.save('/users/justin/Documents/tensorflow/images/cifar/images.npy',masterimgs)

#test images
d=unpickle('/users/justin/Documents/tensorflow/cifar-10/test_batch')
data=d[b'data']
imgs=parse_data(data,10000)
np.save('/users/justin/Documents/tensorflow/images/cifar/test.npy',imgs)
