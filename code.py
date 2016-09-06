import numpy as np
import operator
#导入数据 创建数据集
#随意创建的一个小数据啦哈哈
def creatDataSet():
    point=np.array([[1,1],[1,1.1],[0,0],[0,0.1]])
    group=['A','A','B','B']
    return point,group

#KNN进行分类,input是输入的向量,dataset是数据集,k为最近邻居个数,返回k个邻居和分类名
#这里计算的是欧氏距离
def classify(input,dataset,labels,k):
    datasetSize=dataset.shape
    DIST=[]
    index=0
    for row in dataset:
        tmp=input-row
        tmp2=tmp*tmp
        dist=0
        for num in tmp2:
            dist+=num
        elem=(dist,index,dataset[index:index+1,:],labels[index]) #elem由距离dist，标号index，数据向量，类别组成
        DIST.append(elem)
        index+=1
    DIST.sort(key=lambda d:d[0]) #按照dist排序 得到最近的k个邻居
    KNearNeighbour=np.array(DIST[0][2])
    labelCount={}
    labelCount[DIST[0][3]]=1
    for i in range(1,k):
        tmp=np.vstack((KNearNeighbour,DIST[i][2])) #合并邻居集
        KNearNeighbour=tmp
        if DIST[i][3] in labelCount: #统计各个类别的出现频率
            labelCount[DIST[i][3]]+=1
        else:
            labelCount[DIST[i][3]]=1
    labelSort=sorted(labelCount.items(),key=lambda d:d[1]) #统计出出现次数最多的类别
    label=labelSort[0]
    return KNearNeighbour,label

if __name__=='__main__':
    dataset,group=creatDataSet()
    input=[0,0.3] #小小的测试一下
    neighbour,label=classify(input,dataset,group,2)
    print(neighbour)
    print(label)
