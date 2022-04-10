import random
import csv

with open('Prostate_Cancer.csv', 'r') as file:
    reader = csv.DictReader(file)
    datas = [row for row in reader]

random.shuffle(datas)
n = len(datas)//3

test_set = datas[0:n]
train_set = datas[n:]

# 距离计算
def distance(d1, d2):
    res = 0

    for key in ('radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'symmetry', 'fractal_dimension'):
        res += (float(d1[key]) - float(d2[key]))**2

    return res**0.5

K = 5
def knn(data):
    # 1.待测数据data与训练数据的距离 (字典列表)
    res = [
        {'result':train['diagnosis_result'], 'distance':distance(data, train)}
        for train in train_set
    ]

    # 2.排序--升序
    res = sorted(res, key=lambda item:item['distance'])

    # 3. 取前K个
    res2 = res[0:K]

    # 4. 加权平均
    result = {'B':0, 'M':0}

    sum = 0
    for r in res2:
        sum += r['distance']  # 总距离

    for r in res2:
        result[r['result']] += (1-r['distance']/sum)

    if result['B'] > result['M']:
        return 'B'
    else:
        return 'M'


# 测试
correct = 0
for test in test_set:
    result_true = test['diagnosis_result']
    result_test = knn(test)

    if result_test == result_true:
        correct+=1


print('rate:%s'%(correct/len(test_set)*100) )
