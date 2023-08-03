import random

f = open('frame_00003.txt', 'rb')
byt = f.read()
res = str(byt).split("\\r\\n")
res[0] = '85,199'
tmp = []
for i in res:
    i += ','+str(random.randint(30, 60))+','+str(random.randint(30, 60))
    tmp.append(i)
r = "\r".join(tmp[:-1])
f.close()


f = open('03.txt', 'w')
f.write(r)
f.close()