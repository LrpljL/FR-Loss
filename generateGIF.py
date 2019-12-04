import matplotlib.pyplot as plt
import imageio,os
images = []
cos_images=[]
dir = '/home/ubuntu/LRP/FR-Loss-on-Mnist-master/ArcSoft/'
filenames=sorted((fn for fn in os.listdir(dir) if fn.endswith('.jpg')))
filenames.sort()
i=0
for filename in filenames:
    images.append(imageio.imread(dir + 'epoch=' + str(i) + '.jpg'))
    cos_images.append(imageio.imread(dir+'cos_epoch='+str(i)+'.jpg'))
    i+=1
    if i==len(filenames)/2:
        break
imageio.mimsave('r_insight.gif', images,duration=1)
imageio.mimsave('cos_insight.gif', cos_images,duration=1)