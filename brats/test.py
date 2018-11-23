import sys
sys.path.append("C://Users//minhm//Documents//GitHub//3DUnetCNN")


from unet3d.utils import pickle_dump, pickle_load

data = [158, 66]

pickle_dump(data, 'abc.pkl')

a, b = pickle_load('abc.pkl')

print (a,b)