import pickle

def reader(filename):
    file=open(filename,"rb")
    data=pickle.load(file)
    file.close()
    return data

data=reader("testdatalabels.pickle");
print(data)