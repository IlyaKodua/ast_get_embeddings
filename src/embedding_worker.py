import pickle
import glob
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from cycler import cycler
from matplotlib.colors import hsv_to_rgb
from sklearn import svm
from sklearn.metrics import accuracy_score

clf = svm.SVC()


def what_class_in_file(classes, file):
    leb = np.zeros_like(classes)
    for i, cls in enumerate(classes):
        if cls in file:
            leb[i]= 1

    return np.argmax(leb)



classes = ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"]

anomaly_type = ["normal", "anomaly"]



data_dir = "out/"


list_files = glob.glob(data_dir + '**/*.pkl', recursive=True)

embeddings = [] #dict(zip(classes, [[]]*len(classes)))
labeles = []

for i, file in enumerate(list_files):
    cycl_perent = int((i+1)/len(list_files)*100)
    print(cycl_perent, " %")
    if anomaly_type[0] in file:

        with open(file, 'rb') as f:
            data = pickle.load(f)

        cls = what_class_in_file(classes, file)
        embeddings.append(data)
        labeles.append(cls)


clf.fit(embeddings, labeles)


print("acc ", accuracy_score(labeles, clf.predict(embeddings)))
# colors = ['r.', 'g.', 'b.', 'y.', 'k.', 'c.', 'm.']


# for cls,color in zip(classes, colors):
#     pca = PCA(n_components=2)
#     new_emb = pca.fit_transform(embeddings[cls])
#     plt.plot(new_emb[:,0], new_emb[:,1], color)
#     plt.show()
#     print(cls)






