from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from sklearn.decomposition import PCA
import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

# Load images and sort into arrays
def load_images(folder_path, label):
    images = []
    for filename in os.listdir(folder_path):
        try:
            img = Image.open(os.path.join(folder_path, filename)).convert('L')
            images.append(np.array(img).flatten())
        except:
            pass
    return np.array(images), np.array([label] * len(images))

def load_validation_images(folder_path):
    images = []
    labels = []

    for filename in os.listdir(folder_path):

        try:
            path = os.path.join(folder_path, filename)

            img = Image.open(path).convert('L').resize((300, 300))

            img_array = np.array(img).flatten()

            images.append(img_array)

            if filename[0] == "p":
                labels.append("paper")

            elif filename[0] == "s":
                labels.append("scissors")

            elif filename[0] == "r":
                labels.append("rock")

        except:
            pass

    return np.array(images), np.array(labels)


X_validation, y_validation = load_validation_images('validation')

X_paper_train, y_paper_train = load_images('paper',"paper")
X_scissor_train, y_scissor_train = load_images('scissors',"scissors")
X_rock_train, y_rock_train = load_images('rock',"rock")

X_train = np.vstack([X_rock_train, X_paper_train, X_scissor_train])
y_train = np.concatenate([y_rock_train, y_paper_train, y_scissor_train])

x_paper_test, y_paper_test = load_images('paper_test',"paper")
x_scissor_test, y_scissor_test = load_images('scissors_test',"scissors")
x_rock_test, y_rock_test = load_images('rock_test',"rock")

x_test = np.vstack([x_rock_test, x_paper_test, x_scissor_test])
y_test = np.concatenate([y_rock_test, y_paper_test, y_scissor_test])

# investigate set
plt.imshow(X_train[0].reshape(300, 300))
plt.show()

print("Size of train-set: ", X_train.shape, " distribution: " , y_train.tolist().count("paper"), y_train.tolist().count("rock"), y_train.tolist().count("scissors"))
print("Size of test-set: ", x_test.shape, " distribution: " , y_test.tolist().count("paper"), y_test.tolist().count("rock"), y_test.tolist().count("scissors"))
print("Size of validation-set: ", X_validation.shape, " distribution: " , y_validation.tolist().count("paper"), y_validation.tolist().count("rock"), y_validation.tolist().count("scissors"))

# find relevant amount of components for PCA (elbow point in curve)
pca = PCA(n_components=1000, whiten=True, random_state=0).fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()

# Apply PCA
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(x_test)
X_validation_pca = pca.transform(X_validation)

# Plot after PCA
plt.imshow(X_validation_pca[0].reshape(10,10),cmap="Grays")
plt.show()

# visualize using U-map
from sklearn.utils import shuffle
X_shuffled, y_shuffled = shuffle(X_train_pca, y_train, random_state=42)


reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_shuffled)

plt.scatter(X_umap[:, 0], X_umap[:, 1], c=[{'paper':0,'rock':1,'scissors':2}[l] for l in y_shuffled], cmap='viridis', s=5)
plt.colorbar()
plt.show()

#t-sne
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_shuffled)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
            c=[{'paper':0,'rock':1,'scissors':2}[l] for l in y_train],
            s=10, alpha=0.5, cmap='viridis')
plt.colorbar()
plt.show()

# Find best K for k nearest neighbors
acc_train = []
acc_test = []
K = range(1,51)
for kk in K:
    clf = KNeighborsClassifier(n_neighbors=kk)
    clf.fit(X_train_pca, y_train)
    acc_train.append(clf.score(X_train_pca,y_train))
    acc_test.append(clf.score(X_validation_pca,y_validation))

fig, ax = plt.subplots()
plt.plot(K,acc_train)
plt.plot(K,acc_test)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend(['Training','Validation'])
plt.grid()
plt.show()

best_k = np.argmax(acc_test) + 1
print(f"Bästa k: {best_k}, Accuracy: {acc_test[best_k-1]:.4f}")

# train model and print confusion table

clf = KNeighborsClassifier(n_neighbors=29)
clf.fit(X_train_pca, y_train)
predictions = clf.predict(X_validation_pca)
confusion = confusion_matrix(y_validation, predictions, labels=['paper', 'rock', 'scissors'])

sns.heatmap(confusion, annot=True, fmt='d',
            xticklabels=['paper', 'rock', 'scissors'],
            yticklabels=['paper', 'rock', 'scissors'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Same procedure, but before PCA
acc_train = []
acc_test = []

K = range(1,51)
for kk in K:
    clf = KNeighborsClassifier(n_neighbors=kk)
    clf.fit(X_train, y_train)
    acc_train.append(clf.score(X_train,y_train))
    acc_test.append(clf.score(X_validation,y_validation))

fig, ax = plt.subplots()
plt.plot(K,acc_train)
plt.plot(K,acc_test)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend(['Training','Validation'])
plt.grid()
plt.show()

best_k = np.argmax(acc_test) + 1
print(f"Bästa k: {best_k}, Accuracy: {acc_test[best_k-1]:.4f}")

# train KNN model without PCA
clf = KNeighborsClassifier(n_neighbors=best_k)
clf.fit(X_train, y_train)
predictions = clf.predict(X_validation)
confusion = confusion_matrix(y_validation, predictions, labels=['paper', 'rock', 'scissors'])

sns.heatmap(confusion, annot=True, fmt='d',
            xticklabels=['paper', 'rock', 'scissors'],
            yticklabels=['paper', 'rock', 'scissors'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Train random forest classifier, print accuracy and confusion table

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_pca, y_train)

print("Train accuracy:", rf.score(X_train_pca, y_train))
print("Validation accuracy:", rf.score(X_validation_pca, y_validation))

predictions = rf.predict(X_validation_pca)
confusion = confusion_matrix(y_validation, predictions, labels=['paper', 'rock', 'scissors'])

sns.heatmap(confusion, annot=True, fmt='d',
            xticklabels=['paper', 'rock', 'scissors'],
            yticklabels=['paper', 'rock', 'scissors'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Without PCA

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

print("Train accuracy:", rf.score(X_train, y_train))
print("Validation accuracy:", rf.score(X_validation, y_validation))


predictions = rf.predict(X_validation)
confusion = confusion_matrix(y_validation, predictions, labels=['paper', 'rock', 'scissors'])

sns.heatmap(confusion, annot=True, fmt='d',
            xticklabels=['paper', 'rock', 'scissors'],
            yticklabels=['paper', 'rock', 'scissors'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Train SVC model

svc = SVC(kernel='rbf')
svc.fit(X_train_pca, y_train)

print("Train accuracy:", svc.score(X_train_pca, y_train))
print("Validation accuracy:", svc.score(X_validation_pca, y_validation))

predictions = svc.predict(X_validation_pca)
confusion = confusion_matrix(y_validation, predictions, labels=['paper', 'rock', 'scissors'])

sns.heatmap(confusion, annot=True, fmt='d',
            xticklabels=['paper', 'rock', 'scissors'],
            yticklabels=['paper', 'rock', 'scissors'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Without PCA

svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)

print("Train accuracy:", svc.score(X_train, y_train))
print("Validation accuracy:", svc.score(X_validation, y_validation))

predictions = svc.predict(X_validation)
confusion = confusion_matrix(y_validation, predictions, labels=['paper', 'rock', 'scissors'])

sns.heatmap(confusion, annot=True, fmt='d',
            xticklabels=['paper', 'rock', 'scissors'],
            yticklabels=['paper', 'rock', 'scissors'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# CNN

# Normalize
X_train_cnn = X_train.reshape(-1, 300, 300, 1) / 255.0
X_test_cnn = x_test.reshape(-1, 300, 300, 1) / 255.0
X_val_cnn = X_validation.reshape(-1, 300, 300, 1) / 255.0

# Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
y_val_enc = le.transform(y_validation)

# Build CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

model.fit(X_train_cnn, y_train_enc, epochs=10,
          validation_data=(X_val_cnn, y_val_enc),
          callbacks=[early_stop])

y_pred_cnn = np.argmax(model.predict(X_val_cnn), axis=1)
labels = le.classes_

confusion = confusion_matrix(y_val_enc, y_pred_cnn)

sns.heatmap(confusion, annot=True, fmt='d',
            xticklabels=labels,
            yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Test and print final results
print("Test accuracy:", model.evaluate(X_test_cnn, y_test_enc)[1])

y_pred_cnn = np.argmax(model.predict(X_test_cnn), axis=1)
labels = le.classes_

confusion = confusion_matrix(y_test_enc, y_pred_cnn)

sns.heatmap(confusion, annot=True, fmt='d',
            xticklabels=labels,
            yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()