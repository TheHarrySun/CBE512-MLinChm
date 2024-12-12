import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X, Xsc, X_pocket, Xsc_pocket, Y, Ysc, Y_pocket, Ysc_pocket = preprocessing.main()

keras.utils.set_random_seed(0)

def layer_builder(nodearray):
    model = layers.Dense(nodearray[0], activation='relu')
    for i in range(1, len(nodearray)):
        if (nodearray[i] < 1):
            model = layers.Dropout(nodearray[i])(model)
        else:
            model = layers.Dense(nodearray[i], activation='relu')(model)
    return model

ligand_features = len(X[0]) - 5

total_X = np.concatenate((Xsc, Xsc_pocket), axis = 1)
print(total_X.shape)

X_train, X_test, Y_train, Y_test = train_test_split(total_X, Ysc, test_size = 0.2, random_state = 0)

ligand_train = X_train[:,0:ligand_features]
pocket_train = X_train[:,ligand_features + 5:]
ligand_test = X_test[:,0:ligand_features]
pocket_test = X_test[:,ligand_features + 5:]

pocket_features = pocket_train.shape[1]
ligand_features = ligand_train.shape[1]

ligandInput = keras.Input(shape=(ligand_features,), name="ligandinput")

model1_nodes = [200, 0.2]

ligandlayer1 = layers.Dense(200, activation = 'relu', name = 'l1')
dropout1 = layers.Dropout(0.2)
ligandlayer2 = layers.Dense(100, activation='relu', name='l2')
dropout2 = layers.Dropout(0.2)
ligandlayer3 = layers.Dense(50, activation='relu', name='l3')
dropout3 = layers.Dropout(0.2)
layer4 = layers.Dense(25, activation='relu', name='l4')


pocketInput = keras.Input(shape=(pocket_features,), name='pocketinput')
pocketlayer1 = layers.Dense(75, activation='relu', name = 'p1')
pocketlayer2 = layers.Dense(100, activation='relu')

output = layers.Dense(1, activation=None, name='output')

ligand_model = ligandlayer1(ligandInput)
pocket_model = (pocketlayer1(pocketInput))
network = output((dropout2(ligandlayer2(dropout1(layers.concatenate([ligand_model, pocket_model]))))))

opt = tf.keras.optimizers.Nadam()
lossfunc = keras.losses.MeanSquaredError(reduction="sum_over_batch_size")

model = keras.Model(inputs=[ligandInput, pocketInput], outputs = network)
model.summary()

callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience = 50, mode = "min")
checkpoint_filepath = "checkpoint.model.keras"
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only = True)

model.compile(optimizer = opt, loss = lossfunc)
history = model.fit(x=[ligand_train, pocket_train], y = Y_train, batch_size = 32, epochs = 500, validation_split = 0.2, callbacks=[callback, checkpoint_callback])

train_ypred = model.predict([ligand_train, pocket_train])
test_ypred = model.predict([ligand_test, pocket_test])
print(r2_score(Y_train, train_ypred))
print(r2_score(Y_test, test_ypred))

plt.figure(figsize = (8, 6))
plt.scatter(train_ypred, Y_train, color = 'b', alpha = 0.5)
x = np.linspace(-5, 5)
plt.plot(x, x, color = 'r', alpha = 0.5)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Training Predicted Binding Affinities vs. Actual Binding Affinities")
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.show()

plt.figure(figsize = (8, 6))
plt.scatter(test_ypred, Y_test, color = 'b', alpha = 0.5)
x = np.linspace(-5, 5)
plt.plot(x, x, color = 'r', alpha = 0.5)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Testing Predicted Binding Affinities vs. Actual Binding Affinities")
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# random forest regression

# if you wnat to do conv3d model, then you should take public conv3D; vgd16 (transfer learning stuff)

# shap analysis; random forset regression bar graph of importance value


# scaled data is a must; otherwise we get negative r2 values
# highest value of r2 value for test set is 0.359 with 200, 100, 50, 10 and 100 and all dropout layers with 0.2