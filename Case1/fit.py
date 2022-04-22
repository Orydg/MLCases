from livelossplot.tf_keras import PlotLossesCallback
from model import model
import batch_data as bd

EPOCHS = 30
train_data_gen, val_data_gen = bd.batches()
history = model.fit_generator(
    train_data_gen,
    epochs=EPOCHS,
    validation_data=val_data_gen,
    callbacks=[PlotLossesCallback()])