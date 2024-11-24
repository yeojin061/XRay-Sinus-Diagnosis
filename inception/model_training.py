from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr * 0.5
    elif epoch < 30:
        return lr * 0.2
    else:
        return max(lr * 0.1, 1e-6)

lr_scheduler = LearningRateScheduler(scheduler)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

def compile_and_train_binary_model(model, train_images, train_labels_left, train_labels_right, val_images,
                                   val_labels_left, val_labels_right, batch_size=32, epochs=30, callbacks=None):
    if callbacks is None:
        callbacks = []

    optimizer = Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer,
                  loss={'left_output': 'categorical_crossentropy', 'right_output': 'categorical_crossentropy'},
                  metrics={'left_output': 'accuracy', 'right_output': 'accuracy'})

    model.fit(train_images, {'left_output': train_labels_left, 'right_output': train_labels_right},
              validation_data=(val_images, {'left_output': val_labels_left, 'right_output': val_labels_right}),
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks)
