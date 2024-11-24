import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from custom_callbacks import TQDMProgressBar

def scheduler(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * 0.1

lr_scheduler = LearningRateScheduler(scheduler)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

def compile_and_train_model_efficientnet(model, train_images, train_labels_left, train_labels_right,
                                         val_images, val_labels_left, val_labels_right, batch_size=64, epochs=50):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                  loss={'left_output': custom_loss, 'right_output': custom_loss},
                  metrics={'left_output': 'accuracy', 'right_output': 'accuracy'})

    model.fit(train_images, {'left_output': train_labels_left, 'right_output': train_labels_right},
              validation_data=(val_images, {'left_output': val_labels_left, 'right_output': val_labels_right}),
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[early_stopping, lr_scheduler, TQDMProgressBar(epochs=epochs)])
