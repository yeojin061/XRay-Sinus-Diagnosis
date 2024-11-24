from tensorflow.keras.callbacks import Callback
from tqdm.notebook import tqdm

class TQDMProgressBar(Callback):
    def __init__(self, epochs):
        self.epochs = epochs
        self.epoch_progress_bar = None

    def on_train_begin(self, logs=None):
        self.epoch_progress_bar = tqdm(total=self.epochs, desc='Epochs', position=0, leave=True)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_progress_bar.update(1)
        print(f"Epoch {epoch+1}/{self.epochs} - loss: {logs['loss']:.4f} - "
              f"left_output_accuracy: {logs['left_output_accuracy']:.4f} - "
              f"right_output_accuracy: {logs['right_output_accuracy']:.4f}")

    def on_train_end(self, logs=None):
        self.epoch_progress_bar.close()
