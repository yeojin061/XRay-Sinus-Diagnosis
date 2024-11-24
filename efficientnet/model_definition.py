from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf

def create_efficientnet_model(input_shape, num_classes):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    x = Flatten()(base_model.output)
    x = Dropout(0.5)(x)
    left_output = Dense(num_classes, activation='softmax', name='left_output')(x)
    right_output = Dense(num_classes, activation='softmax', name='right_output')(x)
    model = Model(inputs=base_model.input, outputs=[left_output, right_output])
    return model

def custom_loss(y_true, y_pred):
    mask = tf.reduce_max(y_true, axis=-1)
    mask = tf.cast(mask != 0, tf.float32)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    loss = loss * mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)
