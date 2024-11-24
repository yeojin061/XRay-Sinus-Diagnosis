from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf

def create_binary_classification_model(input_shape):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.7)(x)
    left_output = Dense(2, activation='softmax', name='left_output')(x)
    right_output = Dense(2, activation='softmax', name='right_output')(x)

    model = Model(inputs=base_model.input, outputs=[left_output, right_output])
    return model
