from tensorflow.keras.layers import  Conv2D, concatenate, AveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.activations import tanh
import tensorflow as tf
from tensorflow.experimental import numpy as tnp
from tensorflow.keras.layers import Lambda
import os, sys
import tensorflow as tf

def mask_mse(y_true, y_pred):
    weights = tf.cast(tf.not_equal(y_true, 999), tf.float32)

    d1_tr = tnp.diff(y_true, axis=1)
    d1_hat = tnp.diff(y_pred, axis=1)
    d1_sse = tnp.sum((d1_tr - d1_hat)**2 * weights[:,1:], axis=1)
    d1_mse = d1_sse / tnp.sum(weights[:,1:], axis=1)

    value_sse = tnp.sum((y_true - y_pred)**2 * weights, axis=1)
    value_mse = value_sse / tnp.sum(weights, axis=1)
    masked_loss = value_mse + d1_mse
    return masked_loss


def build_model(m_2d, m_dt, m_woa, learning_rate, verbal=True):
    m2_1 = Dense(8, activation='elu', name='Dense_elu')(m_2d)
    m2_2 = Dense(8, activation='tanh', name='Dense_tanh')(m_2d)
    m2_3 = Dense(8, activation='gelu', name='Dense_gelu')(m_2d)
    m2_4 = Dense(8, activation='linear', name='Dense_linear1')(m_2d)

    m = concatenate([m2_1,m2_2,m2_3,m2_4], name='Concatenate1')
    m = Dense(32, name='Dense_linear2')(m)
    m2 = Conv2D(64, (4,4), activation='linear', name='Conv2D_linear1')(m)
    m2 = AveragePooling2D((2,2), name='AvgPool2D')(m2)
    m2 = Conv2D(128, (3, 3), activation='linear', name='Conv2D_linear2')(m2)
    m2 = Conv2D(128, (3, 3), activation='linear', name='Conv2D_linear3')(m2)

    m = Lambda(lambda x: tf.squeeze(x, axis=[1,2]), name='Squeeze')(m2)

    m = concatenate([m, m_dt, m_woa], name='Concatenate2')

    m = Dense(128, name='Dense_linear3')(m)
    m = Dense(64, name='Dense_linear4')(m)
    m = Dense(32, name='Dense_linear5')(m)
    m = Dense(13, name='Dense_linear6')(m)

    m=Model(inputs=[m_2d, m_dt, m_woa], outputs=m)

    m.compile(loss=mask_mse, optimizer='adam', weighted_metrics=tf.keras.losses.mse)
    return m
