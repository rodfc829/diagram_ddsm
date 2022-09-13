""" Funções construídas e utilizadas para a implementação dos pesos nos exemplos de treinamento
Importante notar que a função loss deverá ser substituída pela função de perda do trecho de rede adequado,
e o objeto model_optimizer deverá ser substituído pelo otimizador do trecho de rede adequado """

import numpy as np
import tensorflow as tf
from tensorflow import keras


def delta_func(x):
    if x == 0:
        return 1
    return 0


def auto_diff_step(model, alpha, images_f, labels_f, images_g, labels_g):
    n = images_f.shape[0]
    m = images_g.shape[0]
    theta = model.get_weights()

    with tf.GradientTape() as tape1:
        y_f = model.predict(images_f)
        eps = np.zeros(n)
        cost_f = loss(y_f, labels_f)
        l_f = np.sum(np.dot(eps, cost_f))
    delta_theta = tape1.gradient(l_f, theta)
    theta_hat = theta - alpha * delta_theta

    with tf.GradientTape() as tape2:
        y_g = model.set_weights(theta_hat).predict(images_g)
        l_g = (1/m) * np.sum(loss(y_g, labels_g))
    delta_eps = tape2.gradient(l_g, eps)

    omega_aux = np.maximum(-delta_eps, np.zeros_like(delta_eps))
    omega = omega_aux/(np.sum(omega_aux) + delta_func(np.sum(omega_aux)))

    with tf.GradientTape as tape3:
        lf_hat = np.sum(np.dot(omega, cost_f))
    delta_theta = tape3.gradient(lf_hat, theta)
    model_optimizer.apply_gradients(zip(delta_theta, model.trainable_variables))
