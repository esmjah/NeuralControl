from typing import Optional, List
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
# import tensorflow.contrib.eager as tfe

keras = tf.keras


def zip_map(zipped, update_op):
    return [update_op(*elems) for elems in zipped]


def euler_update(h_list, dh_list, dt):
    return zip_map(zip(h_list, dh_list), lambda h, dh: h + tf.cast(dt, h.dtype) * dh)


def euler_step(func, dt, state):
    return euler_update(state, func(state), dt)


def rk2_step(func, dt, state):
    k1 = func(state)
    k2 = func(euler_update(state, k1, dt))
    return zip_map(zip(state, k1, k2),
                   lambda h, dk1, dk2: h + tf.cast(dt, h.dtype) * (dk1 + dk2) / 2)


def rk4_step(func, dt, state):
    k1 = func(state)
    k2 = func(euler_update(state, k1, dt / 2))
    k3 = func(euler_update(state, k2, dt / 2))
    k4 = func(euler_update(state, k3, dt))

    return zip_map(
        zip(state, k1, k2, k3, k4),
        lambda h, dk1, dk2, dk3, dk4: h + tf.cast(dt, h.dtype) * (
                dk1 + 2 * dk2 + 2 * dk3 + dk4) / 6,
    )


class NeuralODE:
    def __init__(
            self, model: tf.keras.Model, t=np.linspace(0, 1, 40),
            solver=rk4_step
    ):
        self._t = t
        self._model = model
        self._solver = solver
        self._deltas_t = t[1:] - t[:-1]

    def forward(self, inputs: tf.Tensor, u: np.array, return_states: Optional[str] = None):

        def _forward_dynamics(_state):
            """Used in solver _state == (time, tensor)"""
            return [1.0, self._model(inputs=_state)]

        states = []

        def _append_state(_state):
            tensors = _state[1]
            if return_states == "numpy":
                states.append(tensors.numpy())
            elif return_states == "tf":
                states.append(tensors)

        with tf.name_scope("forward"):
            t0 = tf.cast(self._t[0], dtype=tf.float32)
            dim_u = int(u.size/u.shape[0])
            dim_inputs = len(inputs)
            state = [t0, inputs]
            _append_state(state)
            deltas_t = self._deltas_t
            for k in range(len(deltas_t)):
                ux = state[1]
                x = ux[dim_u:len(ux)] # tf.reshape(ux[1:len(ux)], [2, 1])
                u_tf = tf.cast(u[k], dtype=tf.float32)
                state = [t0, tf.concat([u_tf, x], axis=0)]

                state = self._solver(
                    func=_forward_dynamics, dt=tf.cast(deltas_t[k], dtype=tf.float32), state=state
                )
                _append_state(state)

        outputs = state[1]
        if return_states:
            states_stack = tf.stack(states)
            return outputs, states_stack
        return outputs

    def _backward_dynamics(self, state):
        t = state[0]
        ht = state[1]
        at = -state[2]

        with tf.GradientTape() as g:
            g.watch(ht)
            ht_new = self._model(inputs=[t, ht])

        gradients = g.gradient(
            target=ht_new, sources=[ht] + self._model.weights,
            output_gradients=at
        )

        return [1.0, ht_new, *gradients]

    def backward(self, outputs: tf.Tensor,
                 output_gradients: Optional[tf.Tensor] = None):

        with tf.name_scope("backward"):
            grad_weights = [tf.zeros_like(w) for w in self._model.weights]
            t0 = tf.cast(self._t[-1], dtype=tf.float32)

            if output_gradients is None:
                output_gradients = tf.zeros_like(outputs)

            state = [t0, outputs, output_gradients, *grad_weights]
            for dt in self._deltas_t[::-1]:
                state = self._solver(
                    self._backward_dynamics, dt=-tf.cast(dt, dtype=tf.float32), state=state
                )

        inputs = state[1]
        dLdInputs = state[2]
        dLdWeights = state[3:]
        return inputs, dLdInputs, dLdWeights

    def forward_odeint(
            self,
            inputs: tf.Tensor,
            rtol=1e-6,
            atol=1e-6,
            method='dopri5',
            return_states: bool = False,
    ):
        """Do forward with adaptive solver"""
        with tf.name_scope("forward_odeint"):
            t = tf.cast(self._t, dtype=tf.float32)
            if not return_states:
                t = tf.cast([t[0], t[-1]], dtype=tf.float32)

            outputs, info_dict = tf.contrib.integrate.odeint(
                func=lambda _y, _t: self._model(inputs=(_t, _y)),
                y0=inputs,
                t=t,
                rtol=rtol,
                atol=atol,
                method=method,
                full_output=True,
            )

        if return_states:
            return outputs, info_dict

        return outputs[-1, ...], info_dict


def defun_neural_ode(node: NeuralODE) -> NeuralODE:
    node.forward = tf.function(node.forward)
    node.backward = tf.function(node.backward)
    node.forward_odeint = tf.function(node.forward_odeint)
    return node
