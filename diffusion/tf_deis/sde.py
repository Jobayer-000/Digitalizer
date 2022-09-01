import abc
import numpy as np
import tensorflow as tf


class ExpSDE(abc.ABC):
    @property
    @abc.abstractmethod
    def sampling_T(self):
        """End time of the SDE."""
        pass
    @property
    @abc.abstractmethod
    def sampling_eps(self):
        """Starting time of the SDE."""
        pass

    @property
    def is_continuous(self):
        """continuous model by default"""
        return True

    @abc.abstractmethod
    def t2rho(self, vec_t):
        """transition kernel"""
        pass

    @abc.abstractmethod
    def rho2t(self, vec_rho):
        pass

    @abc.abstractmethod
    def x2v(self, x, t):
        pass

    @abc.abstractmethod
    def v2x(self, x, t):
        pass


class MultiStepSDE(abc.ABC):
    """
    SDE use multistep for sampling
    """
    @abc.abstractmethod
    def psi(self, v_t_start, v_t_end):
        """transition kernel"""
        pass

    @abc.abstractmethod
    def eps_integrand(self, vec_t):
        """transition kernel"""
        pass

def get_rev_ts(exp_sde, num_step, ts_order, ts_phase="t"):
    #assert isinstance(exp_sde, ExpSDE), "only support ExpSDE now"

    t0, t1 = exp_sde.sampling_eps, exp_sde.sampling_T
    t0, t1 = tf.cast(t0, tf.float32), tf.cast(t1, tf.float32)
    if not exp_sde.is_continuous:
        t0 = t0 + 1e-1
    if ts_phase=="t":
        rev_ts = tf.pow(
            tf.linspace(
                tf.pow(t1, 1.0 / ts_order),
                tf.pow(t0, 1.0 / ts_order),
                num_step + 1
            ),
            ts_order
        )
    elif ts_phase=="log":
        rho0, rho1 = exp_sde.t2rho(t0), exp_sde.t2rho(t1)
        rev_rhos = tf.exp(
            tf.linspace(
                tf.math.log(rho1),
                tf.math.log(rho0),
                num_step + 1
            )
        )
        rev_ts = exp_sde.rho2t(rev_rhos)
    elif ts_phase=="rho":
        # recommendation setting by https://arxiv.org/abs/2206.00364
        # rho0, rho1 = 0.002, 80
        rho0, rho1 = exp_sde.t2rho(t0), exp_sde.t2rho(t1)
        rev_rhos = tf.pow(
            tf.pow(rho1, 1.0 / ts_order) + \
                tf.linspace(0, num_step, num_step + 1) / num_step * \
                    (tf.pow(rho0, 1.0 / ts_order) - tf.pow(rho1, 1.0 / ts_order)),
            ts_order
        )
        rev_ts = exp_sde.rho2t(rev_rhos)
    else:
        method = "\n\t".join(["t", "log", "rho"])
        raise RuntimeError(f"only support ts_phase {method}")

    if not exp_sde.is_continuous:
        np_rev_ts = np.asarray(rev_ts, dtype=int)
        _, idx = np.unique(np_rev_ts, return_index=True)
        np_rev_ts = np_rev_ts[np.sort(idx)]

        remain_steps = num_step + 1 - np_rev_ts.shape[0]
        if remain_steps > 0:
            l = np.array([i for i in range(int(t1), int(t0), -1) if i not in np_rev_ts][-remain_steps:], dtype=int)
            np_rev_ts = np.concatenate([np_rev_ts, l], axis=0)
            np_ts = np.sort(np_rev_ts)
            rev_ts = tf.convert_to_tensor(np.flip(np_ts).copy())
        else:
            rev_ts = tf.convert_to_tensor(rev_ts, dtype=tf.int32)

    return rev_ts
