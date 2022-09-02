
import tensorflow as tf
import warnings
from sde import ExpSDE, MultiStepSDE




def quad_root(a, b, c):
    num = -b + tf.sqrt(b**2 - 4 * a * c) 
    return num / 2 / a

def get_linear_alpha_fns(beta_0, beta_1):
    def log_alpha_fn(t):
        log_mean_coef = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
        return 2 * log_mean_coef

    def t2alpha_fn(t):
        return tf.math.exp(log_alpha_fn(t))

    def alpha2t_fn(alpha):
        log_mean_coef_from_alpha = tf.math.log(alpha) / 2
        return quad_root(0.25 * (beta_1 - beta_0), 0.5 * beta_0, log_mean_coef_from_alpha)

    return t2alpha_fn, alpha2t_fn

def get_cos_alpha_fns():
    def t2alpha_fn(t):
        return tf.math.cos(
            ((t+0.008) / 1.008 * math.pi / 2)
        )**2
    def alpha2t_fn(alpha):
        return tf.math.acos(
            tf.sqrt(alpha)
        ) * 2 / math.pi * 1.008 - 0.008

    return t2alpha_fn, alpha2t_fn

class VPSDE(ExpSDE, MultiStepSDE):
    def __init__(self, t2alpha_fn, alpha2t_fn, sampling_eps, sampling_T):
        self._sampling_eps = sampling_eps
        self._sampling_T = sampling_T
        self.t2alpha_fn = t2alpha_fn
        self.alpha2t_fn = alpha2t_fn
        self.alpha_start = 1.0
        self.log_alpha_fn = lambda t: tf.math.log(self.t2alpha_fn(t))

    @property
    def sampling_T(self):
        return self._sampling_T

    @property
    def sampling_eps(self):
        return self._sampling_eps

    def psi(self, t_start, t_end):
        if len(t_start.shape)>len(t_end.shape):
            t_end = t_end[...,None]
       
        return tf.sqrt(self.t2alpha_fn(t_end) / self.t2alpha_fn(t_start))

    def eps_integrand(self, vec_t):
        with tf.GradientTape() as tape:
            tape.watch(vec_t)
            y = self.t2alpha_fn(vec_t)
        d_log_alpha_dtau = tape.gradient(y, vec_t)
        print('d_log_alpha_dtau', d_log_alpha_dtau)
        integrand = -0.5 * d_log_alpha_dtau / tf.sqrt(1 - self.t2alpha_fn(vec_t))
        print('eps_integ', integrand)
        return integrand

    def t2rho(self, t):
        alpha_t  = self.t2alpha_fn(t)
        return tf.sqrt(self.alpha_start / alpha_t * (1-alpha_t)) - tf.sqrt(1.0 - self.alpha_start)

    def rho2t(self, rho):
        num = self.alpha_start
        denum = (rho + tf.sqrt(1 - self.alpha_start))**2 + self.alpha_start
        cur_alpha = num / denum
        return self.alpha2t_fn(cur_alpha)

    def x2v(self, x, t):
        return tf.sqrt(self.alpha_start / self.t2alpha_fn(t)) * x

    def v2x(self, v, t):
        coef = tf.sqrt(self.alpha_start / self.t2alpha_fn(t))
        return v / coef

def get_interp_fn(xp_, fp):
  def _fn(x):
      if tf.shape(xp_) != tf.shape(fp) or tf.rank(xp_) != 1:
          raise ValueError("xp and fp must be one-dimensional arrays of equal size")
      
      x = tf.cast(x, tf.float32)
     
      
      if len(x.shape)>len(xp_.shape):
            xp = tf.tile(xp_[None,...], [x.shape[0], 1])
            i = tf.clip_by_value(tf.searchsorted(xp, x, side='right'), clip_value_min=1, clip_value_max=tf.size(xp) - 1)
            xp = xp[0]
      else:
        xp = xp_
        i = tf.clip_by_value(tf.searchsorted(xp, x, side='right'), clip_value_min=1, clip_value_max=tf.size(xp) - 1)
      df = tf.gather(fp,i) - tf.gather(fp, i - 1)
      dx = tf.gather(xp, i) - tf.gather(xp, i - 1)
      delta = x - tf.gather(xp, i - 1)
      f = tf.where((dx == 0), tf.gather(fp, i), tf.gather(fp, i - 1) + (delta / dx) * df)
      return f
  return _fn

class DiscreteVPSDE(VPSDE):
    def __init__(self, discrete_alpha):
        j_alphas = tf.cast(tf.convert_to_tensor(discrete_alpha), tf.float32)
        j_times = tf.range(len(discrete_alpha), dtype=tf.float32)

        # use a piecewise linear function to fit alpha
        _t2alpha_fn = get_interp_fn(j_times, j_alphas)
        _alpha2t_fn = get_interp_fn(2.0 - j_alphas, j_times)
        t2alpha_fn = lambda item: tf.clip_by_value(
            _t2alpha_fn(item), clip_value_min=1e-7, clip_value_max=1.0 - 1e-7
        )
        alpha2t_fn = lambda item: tf.clip_by_value(
            _alpha2t_fn(2.0 - item), clip_value_min=j_times[0], clip_value_max=j_times[-1]
        )
        #t2alpha_fn = lambda item:_t2alpha_fn(item)
        #alpha2t_fn = lambda item: _alpha2t_fn(2.0 - item)
        super().__init__(t2alpha_fn, alpha2t_fn, j_times[0], j_times[-1])
        warnings.warn(
            "\nWe are using a piecewise linear function to fit alpha and construct continuous time SDE\n" + \
            f"The continuous time SDE uses integer timestamps 0, 1, ... , {int(j_times[-1])} by default\n" + \
            "The default time scheduling uses continuous time that may be suboptimal for models trained with discrete time.\n" + \
            "Modify time scheduling in sampling algorithm and choose proper time discretization for your model if needed"
        )

    @property
    def is_continuous(self):
        return False
