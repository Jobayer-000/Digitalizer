import tensorflow as tf

from sde import MultiStepSDE

def get_integrator_basis_fn(sde):
    def _worker(t_start, t_end, num_item):
        dt = (t_end - t_start) / tf.cast(num_item,tf.float32)

        t_inter = tf.linspace(t_start, t_end, num_item)
        psi_coef = sde.psi(t_inter, t_end)
        integrand = sde.eps_integrand(t_inter)

        return psi_coef * integrand, t_inter, dt
    return _worker


def single_poly_coef(t_val, ts_poly, coef_idx):
    """
    \prod_{k \neq j} \frac{\tau - t_{i+k}}{t_{i+j}-t_{i+k}}
    t_val: tau
    ts_poly: t_{i+k}
    j: coef_idx
    """
    num = t_val - ts_poly
    print(ts_poly)
    denum = tf.gather(ts_poly, coef_idx) - ts_poly
    print(coef_idx)
    num_ = tf.scatter_nd(coef_idx[:,None], tf.ones_like(coef_idx), num.shape)
    num = tf.where(num_==1, num_, num)
    denum_ = tf.scatter_nd(coef_idx[:, None], tf.ones_like(coef_idx), denum.shape)
    denum = tf.where(denum_==1, denum_, denum)
    return tf.reduce_prod(num) / tf.reduce_prod(denum)



def get_one_coef_per_step_fn(sde):
    _eps_coef_worker_fn = get_integrator_basis_fn(sde)
    def _worker(t_start, t_end, ts_poly, coef_idx, num_item):
        """
        C_{ij}
        j: coef_idx
        """
        print(0)
        integrand, t_inter, dt = _eps_coef_worker_fn(t_start, t_end, num_item)
        print(1)
        print(len(t_inter), ts_poly)
        poly_coef = single_poly_coef(t_inter, ts_poly, coef_idx)
        print(3)
        return tf.reduce_sum(integrand * poly_coef) * dt
    return _worker

def get_coef_per_step_fn(sde, highest_order, order):
    eps_coef_fn = get_one_coef_per_step_fn(sde)
    def _worker(t_start, t_end, ts_poly, num_item = 10000):
        """
        C_i
        #!: we do flip of j here!
        """
        rtn = tf.zeros((highest_order+1, ), dtype=float)
        ts_poly = ts_poly[:order+1]
        
        coef = eps_coef_fn(t_start, t_end, ts_poly, tf.range(order+1)[::-1], num_item)
        rtn = tf.concat([tf.ones_like(rtn[:order+1])*coef, rtn[order+1:]],axis=0)
        return rtn
    return _worker

def get_ab_eps_coef_order0(sde, highest_order, timesteps):
    _worker = get_coef_per_step_fn(sde, highest_order, 0)
    col_idx = tf.range(len(timesteps)-1)[:,None]
    idx = col_idx + tf.range(1)[None, :]
    vec_ts_poly = tf.gather(timesteps, idx)
    
    return _worker(timesteps[:-1], timesteps[1:], vec_ts_poly)

def get_ab_eps_coef(sde, highest_order, timesteps, order):
    assert isinstance(sde, MultiStepSDE)
    if order == 0:
        return get_ab_eps_coef_order0(sde, highest_order, timesteps)
    
    prev_coef = get_ab_eps_coef(sde, highest_order, timesteps[:order+1], order=order-1)

    cur_coef_worker = get_coef_per_step_fn(sde, highest_order, order)

    col_idx = tf.range(len(timesteps)-order-1)[:,None]
    idx = col_idx + tf.range(order+1)[None, :]
    vec_ts_poly = timesteps[idx]
    
    
    cur_coef = cur_coef_worker(timesteps[order:-1], timesteps[order+1:], vec_ts_poly) #[3, 4, (0,1,2,3)]

    return tf.concat(
        [
            prev_coef,
            cur_coef
        ],
        axis=0
    )

def ab_step(x, ei_coef, new_eps, eps_pred):
    x_coef, eps_coef = ei_coef[0], ei_coef[1:]
    full_eps = tf.concat([new_eps[None], eps_pred])
    eps_term = tf.einsum("i,i...->...", eps_coef, full_eps)
    return x_coef * x + eps_term, full_eps[:-1]
