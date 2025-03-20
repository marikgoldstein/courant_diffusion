import torch
import utils

# This file does two things

# 1) defines functions that convert any model output to all possible predictions

# 2) defines targets for all possible predictions

# For both 1) and 2), the possible types are: 
# x0 prediction E[x0|xt]
# score prediction E[-x1/sigma | xt]
# x1 prediction is E[x1 | xt] (i.e., epsilon pred)
# velocity prediction pred E[adot x0 + sdot x1 | xt]
# "v" prediction (not same as velocity) E[alpha * x1 - sigma * x0 | xt]

import utils

def nan(x):
    return torch.any(torch.isnan(x))

def inf(x):
    return torch.any(torch.isinf(x))

class Prediction:

    def __init__(self, x0, score, x1, velocity, v, delta):
        self.x0 = x0
        self.score = score
        self.x1 = x1
        self.velocity = velocity
        self.v = v
        self.f_fwd = velocity + utils.bcast_right(delta, x0.ndim) * score
        self.f_rev = velocity - utils.bcast_right(delta, x0.ndim) * score
        self.delta = delta


def get_model_out_to_pred_obj_fn(model_type):

    prediction_type = model_type

    def x0_to_score(t, xt, x0, coefs):
        sigma_squared = utils.bcast_right(coefs.sigma**2, xt.ndim)
        alpha = utils.bcast_right(coefs.alpha, xt.ndim)
        return (alpha * x0 - xt) / sigma_squared

    def score_to_x0(t, xt, score, coefs):
        sigma2 = utils.bcast_right(coefs.sigma**2, xt.ndim)
        alpha = utils.bcast_right(coefs.alpha, xt.ndim)
        return (sigma2 * score + xt) / alpha

    def x0_to_x1(t, xt, x0, coefs):
        sigma = utils.bcast_right(coefs.sigma, xt.ndim)
        alpha = utils.bcast_right(coefs.alpha, xt.ndim)
        return (xt - alpha * x0) / sigma

    def x1_to_x0(t, xt, x1, coefs):
        sigma = utils.bcast_right(coefs.sigma, xt.ndim)
        alpha = utils.bcast_right(coefs.alpha, xt.ndim)
        return (xt - sigma * x1) / alpha

    def x1_x0_to_velocity(
        t, xt, x1, x0, coefs
    ):
        sigma_dot = utils.bcast_right(coefs.sigma_dot, xt.ndim)
        alpha_dot = utils.bcast_right(coefs.alpha_dot, xt.ndim)
        return alpha_dot * x0 + sigma_dot * x1

    def velocity_to_x0(t, xt, velocity, coefs):
        sigma = utils.bcast_right(coefs.sigma, xt.ndim)
        alpha = utils.bcast_right(coefs.alpha, xt.ndim)
        sigma_dot = utils.bcast_right(coefs.sigma_dot, xt.ndim)
        alpha_dot = utils.bcast_right(coefs.alpha_dot, xt.ndim)
        numerator = velocity - (sigma_dot / sigma) * xt
        denominator = alpha_dot - alpha * sigma_dot / sigma
        denominator += 1e-8
        return numerator / denominator

    def x1_x0_to_v(t, xt, x1, x0, coefs):
        sigma = utils.bcast_right(coefs.sigma, xt.ndim)
        alpha = utils.bcast_right(coefs.alpha, xt.ndim)
        return alpha * x1 - sigma * x0

    def v_to_x0(t, xt, v, coefs):
        sigma = utils.bcast_right(coefs.sigma, xt.ndim)
        alpha = utils.bcast_right(coefs.alpha, xt.ndim)
        numerator = alpha * xt - sigma * v
        denominator = alpha ** 2 + sigma ** 2
        return numerator / denominator

    def x0_to_prediction(t, xt, model_out, coefs):
        x0 = model_out
        score = x0_to_score(t=t,xt=xt,x0=x0,coefs=coefs)
        x1 = x0_to_x1(t=t,xt=xt,x0=x0,coefs=coefs)
        velocity = x1_x0_to_velocity(t=t,xt=xt, x1=x1, x0=x0,coefs=coefs)
        v = x1_x0_to_v(t=t, xt=xt, x1=x1, x0=x0,coefs=coefs)
        return Prediction(
            x0=x0, score=score, x1=x1, velocity=velocity, v=v, delta=coefs.delta,
        )

    def score_to_prediction(t, xt, model_out, coefs):
        score = model_out
        x0 = score_to_x0(t=t, xt=xt, score=score,coefs=coefs)
        x1 = x0_to_x1(t=t, xt=xt, x0=x0,coefs=coefs)
        velocity = x1_x0_to_velocity(t=t, xt=xt, x1=x1, x0=x0,coefs=coefs)
        v = x1_x0_to_v(t=t, xt=xt, x1=x1, x0=x0,coefs=coefs)
        return Prediction(
            x0=x0,score=score,x1=x1, velocity=velocity, v=v, delta=coefs.delta,
        )
    
    def x1_to_prediction(t, xt, model_out, coefs):
        x1 = model_out 
        x0 = x1_to_x0(t=t, xt=xt, x1=x1,coefs=coefs)
        score = x0_to_score(t=t, xt=xt, x0=x0,coefs=coefs)
        velocity = x1_x0_to_velocity(t=t, xt=xt, x1=x1, x0=x0,coefs=coefs)
        v = x1_x0_to_v(t=t, xt=xt, x1=x1, x0=x0,coefs=coefs)
        return Prediction(
            x0=x0, score=score, x1=x1, velocity=velocity, v=v, delta=coefs.delta,
        )

    def velocity_to_prediction(
        t, xt, model_out, coefs,
    ):
        velocity = model_out 
        x0 = velocity_to_x0(t=t, xt=xt, velocity=velocity,coefs=coefs)
        score = x0_to_score(t=t, xt=xt, x0=x0,coefs=coefs)
        x1 = x0_to_x1(t=t, xt=xt, x0=x0,coefs=coefs)
        v = x1_x0_to_v(t=t, xt=xt, x1=x1, x0=x0,coefs=coefs)
        return Prediction(
            x0=x0, score=score, x1=x1, velocity=velocity, v=v, delta=coefs.delta,
        )

    def v_to_prediction(
        t, xt, model_out, coefs,
    ):
        v = model_out
        x0 = v_to_x0(t=t,xt=xt,v=v,coefs=coefs)
        score = x0_too_scoe(t=t, xt=xt, x0=x0,coefs=coefs)
        x1 = x0_to_x1(t=t,xt=xt, x0=x0,coefs=coefs)
        velocity = x1_x0_to_velocity(t=t, xt=xt, x1=x1, x0=x0,coefs=coefs)
        return Prediction(
            x0=x0,score=score, x1=x1, velocity=velocity, v=v, delta=coefs.delta,
        )

    if prediction_type == 'x0':
        return x0_to_prediction
    elif prediction_type == 'score':
        return score_to_prediction
    elif prediction_type == 'x1':
        return x1_to_prediction
    elif prediction_type == 'velocity':
        return velocity_to_prediction
    elif prediction_type == 'v':
        return v_to_prediction
    else:
        raise ValueError(f"Unknown Prediction Type: {prediction_type}")


def get_target_fn():

    def x1_to_score(t, xt, x1, coefs):
        sigma = utils.bcast_right(coefs.sigma, xt.ndim)
        return -x1 / sigma
    
    def x1_x0_to_velocity(t, xt, x0, x1, coefs):
        sigma_dot = utils.bcast_right(coefs.sigma_dot, xt.ndim)
        alpha_dot = utils.bcast_right(coefs.alpha_dot, xt.ndim)
        return alpha_dot * x0 + sigma_dot * x1

    def x1_x0_to_v(t, xt, x1, x0, coefs):
        sigma = utils.bcast_right(coefs.sigma, xt.ndim)
        alpha = utils.bcast_right(coefs.alpha, xt.ndim)
        return alpha * x1 - sigma * x0

    def target_to_prediction(t, xt, x0, x1, coefs):
        score = x1_to_score(t=t,xt=xt, x1=x1,coefs=coefs)
        velocity = x1_x0_to_velocity(t=t,xt=xt, x1=x1, x0=x0,coefs=coefs)
        v = x1_x0_to_v(t=t, xt=xt, x1=x1, x0=x0,coefs=coefs)
        return Prediction(
            x0=x0, score=score, x1=x1, velocity=velocity, v=v, delta=coefs.delta,
        )
    
    return target_to_prediction

