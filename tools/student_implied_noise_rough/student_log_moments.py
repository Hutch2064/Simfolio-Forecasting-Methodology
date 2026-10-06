"""Exact Student log-square moments from the squared-return Mellin transform."""
import math

from scipy.special import digamma, polygamma


def log_square_moments(inverse_df):
    u=float(inverse_df)
    base_bias=-float(digamma(.5))-math.log(2.)
    if u==0:return -base_bias,math.pi**2/2
    x=.5/u
    correction=digamma(x)-math.log(x)-math.log1p(-2*u)
    return -float(base_bias+correction),float(math.pi**2/2+polygamma(1,x))


