import numpy as np
import scipy.stats as stats 
from typing import List, Tuple

# This script lists a bunch of a/b testing functions

def z_test(a: List[float], b: List[float], alpha: float = .05) -> Tuple[float]:
    """
    Manual Z-Test
    Calculates z-statistic using pooled variance

    Parameters
    ----------
    a : {float, array_like}
    b : {float, array_like}
    alpha : float

    Returns
    ------- 
    lower : float
    mean : float
    upper : float
    sd : float  
    z_statistic : float
    p_value : float
    """
    ma = np.mean(a)
    mb = np.mean(b)
    na = len(a)
    nb = len(b)
    var_a = np.var(a)
    var_b = np.var(b)
 
    z_sd = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)) * np.sqrt(1 / na + 1 / nb)
    z_ci = stats.norm.ppf(1 - alpha / 2) * z_sd # norm.ppf eq qnorm in R
    z_statistic = (mb - ma) / z_sd

    lower = (mb - ma) - z_ci
    mean = (mb - ma)
    upper = (mb - ma) + z_ci
    sd = z_sd
    p_value = 2 * stats.norm.cdf(-abs(z_statistic)) # norm.cdf eq pnorm in R
     
    return lower, mean, upper, sd, p_value

# Auto Z-Test (sample N > 150)
# from statsmodels.stats.proportion import proportions_ztest
# proportions_ztest(count, nobs, value=0.05, alternative='two-sided', prop_var=False)


def t_test(
           a: List[float],
           b: List[float],
           alpha: float = .05,
           pct: bool = False
    ) -> Tuple[float]:
    
    """
    Manual Welch's T-Test
    This assumes unequal variance & unequal/equal sample size
    
    Parameters
    ----------
    a : {float, array_like}
    b : {float, array_like}
    alpha : float
    pct : bool

    Returns
    -------
    lower_diff : float
    mean_diff : float
    upper_diff : float
    sd_diff : float
    p_val : float
    """

    ma = np.mean(a)
    mb = np.mean(b)
    na = len(a)
    nb = len(b)
    var_a = np.var(a)
    var_b = np.var(b)

    se_a = var_a / na
    se_b = var_b / nb
   
    t_sd = np.sqrt(se_a + se_b)
    t_statistic = (mb - ma) / t_sd
   
    df_t = ((se_a + se_b) ** 2) / ((se_a ** 2) + (se_b ** 2) / (na + nb - 2))
    ci = stats.t.ppf(1 - alpha / 2, df=df_t) * t_sd
    
    ma_or_one = ma * pct or 1 # ma if pct=True else 1
    lower_diff = mb - ma - ci / ma_or_one 
    mean_diff = mb - ma / ma_or_one
    upper_diff = mb - ma + ci / ma_or_one
    sd_diff = t_sd / ma_or_one
    p_val = 2 * stats.t.cdf(-abs(t_statistic), df=df_t)
    
    return lower_diff, mean_diff, upper_diff, sd_diff, p_val
