from scipy.stats import norm
from scipy.integrate import quad, dblquad
import math
from scipy import optimize
import numpy as np
# from matplotlib import pyplot as plt

'''Parameters of the model'''

# The main parameters
mean_i = 4  # mean of the general ability of the current employee i
mean_j = 4  # mean of the general ability of the job candidates j,m
st_dev = 1  # standard deviation of the general ability of i, j, m
rho = 0.5  # correlation coefficient between general, specific abilities and outputs of referred and referring workers
psi = 0.5  # social preference parameter showing the strength of the tie between referred and referring workers

st_dev_y = (st_dev ** 2 + 1) ** 0.5  # standard deviation of the workers' output y = sqrt(var_general _ability +
# var_specific_ability).

# Parameters of the ERP:
# If parameter p_b == 0 bonus is paid when the candidate j is referred.
# If parameter p_b == 1 bonus is paid only when the candidate j stayed for t_j = 2.
p_b = 0

'''Costs of referral for the current employee i referring j for the position with mean mj'''

# Parameters for referral cost function'
a_c = 0.01
b_c = 0.01
c_c = 0.01


# Function of the referral costs
def c(m_j):
    return a_c * m_j ** 2 + b_c * m_j + c_c
    # continuous twice differentiable increasing and convex function of average general ability of the job candidates


'''Technical functions'''


# PDF of bi-variate normal distribution with mean = (m1,m2) variance = ((sd1^2, r*sd1*sd2),(r*sd1*sd2, sd2^2))
def pdf_ij(x1, x2, m1, m2, sd1, sd2, r):
    prob_dens_func = (1 / (2 * math.pi * sd1 * sd2 * ((1 - r ** 2) ** 0.5))) * math.exp(-1 / (2 * (1 - r ** 2)) * (
            ((x1 - m1) / sd1) ** 2 + ((x2 - m2) / sd2) ** 2 - 2 * r * (x1 - m1) * (x2 - m2) / (sd1 * sd2)))
    return prob_dens_func


# Conditional expectation of x2 given x1 = x, where x1, x2 are normally distributed and correlated
def exp_x2_x1(x, m1, m2, sd1, sd2, r):
    conditional_expectation = m2 + r * sd2 * (x - m1) / sd1  # E[x2|x1 = x] = E[x2]+r*sd2*(x-E[x1])/sd1
    return conditional_expectation


# Conditional variance of x2 given x1 =x, where x1, x2 are normally distributed and correlated
def var_x2_x1(sd2, r):
    conditional_variance = (1 - r ** 2) * (sd2 ** 2)  # Var[x2|x1=x] = (1-r**2)*(1+sd_x2**2)
    return conditional_variance


# Inverse Mills ratio'
def imr(x, m, sd):
    mills = norm.pdf((x - m) / sd) / (1 - norm.cdf((x - m) / sd))  # IMR(x) = f(x)/(1-F(x)) if x is st norm dist var.
    return mills


# Total expected probability given that x1 in (a1,b1), x2 in (a2,b2)'''
def prob_trunc_ij(m1, m2, sd1, sd2, r, a1, b1, a2, b2):
    f = lambda y, x: pdf_ij(x, y, m1, m2, sd1, sd2, r)
    probability = dblquad(f, a1, b1, a2, b2)
    return probability[0]


# Expected value of y given that y in (a2,b2), x in (a1, b1)'''
def exp_trunc_ij(m1, m2, sd1, sd2, r, a1, b1, a2, b2):
    f1 = lambda y, x: x * y * pdf_ij(x, y, m1, m2, sd1, sd2, r)
    expected_truncated_value = dblquad(f1, a1, b1, a2, b2)
    return expected_truncated_value[0] / prob_trunc_ij(m1, m2, sd1, sd2, r, a1, b1, a2, b2)


'''Main functions of the model'''
'Probabilities'


# Probability that the labor market candidate m will stay in the firm in t_m=2
def p_m_2(m_m, sd):
    sd_y = (sd ** 2 + 1) ** 0.5  # Standard deviation of y_m
    return 1 - norm.cdf((m_m - m_m) / sd_y)  # P(y_m >= m_m)


# Probability that the candidate j referred by current employee i with the output y_i will stay in t_j=2
def p_j_2(y_i, m_i, m_j, sd, r):
    sd_y = (sd ** 2 + 1) ** 0.5  # Standard deviation of y_m
    return 1 - norm.cdf(
        (m_j - exp_x2_x1(y_i, m_i, m_j, sd_y, sd_y, r)) / (var_x2_x1(sd_y, r)) ** 0.5)  # P(y_j >= m_j | y_i)


# Probability that the bonus b will be paid under ERP to the current employee i with y_i if she referred candidate j
def p_b_paid(y_i, m_i, m_j, sd, r):
    prob_b_paid = None
    if p_b == 0:
        prob_b_paid = 1
    if p_b == 1:
        prob_b_paid = p_j_2(y_i, m_i, m_j, sd, r)
    return prob_b_paid


'Expected outputs of the workers'


# Expected output of the labor market candidate m
# In the period t_m = 1
def e_y_m_1(m_m):
    return m_m


# In the period t_m = 2 given that m stayed in the firm
def e_y_m_2(m_m, sd):
    sd_y = (sd ** 2 + 1) ** 0.5  # Standard deviation of y_m
    return m_m + sd_y * imr(m_m, m_m, sd_y)  # Conditional expected mean: E[y_m | y_m >= m_m]


# Total expected output of the m in two periods
def e_y_m_t(m_m, sd):
    sd_y = (sd ** 2 + 1) ** 0.5  # Standard deviation of y_m
    return e_y_m_1(m_m) + p_m_2(m_m, sd_y) * e_y_m_2(m_m, sd_y)


# ----------------------------------------------------------------------------
# Expected output of the candidate j referred by the current worker i with y_i
# In the period t_j = 1
def e_y_j_1(y_i, m_i, m_j, sd, r):
    sd_y = (sd ** 2 + 1) ** 0.5  # Standard deviation of y_m
    return exp_x2_x1(y_i, m_i, m_j, sd_y, sd_y, r)  # E[y_j|y_i]


# In the period t_j = 2, given that j stayed for t_j =2
def e_y_j_2(y_i, m_i, m_j, sd, r):
    sd_y = (sd ** 2 + 1) ** 0.5  # Standard deviation of y_m
    return exp_x2_x1(y_i, m_i, m_j, sd_y, sd_y, r) + (var_x2_x1(sd_y, r)) ** 0.5 * imr(
        m_j,
        exp_x2_x1(y_i, m_i, m_j, sd_y, sd_y, r),
        (var_x2_x1(sd_y, r)) ** 0.5
    )  # E[{y_j|y_i} | {y_j| y_i} >= m_j]


# Total expected output of the candidate j referred by the current worker i with y_i
def e_y_j_t(y_i, m_i, m_j, sd, r):
    sd_y = (sd ** 2 + 1) ** 0.5  # Standard deviation of y_m
    return e_y_j_1(y_i, m_i, m_j, sd_y, r) + p_j_2(y_i, m_i, m_j, sd_y, r) * e_y_j_2(y_i, m_i, m_j, sd_y, r)


'Wages of the workers'


# Wage of the labor market candidate m
# In the period t_m = 1
def w_m_1(m_m):
    return m_m  # E[s_m] where s_m = is the general ability of m


# In the period t_m = 2, given his output in t_m = 1 equal to y_m
def w_m_2(y_m, m_m, sd):
    sd_y = (sd ** 2 + 1) ** 0.5  # Standard deviation of y_m
    r_s_y_m = sd / sd_y  # Corr(s_m, y_m)
    return exp_x2_x1(y_m, m_m, m_m, sd_y, sd, r_s_y_m)  # E[s_m|y_m]


# Expected wage in t_m = 2, given that m stayed in the firm
def e_w_m_2(m_m, sd):
    return w_m_2(
        e_y_m_2(m_m, sd),
        m_m,
        sd
    )  # E[s_m | y_m >= m_m]


# exp_trunc_ij(m_m, m_m, sd_y, sd, r_s_y_m, m_m, np.inf, -np.inf, np.inf)

# ----------------------------------------------------------------------------
# Wage of the candidate j referred by the current employee i with y_i
# In the period t_j = 1, given y_i
def w_j_1(y_i, m_i, m_j, sd, r):
    sd_y = (sd ** 2 + 1) ** 0.5  # Standard deviation of y_j
    r_s_y_ij = r * sd / sd_y  # Corr(s_j, y_i)
    return exp_x2_x1(y_i, m_i, m_j, sd_y, sd, r_s_y_ij)  # E[s_j | y_i] where s_j = is the general ability of j


# In the period t_j = 2, given y_j
def w_j_2(y_j, m_j, sd):
    sd_y = (sd ** 2 + 1) ** 0.5  # Standard deviation of y_m
    r_s_y = sd ** 2 / (sd * sd_y)  # Corr(s_j, y_j)
    return exp_x2_x1(y_j, m_j, m_j, sd_y, sd, r_s_y)  # E[s_j|y_j]


# In the period t_j = 2, given {y_j | y_i} >= m_j
def e_w_j_2_upper(y_i, m_i, m_j, sd, r):
    return w_j_2(
        e_y_j_2(y_i, m_i, m_j, sd, r),
        m_j,
        sd
    )  # E[s_j|{y_j|y_i}>=m_j]


'Profits of the firm'


# (Expected) profit of the firm from employing the labor market candidate m
# In the period t_m = 1
def pi_m_1(m_m):
    return e_y_m_1(m_m) - w_m_1(m_m)  # difference between the output and the wage


# In the period t_m = 2, given that m stayed in the firm
def pi_m_2(m_m, sd):
    return e_y_m_2(m_m, sd) - e_w_m_2(m_m, sd)


# Overall expected profit of the firm from the candidate m
def pi_m_t(m_m, sd):
    return pi_m_1(m_m) + p_m_2(m_m, sd) * pi_m_2(m_m, sd)


# ----------------------------------------------------------------------------
# Expected profit of the firm from employing the candidate j referred by the current worker i with y_i
# In the period t_j = 1
def pi_j_1(y_i, m_i, m_j, sd, r):
    return e_y_j_1(y_i, m_i, m_j, sd, r) - w_j_1(y_i, m_i, m_j, sd, r)


# In the period t_j = 2, given that j stayed in the firm
def pi_j_2(y_i, m_i, m_j, sd, r):
    return e_y_j_2(y_i, m_i, m_j, sd, r) - e_w_j_2_upper(y_i, m_i, m_j, sd, r)


# Overall expected profit of the firm from the candidate j referred by the current worker i with y_i
def pi_j_t(y_i, m_i, m_j, sd, r):
    return pi_j_1(y_i, m_i, m_j, sd, r) + p_j_2(y_i, m_i, m_j, sd, r) * pi_j_2(y_i, m_i, m_j, sd, r) + (
            1 - p_j_2(y_i, m_i, m_j, sd, r)
    ) * pi_m_1(m_j)


# Difference between the overall expected profit of the candidate j referred by i with y_i and the market candidate m
def delta_pi(y_i, m_i, m_j, sd, r):
    return pi_j_t(y_i, m_i, m_j, sd, r) - pi_m_t(m_j, sd)


# ----------------------------------------------------------------------------
# Expected profit of the firm from employing the candidate j referred by the current worker i with y_i under ERP
# In the period t_j = 1
def pi_j_1_b(y_i, m_i, m_j, sd, r, b):
    return e_y_j_1(y_i, m_i, m_j, sd, r) - w_j_1(y_i, m_i, m_j, sd, r) - b * p_b_paid(y_i, m_i, m_j, sd, r)


# Overall expected profit of the firm from the candidate j referred by the current worker i with y_i under ERP
def pi_j_t_b(y_i, m_i, m_j, sd, r, b):
    return pi_j_1_b(y_i, m_i, m_j, sd, r, b) + p_j_2(y_i, m_i, m_j, sd, r) * pi_j_2(y_i, m_i, m_j, sd, r) + (
            1 - p_j_2(y_i, m_i, m_j, sd, r)
    ) * pi_m_1(m_j)


# Difference between the overall expected profit of j referred by i with y_i and the market candidate m under ERP
def delta_pi_b(y_i, m_i, m_j, sd, r, b):
    return pi_j_t_b(y_i, m_i, m_j, sd, r, b) - pi_m_t(m_j, sd)


'''Calculating thresholds, optimal bonus and overall expected profit of the firm unconditional on y_i'''


# Threshold of the current employee i, under which she never refers: y_tilde
def y_tilde(m_i, m_j, sd, r, p):
    sd_y = (sd ** 2 + 1) ** 0.5  # Standard deviation of y_m
    return m_i + (c(m_j) * sd_y ** 2) / (r * p * sd ** 2)


# Threshold of the firm, under which it never hires candidate j referred by i with y_i: y_star
def y_star(m_i, m_j, sd, r):
    return optimize.fsolve(delta_pi, x0=m_i, args=(m_i, m_j, sd, r))[0]


# Threshold of the current employee i, under which she never refers in case of ERP with bonus b: y_tilde_b
def utility_i_b(y_i, m_i, m_j, sd, r, p, b):
    return p * (w_j_1(y_i, m_i, m_j, sd, r) - w_m_1(m_j)) - c(m_j) + b * p_b_paid(y_i, m_i, m_j, sd, r)


# Threshold of the current employee i, under which she never refers in case of ERP with bonus b: y_tilde_b
def y_tilde_b(m_i, m_j, sd, r, p, b):
    return optimize.fsolve(utility_i_b, x0=m_i, args=(m_i, m_j, sd, r, p, b))[0]


# Threshold of the firm, under which it never hires candidate j referred by i with y_i under ERP with bonus b: y_star_b
def y_star_b(m_i, m_j, sd, r, b):
    return optimize.fsolve(delta_pi_b, x0=m_i, args=(m_i, m_j, sd, r, b))[0]


'''Finding the optimal bonus'''


# Optimal bonus is defined as the bonus, under which the firm maximizes its overall expected profit unconditional on y_i
# The formula of the total firm's profit: total profit = Pr(referral)*profit_from_referral + Pr(no ref)* profit_from_m
# It should be calculated in several steps:
# 1.1) Find the probability of referral Pr(ref) under no ERP:
def p_ref(m_i, m_j, sd, r, p):
    sd_y = (sd ** 2 + 1) ** 0.5  # Standard deviation of y_m
    y_thr = max(y_tilde(m_i, m_j, sd, r, p), y_star(m_i, m_j, sd, r))
    return 1 - norm.cdf((y_thr - m_i) / sd_y)


# 1.2) Find the probability of referral Pr_b(ref) under ERP with bonus b:
def p_ref_b(m_i, m_j, sd, r, p, b):
    sd_y = (sd ** 2 + 1) ** 0.5  # Standard deviation of y_m
    if y_tilde(m_i, m_j, sd, r, p) < y_star(m_i, m_j, sd, r):
        prob_ref_erp = p_ref(m_i, m_j, sd, r, p)
    else:
        prob_ref_erp = 1 - norm.cdf((y_tilde_b(m_i, m_j, sd, r, p, b) - m_i) / sd_y)
    return prob_ref_erp


# 2.1) Find the profit from referral under voluntary referrals:
def pi_vr_j(m_i, m_j, sd, r, p):
    sd_y = (sd ** 2 + 1) ** 0.5  # Standard deviation of y_m
    integrand_vr = lambda x: pi_j_t(x, m_i, m_j, sd, r) * norm.pdf((x - m_i) / sd_y) / sd_y
    profit_vr_j = quad(
        integrand_vr,
        max(y_tilde(m_i, m_j, sd, r, p), y_star(m_i, m_j, sd, r)),
        np.infty
    )
    return profit_vr_j[0] / p_ref(m_i, m_j, sd, r, p)


# 2.2) Find the profit from referral under ERP with bonus b:
def pi_erp_j(m_i, m_j, sd, r, p, b):
    sd_y = (sd ** 2 + 1) ** 0.5  # Standard deviation of y_m
    y_thr = max(y_tilde_b(m_i, m_j, sd, r, p, b), y_star_b(m_i, m_j, sd, r, b))
    integrand_erp = lambda x: pi_j_t_b(x, m_i, m_j, sd, r, b) * norm.pdf((x - m_i) / sd_y) / sd_y
    profit_erp_j = quad(
        integrand_erp,
        y_thr,
        np.infty
    )
    return profit_erp_j[0] / p_ref_b(m_i, m_j, sd, r, p, b)


# 3.1) Find the overall profit under voluntary referrals:
def pi_total_vr(m_i, m_j, sd, r, p):
    return p_ref(m_i, m_j, sd, r, p) * pi_vr_j(m_i, m_j, sd, r, p) + (1 - p_ref(m_i, m_j, sd, r, p)) * pi_m_t(m_j, sd)


# 3.1) Find the overall profit under ERP with bonus b:
def pi_total_erp(m_i, m_j, sd, r, p, b):
    return p_ref_b(m_i, m_j, sd, r, p, b) * pi_erp_j(m_i, m_j, sd, r, p, b) + \
           (1 - p_ref_b(m_i, m_j, sd, r, p, b)) * pi_m_t(m_j, sd)


# 4) Find the optimal bonus b_star:
def neg_pi_total_erp(b, m_i, m_j, sd, r, p):
    return - pi_total_erp(m_i, m_j, sd, r, p, b)  # Change the sign of the total expected profit to use minimization


def b_star(m_i, m_j, sd, r, p):
    if y_tilde(m_i, m_j, sd, r, p) < y_star(m_i, m_j, sd, r):
        bonus_star = [0]
    else:
        bonus_star = optimize.fmin(neg_pi_total_erp, x0=c(m_j), args=(m_i, m_j, sd, r, p))
    return max(0, bonus_star[0])


# Testing
yi = mean_i + 0.5
yj = yi
ym = yj
# bonus = c(mean_j)/2
B_star = b_star(mean_i, mean_j, st_dev, rho, psi)

print('p_m_2 = ', p_m_2(mean_j, st_dev))
print('p_j_2 = ', p_j_2(yi, mean_i, mean_j, st_dev, rho))
print('e_y_m_1 = ', e_y_m_1(mean_j))
print('e_y_m_2 = ', e_y_m_2(mean_j, st_dev))
print('e_y_m_t = ', e_y_m_t(mean_j, st_dev))
print('e_y_j_1 = ', e_y_j_1(yi, mean_i, mean_j, st_dev, rho))
print('e_y_j_2 = ', e_y_j_2(yi, mean_i, mean_j, st_dev, rho))
print('e_y_j_t = ', e_y_j_t(yi, mean_i, mean_j, st_dev, rho))
print('w_m_1 = ', w_m_1(mean_j))
print('w_m_2 = ', w_m_2(ym, mean_j, st_dev))
print('e_w_m_2 = ', e_w_m_2(mean_j, st_dev))
print('w_j_1 = ', w_j_1(yi, mean_i, mean_j, st_dev, rho))
print('w_j_2 = ', w_j_2(yj, mean_j, st_dev))
print('e_w_j_2_upper = ', e_w_j_2_upper(yi, mean_i, mean_j, st_dev, rho))
print('pi_m_1 = ', pi_m_1(mean_j))
print('pi_m_2 = ', pi_m_2(mean_j, st_dev))
print('pi_m_t = ', pi_m_t(mean_j, st_dev))
print('pi_j_1 = ', pi_j_1(yi, mean_i, mean_j, st_dev, rho))
print('pi_j_2 = ', pi_j_2(yi, mean_i, mean_j, st_dev, rho))
print('pi_j_t = ', pi_j_t(yi, mean_i, mean_j, st_dev, rho))
print('pi_j_1_b = ', pi_j_1_b(yi, mean_i, mean_j, st_dev, rho, B_star))
print('pi_j_t_b = ', pi_j_t_b(yi, mean_i, mean_j, st_dev, rho, B_star))
print('y_tilde = ', y_tilde(mean_i, mean_j, st_dev, rho, psi))
print('y_star = ', y_star(mean_i, mean_j, st_dev, rho))
print('y_tilde_b = ', y_tilde_b(mean_i, mean_j, st_dev, rho, psi, B_star))
print('y_star_b = ', y_star_b(mean_i, mean_j, st_dev, rho, B_star))
print('p_ref = ', p_ref(mean_i, mean_j, st_dev, rho, psi))
print('p_ref_b = ', p_ref_b(mean_i, mean_j, st_dev, rho, psi, B_star))
print('pi_vr_j = ', pi_vr_j(mean_i, mean_j, st_dev, rho, psi))
print('pi_erp_j = ', pi_erp_j(mean_i, mean_j, st_dev, rho, psi, B_star))
print('pi_total_vr = ', pi_total_vr(mean_i, mean_j, st_dev, rho, psi))
print('pi_total_erp = ', pi_total_erp(mean_i, mean_j, st_dev, rho, psi, B_star))
print('pi_total_erp_optimal = ', pi_total_erp(mean_i, mean_j, st_dev, rho, psi, B_star))
print('b_star = ', B_star)

# bonus = np.linspace(0, c(mean_j), 20)
# plt.plot(bonus, pi_total_erp(mean_i, mean_j, st_dev, rho, psi, bonus) )
# # plt.plot(bonus, p_ref_b(mean_i, mean_j, st_dev, rho, psi, bonus))
# plt.show()
