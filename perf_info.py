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
def p_m_2(m_mu, sd_mu):
    return 1 - norm.cdf((m_mu - m_mu) / sd_mu)  # P(mu_m >= m_mu_m)


# Probability that the candidate j referred by current employee i with abilities s_i, m_i will stay in t_j=2
def p_j_2(mu_i, r):
    return 1 - norm.cdf(
        (0 - exp_x2_x1(mu_i, 0, 0, 1, 1, r)) / var_x2_x1(1, r) ** 0.5
    )  # P({mu_j | mu_i} >= 0)


# Probability that the bonus b will be paid under ERP to the current employee i with y_i if she referred candidate j
def p_b_paid(mu_i, r):
    prob_b_paid = None
    if p_b == 0:
        prob_b_paid = 1
    if p_b == 1:
        prob_b_paid = p_j_2(mu_i, r)
    return prob_b_paid


'Expected outputs of the workers'


# Expected output of the labor market candidate m
# In the period t_m = 1
def e_y_m_1(m_s_m):
    return m_s_m   # m_s_m stays for mean of the s_m, i.e. mean of the general ability of the candidate m


# In the period t_m = 2 given that m stayed in the firm
def e_y_m_2(m_s_m):     # m_mu - mean of the specific ability of the candidate m, sd_mu - standard deviation.
    return m_s_m + imr(0, 0, 1)  # Conditional expected mean: E[y_m | mu_m >= 0]


# Total expected output of the m in two periods
def e_y_m_t(m_s_m):
    return e_y_m_1(m_s_m) + p_m_2(0, 1) * e_y_m_2(m_s_m)


# ----------------------------------------------------------------------------
# Expected output of the candidate j referred by the current worker i with s_i, mu_i
# In the period t_j = 1
def e_y_j_1(s_i, mu_i, m_s_i, m_s_j, sd_s, r):
    return exp_x2_x1(s_i, m_s_i, m_s_j, sd_s, sd_s, r) + exp_x2_x1(mu_i, 0, 0, 1, 1, r)  # E[s_j|s_i] + E[mu_j | mu_i]


# In the period t_j = 2, given that j stayed for t_j =2
def e_y_j_2(s_i, mu_i, m_s_i, m_s_j, sd_s, r):
    return exp_x2_x1(s_i, m_s_i, m_s_j, sd_s, sd_s, r) + \
           exp_x2_x1(mu_i, 0, 0, 1, 1, r) + (var_x2_x1(1, r) ** 0.5) * imr(
        0,
        exp_x2_x1(mu_i, 0, 0, 1, 1, r),
        var_x2_x1(1, r) ** 0.5
    )   # E[s_j|s_i] + E[{mu_j|mu_i}|{mu_j|mu_i}>=0]


# Total expected output of the candidate j referred by the current worker i with s_i, mu_i
def e_y_j_t(s_i, mu_i, m_s_i, m_s_j, sd_s, r):
    return e_y_j_1(s_i, mu_i, m_s_i, m_s_j, sd_s, r) + \
           p_j_2(mu_i, r) * e_y_j_2(s_i, mu_i, m_s_i, m_s_j, sd_s, r)


'Wages of the workers'


# Wage of the labor market candidate m
# In the period t_m = 1
def w_m_1(m_s_m):
    return m_s_m  # E[s_m] where s_m = is the general ability of m


# In the period t_m = 2, given his abilities observed in t_m = 1 are equal to s_m and mu_m
def w_m_2(s_m):
    return s_m  # E[s_m|s_m] = s_m


# Expected wage in t_m = 2, given that m stayed in the firm
def e_w_m_2(m_s_m):
    return m_s_m  # E[s_m | mu_m >= 0] = E[s_m]


# ----------------------------------------------------------------------------
# Wage of the candidate j referred by the current employee i with s_i, mu_i
# In the period t_j = 1, given y_i
def w_j_1(s_i, m_s_i, m_s_j, sd_s, r):
    return exp_x2_x1(s_i, m_s_i, m_s_j, sd_s, sd_s, r)  # E[s_j | s_i] where s_j = is the general ability of j


# In the period t_j = 2, given s_j
def w_j_2(s_j):
    return s_j  # E[s_j|s_j] = s_j


# In the period t_j = 2, given that j stayed in the firm, i.e. given that {m_j|mu_i} >= 0
def e_w_j_2_upper(s_i, m_s_i, m_s_j, sd_s, r):
    return exp_x2_x1(s_i, m_s_i, m_s_j, sd_s, sd_s, r)  # E[s_j|s_i, {mu_j|mu_i} >= 0]


'Profits of the firm'


# (Expected) profit of the firm from employing the labor market candidate m
# In the period t_m = 1
def pi_m_1(m_s_m):
    return e_y_m_1(m_s_m) - w_m_1(m_s_m)  # difference between the output and the wage


# In the period t_m = 2, given that m stayed in the firm
def pi_m_2(m_s_m):
    return e_y_m_2(m_s_m) - e_w_m_2(m_s_m)


# Overall expected profit of the firm from the candidate m
def pi_m_t(m_s_m):
    return pi_m_1(m_s_m) + p_m_2(0, 1) * pi_m_2(m_s_m)


# ----------------------------------------------------------------------------
# Expected profit of the firm from employing the candidate j referred by the current worker i with s_i, mu_i
# In the period t_j = 1
def pi_j_1(s_i, mu_i, m_s_i, m_s_j, sd_s, r):
    return e_y_j_1(s_i, mu_i, m_s_i, m_s_j, sd_s, r) - w_j_1(s_i, m_s_i, m_s_j, sd_s, r)


# In the period t_j = 2, given that j stayed in the firm
def pi_j_2(s_i, mu_i, m_s_i, m_s_j, sd_s, r):
    return e_y_j_2(s_i, mu_i, m_s_i, m_s_j, sd_s, r) - e_w_j_2_upper(s_i, m_s_i, m_s_j, sd_s, r)


# Overall expected profit of the firm from the candidate j referred by the current worker i with s_i, mu_i
def pi_j_t(s_i, mu_i, m_s_i, m_s_j, sd_s, r):
    return pi_j_1(s_i, mu_i, m_s_i, m_s_j, sd_s, r) + p_j_2(mu_i, r) * pi_j_2(s_i, mu_i, m_s_i, m_s_j, sd_s, r) + (
            1 - p_j_2(mu_i, r)
    ) * pi_m_1(m_s_j)


# Difference between the overall expected profit of j referred by i with s_i, mu_i and the market candidate m
def delta_pi(mu_i, s_i, m_s_i, m_s_j, sd_s, r):
    return pi_j_t(s_i, mu_i, m_s_i, m_s_j, sd_s, r) - pi_m_t(m_s_j)


# ----------------------------------------------------------------------------
# Expected profit of the firm from employing the candidate j referred by the current worker i with s_i, mu_i under ERP
# In the period t_j = 1
def pi_j_1_b(s_i, mu_i, m_s_i, m_s_j, sd_s, r, b):
    return e_y_j_1(s_i, mu_i, m_s_i, m_s_j, sd_s, r) - w_j_1(s_i, m_s_i, m_s_j, sd_s, r) - b * p_b_paid(mu_i, r)


# Overall expected profit of the firm from the candidate j referred by the current worker i with s_i, mu_i under ERP
def pi_j_t_b(s_i, mu_i, m_s_i, m_s_j, sd_s, r, b):
    return pi_j_1_b(s_i, mu_i, m_s_i, m_s_j, sd_s, r, b) + p_j_2(mu_i, r) * pi_j_2(s_i, mu_i, m_s_i, m_s_j, sd_s, r) + (
            1 - p_j_2(mu_i, r)
    ) * pi_m_1(m_s_j)


# Difference between the overall expected profit of j referred by i with y_i and the market candidate m under ERP
def delta_pi_b(mu_i, s_i, m_s_i, m_s_j, sd_s, r, b):
    return pi_j_t_b(s_i, mu_i, m_s_i, m_s_j, sd_s, r, b) - pi_m_t(m_s_j)


'''Calculating thresholds, optimal bonus and overall expected profit of the firm unconditional on y_i'''


# Utility difference of the current employee i with mu_i, s_i, who decided to refer j
def utility_i(s_i, m_s_i, m_s_j, sd_s, r, p):
    return p * (w_j_1(s_i, m_s_i, m_s_j, sd_s, r) - w_m_1(m_s_j)) - c(m_s_j)


# Threshold of the current employee i, under which she never refers: s_star
def s_star(m_s_i, m_s_j, r, p):
    return m_s_i + c(m_s_j) / (r * p)


# Threshold of the firm, under which it never hires candidate j referred by i with mu_i: mu_star
def mu_star(m_s_i, m_s_j, sd_s, r):
    s_i_d = m_s_i   # dummy theta_i for mu_star, which does not in fact depend on theta_i
    return optimize.fsolve(delta_pi, x0=0, args=(s_i_d, m_s_i, m_s_j, sd_s, r))[0]


# Threshold of the current employee i, under which she never refers in case of ERP with bonus b: y_tilde_b
def utility_i_b(s_i, m_s_i, m_s_j, sd_s, r, p, b):
    return p * (w_j_1(s_i, m_s_i, m_s_j, sd_s, r) - w_m_1(m_s_j)) - c(m_s_j) + b    # * p_b_paid(mu_i, r)


# Threshold of the current employee i, under which she never refers in case of ERP with bonus b: y_tilde_b
def s_star_b(m_s_i, m_s_j, sd_s, r, p, b):
    # return m_s_i + (c(m_s_j) - b) / (r * p)
    return optimize.fsolve(utility_i_b, x0=m_s_i, args=(m_s_i, m_s_j, sd_s, r, p, b))[0]


# Threshold of the firm, under which it never hires candidate j referred by i with y_i under ERP with bonus b: y_star_b
def mu_star_b(m_s_i, m_s_j, sd_s, r, b):
    s_i_d = m_s_i   # dummy theta_i for mu_star, which does not in fact depend on theta_i
    return optimize.fsolve(delta_pi_b, x0=0, args=(s_i_d, m_s_i, m_s_j, sd_s, r, b))[0]


'''Finding the optimal bonus'''


# Optimal bonus is defined as the bonus, under which the firm maximizes its overall expected profit unconditional on y_i
# The formula of the total firm's profit: total profit = Pr(referral)*profit_from_referral + Pr(no ref)* profit_from_m
# It should be calculated in several steps:
# 1.1) Find the probability of referral Pr(ref) under voluntary referrals:
def p_ref(m_s_i, m_s_j, sd_s, r, p):
    return (1 - norm.cdf((s_star(m_s_i, m_s_j, r, p) - m_s_i) / sd_s)) * \
           (1-norm.cdf(mu_star(m_s_i, m_s_j, sd_s, r)))


# 1.2) Find the probability of referral Pr_b(ref) under ERP with bonus b:
def p_ref_b(m_s_i, m_s_j, sd_s, r, p, b):
    return (1 - norm.cdf((s_star_b(m_s_i, m_s_j, sd_s, r, p, b) - m_s_i) / sd_s)) * \
           (1-norm.cdf(mu_star_b(m_s_i, m_s_j, sd_s, r, b)))


# 2.1) Find the profit from referral under voluntary referrals:
def pi_vr_j(m_s_i, m_s_j, sd_s, r):
    s_i_d = m_s_i
    integrand_vr = lambda x: pi_j_t(s_i_d, x, m_s_i, m_s_j, sd_s, r) * norm.pdf(x)
    profit_vr_j = quad(
        integrand_vr,
        mu_star(m_s_i, m_s_j, sd_s, r),
        np.infty
    )
    return profit_vr_j[0] / (1-norm.cdf(mu_star(m_s_i, m_s_j, sd_s, r)))  # !!!!!!!


# 2.2) Find the profit from referral under ERP with bonus b:
def pi_erp_j(m_s_i, m_s_j, sd_s, r, b):
    s_i_d = m_s_i
    integrand_erp = lambda x: pi_j_t_b(s_i_d, x, m_s_i, m_s_j, sd_s, r, b) * norm.pdf(x)
    profit_erp_j = quad(
        integrand_erp,
        mu_star_b(m_s_i, m_s_j, sd_s, r, b),
        np.infty
    )
    return profit_erp_j[0] / (1-norm.cdf(mu_star_b(m_s_i, m_s_j, sd_s, r, b)))  # !!!!!!!


# 3.1) Find the overall profit under voluntary referrals:
def pi_total_vr(m_s_i, m_s_j, sd_s, r, p):
    return p_ref(m_s_i, m_s_j, sd_s, r, p) * pi_vr_j(m_s_i, m_s_j, sd_s, r) + \
           (1 - p_ref(m_s_i, m_s_j, sd_s, r, p)) * pi_m_t(m_s_j)


# 3.1) Find the overall profit under ERP with bonus b:
def pi_total_erp(m_s_i, m_s_j, sd_s, r, p, b):
    return p_ref_b(m_s_i, m_s_j, sd_s, r, p, b) * pi_erp_j(m_s_i, m_s_j, sd_s, r, b) + \
           (1 - p_ref_b(m_s_i, m_s_j, sd_s, r, p, b)) * pi_m_t(m_s_j)


# 4) Find the optimal bonus b_star:
def neg_pi_total_erp(b, m_s_i, m_s_j, sd_s, r, p):
    return - pi_total_erp(m_s_i, m_s_j, sd_s, r, p, b)  # Change the sign of the total expected profit to use fmin


def b_star(m_s_i, m_s_j, sd_s, r, p):
    bonus_star = optimize.fmin(neg_pi_total_erp, x0=c(m_s_j), args=(m_s_i, m_s_j, sd_s, r, p))
    return max(0, bonus_star[0])


# Testing
si = mean_i + 0.5
mui = 0.5
sj = si
sm = sj
# bonus = c(mean_j)/2
B_star = b_star(mean_i, mean_j, st_dev, rho, psi)

print('p_m_2 = ', p_m_2(0, 1))
print('p_j_2 = ', p_j_2(mui, rho))
print('e_y_m_1 = ', e_y_m_1(mean_j))
print('e_y_m_2 = ', e_y_m_2(mean_j))
print('e_y_m_t = ', e_y_m_t(mean_j))
print('e_y_j_1 = ', e_y_j_1(si, mui, mean_i, mean_j, st_dev, rho))
print('e_y_j_2 = ', e_y_j_2(si, mui, mean_i, mean_j, st_dev, rho))
print('e_y_j_t = ', e_y_j_t(si, mui, mean_i, mean_j, st_dev, rho))
print('w_m_1 = ', w_m_1(mean_j))
print('w_m_2 = ', w_m_2(sm))
print('e_w_m_2 = ', e_w_m_2(mean_j))
print('w_j_1 = ', w_j_1(si, mean_i, mean_j, st_dev, rho))
print('w_j_2 = ', w_j_2(sj))
print('e_w_j_2_upper = ', e_w_j_2_upper(si, mean_i, mean_j, st_dev, rho))
print('pi_m_1 = ', pi_m_1(mean_j))
print('pi_m_2 = ', pi_m_2(mean_j))
print('pi_m_t = ', pi_m_t(mean_j))
print('pi_j_1 = ', pi_j_1(si, mui, mean_i, mean_j, st_dev, rho))
print('pi_j_2 = ', pi_j_2(si, mui, mean_i, mean_j, st_dev, rho))
print('pi_j_t = ', pi_j_t(si, mui, mean_i, mean_j, st_dev, rho))
print('pi_j_1_b = ', pi_j_1_b(si, mui, mean_i, mean_j, st_dev, rho, B_star))
print('pi_j_t_b = ', pi_j_t_b(si, mui, mean_i, mean_j, st_dev, rho, B_star))
print('s_star = ', s_star(mean_i, mean_j, rho, psi))
print('mu_star = ', mu_star(mean_i, mean_j, st_dev, rho))
print('s_star_b = ', s_star_b(mean_i, mean_j, st_dev, rho, psi, B_star))
print('mu_star_b = ', mu_star_b(mean_i, mean_j, st_dev, rho, B_star))
print('p_ref = ', p_ref(mean_i, mean_j, st_dev, rho, psi))
print('p_ref_b = ', p_ref_b(mean_i, mean_j, st_dev, rho, psi, B_star))
print('pi_vr_j = ', pi_vr_j(mean_i, mean_j, st_dev, rho))
print('pi_erp_j = ', pi_erp_j(mean_i, mean_j, st_dev, rho, B_star))
print('pi_total_vr = ', pi_total_vr(mean_i, mean_j, st_dev, rho, psi))
print('pi_total_erp = ', pi_total_erp(mean_i, mean_j, st_dev, rho, psi, B_star))
# print('pi_total_erp_optimal = ', pi_total_erp(mean_i, mean_j, st_dev, rho, psi, B_star))
print('b_star = ', B_star)

# si = np.linspace(2, 6, 20)
# plt.plot(si, pi_j_t(si, mui, mean_i, mean_j, st_dev, rho), label='pi_j_t')
# plt.plot(si, pi_m_t(mean_j)+ si*0, label='pi_m_2')
# plt.plot(si, utility_i(si, mui, mean_i, mean_j, st_dev, rho, psi))
# plt.plot(si, si*0)
# plt.axvline(mu_star(si, mean_i, mean_j, st_dev, rho))
# plt.axvline(s_star(mean_i, mean_j, rho, psi))
# plt.plot(bonus, p_ref_b(mean_i, mean_j, st_dev, rho, psi, bonus))
# plt.legend()
# plt.show()
