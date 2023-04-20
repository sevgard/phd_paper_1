from matplotlib import pyplot as plt
from imperf_info import c
from imperf_info import y_tilde
from imperf_info import y_star
from imperf_info import p_ref
from imperf_info import pi_total_vr
from imperf_info import b_star
from imperf_info import y_tilde_b
from imperf_info import y_star_b
from imperf_info import p_ref_b
from imperf_info import pi_total_erp

plt.rcParams["figure.figsize"] = [7.50, 7.50]
plt.rcParams["figure.autolayout"] = True

'''Plot the graphs for the mean when mean_i = mean_j'''


# Enter parameters
mean_i = 4
mean_j = 4
st_dev = 1
rho = 0.5
psi = 0.5

# Create empty lists:
list_cost_ref = []
list_y_tilde = []
list_y_star = []
list_p_ref = []
list_pi_total_vr = []
list_b_star = []
list_y_tilde_b = []
list_y_star_b = []
list_p_ref_b = []
list_pi_total_erp = []

# Iterate over the variable of interest to append the lists
# 1. common mean: mean_i = mean_j
for i in range(30):
    j = 0 + i*(8/30)
    B_star = b_star(j, j, st_dev, rho, psi)
    list_cost_ref.append((j, c(j)))
    list_y_tilde.append((j, y_tilde(j, j, st_dev, rho, psi)))
    list_y_star.append((j, y_star(j, j, st_dev, rho)))
    list_p_ref.append((j, p_ref(j, j, st_dev, rho, psi)))
    list_pi_total_vr.append((j, pi_total_vr(j, j, st_dev, rho, psi)))
    list_b_star.append((j, B_star))
    list_y_tilde_b.append((j, y_tilde_b(j, j, st_dev, rho, psi, B_star)))
    list_y_star_b.append((j, y_star_b(j, j, st_dev, rho, B_star)))
    list_p_ref_b.append((j, p_ref_b(j, j, st_dev, rho, psi, B_star)))
    list_pi_total_erp.append((j, pi_total_erp(j, j, st_dev, rho, psi, B_star)))

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig1.suptitle(('\u0398\u0305_i = variable', '\u0398\u0305_j = variable', '\u03C3 = {}'.format(st_dev),
              '\u03C1={}'.format(rho), '\u03C8={}'.format(psi)), fontsize=12)
ax1.plot(*zip(*list_y_tilde_b), '--g', label='\u0079\u0303(b)')
ax1.plot(*zip(*list_y_star_b), '--b', label='\u0079\u002A(b)')
ax1.plot(*zip(*list_y_tilde), '-g', label='\u0079\u0303')
ax1.plot(*zip(*list_y_star), '-b', label='\u0079\u002A')
ax1.set_title('Thresholds')
leg_1 = ax1.legend()
ax2.plot(*zip(*list_p_ref_b), '--b', label='P(refer under ERP)')
ax2.plot(*zip(*list_p_ref), '-g', label='P(voluntary refer)')
ax2.set_title('Probability of Referral')
leg_2 = ax2.legend()
ax3.plot(*zip(*list_pi_total_vr), '-b', label='\u03A0(VR)')
ax3.plot(*zip(*list_pi_total_erp), '--b', label='\u03A0(ERP)')
ax3.set_title('Firm Profit')
leg_3 = ax3.legend()
ax4.plot(*zip(*list_cost_ref), '-k', label='costs C(\u0398\u0305)')
ax4.plot(*zip(*list_b_star), '-b', label='bonus b\u002A')
ax4.set_title('Referral Costs and Bonus')
leg_4 = ax4.legend()
plt.show()


# # Empty lists:
# list_cost_ref = []
# list_y_tilde = []
# list_y_star = []
# list_p_ref = []
# list_pi_total_vr = []
# list_b_star = []
# list_y_tilde_b = []
# list_y_star_b = []
# list_p_ref_b = []
# list_pi_total_erp = []
#
# # 2. mean_i
# for i in range(30):
#     j = 0 + i*(8/30)
#     B_star = b_star(j, mean_j, st_dev, rho, psi)
#     list_cost_ref.append((j, c(mean_j)))
#     list_y_tilde.append((j, y_tilde(j, mean_j, st_dev, rho, psi)))
#     list_y_star.append((j, y_star(j, mean_j, st_dev, rho)))
#     list_p_ref.append((j, p_ref(j, mean_j, st_dev, rho, psi)))
#     list_pi_total_vr.append((j, pi_total_vr(j, mean_j, st_dev, rho, psi)))
#     list_b_star.append((j, B_star))
#     list_y_tilde_b.append((j, y_tilde_b(j, mean_j, st_dev, rho, psi, B_star)))
#     list_y_star_b.append((j, y_star_b(j, mean_j, st_dev, rho, B_star)))
#     list_p_ref_b.append((j, p_ref_b(j, mean_j, st_dev, rho, psi, B_star)))
#     list_pi_total_erp.append((j, pi_total_erp(j, mean_j, st_dev, rho, psi, B_star)))
#
# fig2, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2)
# fig2.suptitle(('\u0398\u0305_i = variable', '\u0398\u0305_j = {}'.format(mean_j), '\u03C3 = {}'.format(st_dev),
#               '\u03C1={}'.format(rho), '\u03C8={}'.format(psi)), fontsize=12)
# ax21.plot(*zip(*list_y_tilde_b), '--g', label='\u0079\u0303(b)')
# ax21.plot(*zip(*list_y_star_b), '--b', label='\u0079\u002A(b)')
# ax21.plot(*zip(*list_y_tilde), '-g', label='\u0079\u0303')
# ax21.plot(*zip(*list_y_star), '-b', label='\u0079\u002A')
# ax21.set_title('Thresholds')
# leg_1 = ax21.legend()
# ax22.plot(*zip(*list_p_ref_b), '--b', label='P(refer under ERP)')
# ax22.plot(*zip(*list_p_ref), '-g', label='P(voluntary refer)')
# ax22.set_title('Probability of Referral')
# leg_2 = ax22.legend()
# ax23.plot(*zip(*list_pi_total_vr), '-b', label='\u03A0(VR)')
# ax23.plot(*zip(*list_pi_total_erp), '--b', label='\u03A0(ERP)')
# ax23.set_title('Firm Profit')
# leg_3 = ax23.legend()
# ax24.plot(*zip(*list_cost_ref), '-k', label='costs C(\u0398\u0305)')
# ax24.plot(*zip(*list_b_star), '-b', label='bonus b\u002A')
# ax24.set_title('Referral Costs and Bonus')
# leg_4 = ax24.legend()
# plt.show()


# Empty lists:
list_cost_ref = []
list_y_tilde = []
list_y_star = []
list_p_ref = []
list_pi_total_vr = []
list_b_star = []
list_y_tilde_b = []
list_y_star_b = []
list_p_ref_b = []
list_pi_total_erp = []

# 3. mean_j
for i in range(30):
    j = 0 + i*(8/30)
    B_star = b_star(mean_i, j, st_dev, rho, psi)
    list_cost_ref.append((j, c(j)))
    list_y_tilde.append((j, y_tilde(mean_i, j, st_dev, rho, psi)))
    list_y_star.append((j, y_star(mean_i, j, st_dev, rho)))
    list_p_ref.append((j, p_ref(mean_i, j, st_dev, rho, psi)))
    list_pi_total_vr.append((j, pi_total_vr(mean_i, j, st_dev, rho, psi)))
    list_b_star.append((j, B_star))
    list_y_tilde_b.append((j, y_tilde_b(mean_i, j, st_dev, rho, psi, B_star)))
    list_y_star_b.append((j, y_star_b(mean_i, j, st_dev, rho, B_star)))
    list_p_ref_b.append((j, p_ref_b(mean_i, j, st_dev, rho, psi, B_star)))
    list_pi_total_erp.append((j, pi_total_erp(mean_i, j, st_dev, rho, psi, B_star)))

fig3, ((ax31, ax32), (ax33, ax34)) = plt.subplots(2, 2)
fig3.suptitle(('\u0398\u0305_i = {}'.format(mean_i), '\u0398\u0305_j = variable', '\u03C3 = {}'.format(st_dev),
              '\u03C1={}'.format(rho), '\u03C8={}'.format(psi)), fontsize=12)
ax31.plot(*zip(*list_y_tilde_b), '--g', label='\u0079\u0303(b)')
ax31.plot(*zip(*list_y_star_b), '--b', label='\u0079\u002A(b)')
ax31.plot(*zip(*list_y_tilde), '-g', label='\u0079\u0303')
ax31.plot(*zip(*list_y_star), '-b', label='\u0079\u002A')
ax31.set_title('Thresholds')
leg_1 = ax31.legend()
ax32.plot(*zip(*list_p_ref_b), '--b', label='P(refer under ERP)')
ax32.plot(*zip(*list_p_ref), '-g', label='P(voluntary refer)')
ax32.set_title('Probability of Referral')
leg_2 = ax32.legend()
ax33.plot(*zip(*list_pi_total_vr), '-b', label='\u03A0(VR)')
ax33.plot(*zip(*list_pi_total_erp), '--b', label='\u03A0(ERP)')
ax33.set_title('Firm Profit')
leg_3 = ax33.legend()
ax34.plot(*zip(*list_cost_ref), '-k', label='costs C(\u0398\u0305)')
ax34.plot(*zip(*list_b_star), '-b', label='bonus b\u002A')
ax34.set_title('Referral Costs and Bonus')
leg_4 = ax34.legend()
plt.show()


# Empty lists:
list_cost_ref = []
list_y_tilde = []
list_y_star = []
list_p_ref = []
list_pi_total_vr = []
list_b_star = []
list_y_tilde_b = []
list_y_star_b = []
list_p_ref_b = []
list_pi_total_erp = []


# 4. st_dev
for i in range(30):
    j = 0.5 + i*(1.5/30)
    B_star = b_star(mean_i, mean_j, j, rho, psi)
    list_cost_ref.append((j, c(mean_j)))
    list_y_tilde.append((j, y_tilde(mean_i, mean_j, j, rho, psi)))
    list_y_star.append((j, y_star(mean_i, mean_j, j, rho)))

    list_p_ref.append((j, p_ref(mean_i, mean_j, j, rho, psi)))
    list_pi_total_vr.append((j, pi_total_vr(mean_i, mean_j, j, rho, psi)))
    list_b_star.append((j, B_star))
    list_y_tilde_b.append((j, y_tilde_b(mean_i, mean_j, j, rho, psi, B_star)))
    list_y_star_b.append((j, y_star_b(mean_i, mean_j, j, rho, B_star)))
    list_p_ref_b.append((j, p_ref_b(mean_i, mean_j, j, rho, psi, B_star)))
    list_pi_total_erp.append((j, pi_total_erp(mean_i, mean_j, j, rho, psi, B_star)))

fig4, ((ax41, ax42), (ax43, ax44)) = plt.subplots(2, 2)
fig4.suptitle(('\u0398\u0305_i = {}'.format(mean_i), '\u0398\u0305_j = {}'.format(mean_i), '\u03C3\u00B2 = variable',
              '\u03C1={}'.format(rho), '\u03C8={}'.format(psi)), fontsize=12)
ax41.plot(*zip(*list_y_tilde_b), '--g', label='\u0079\u0303(b)')
ax41.plot(*zip(*list_y_star_b), '--b', label='\u0079\u002A(b)')
ax41.plot(*zip(*list_y_tilde), '-g', label='\u0079\u0303')
ax41.plot(*zip(*list_y_star), '-b', label='\u0079\u002A')
ax41.set_title('Thresholds')
leg_1 = ax41.legend()
ax42.plot(*zip(*list_p_ref_b), '--b', label='P(refer under ERP)')
ax42.plot(*zip(*list_p_ref), '-g', label='P(voluntary refer)')
ax42.set_title('Probability of Referral')
leg_2 = ax42.legend()
ax43.plot(*zip(*list_pi_total_vr), '-b', label='\u03A0(VR)')
ax43.plot(*zip(*list_pi_total_erp), '--b', label='\u03A0(ERP)')
ax43.set_title('Firm Profit')
leg_3 = ax43.legend()
ax44.plot(*zip(*list_cost_ref), '-k', label='costs C(\u0398\u0305)')
ax44.plot(*zip(*list_b_star), '-b', label='bonus b\u002A')
ax44.set_title('Referral Costs and Bonus')
leg_4 = ax44.legend()
plt.show()


# Empty lists:
list_cost_ref = []
list_y_tilde = []
list_y_star = []
list_p_ref = []
list_pi_total_vr = []
list_b_star = []
list_y_tilde_b = []
list_y_star_b = []
list_p_ref_b = []
list_pi_total_erp = []


# 5. rho
for i in range(30):
    j = 0.1 + i*(0.89/30)
    B_star = b_star(mean_i, mean_j, st_dev, j, psi)
    list_cost_ref.append((j, c(mean_j)))
    list_y_tilde.append((j, y_tilde(mean_i, mean_j, st_dev, j, psi)))
    list_y_star.append((j, y_star(mean_i, mean_j, st_dev, j)))
    list_p_ref.append((j, p_ref(mean_i, mean_j, st_dev, j, psi)))
    list_pi_total_vr.append((j, pi_total_vr(mean_i, mean_j, st_dev, j, psi)))
    list_b_star.append((j, B_star))
    list_y_tilde_b.append((j, y_tilde_b(mean_i, mean_j, st_dev, j, psi, B_star)))
    list_y_star_b.append((j, y_star_b(mean_i, mean_j, st_dev, j, B_star)))
    list_p_ref_b.append((j, p_ref_b(mean_i, mean_j, st_dev, j, psi, B_star)))
    list_pi_total_erp.append((j, pi_total_erp(mean_i, mean_j, st_dev, j, psi, B_star)))

fig5, ((ax51, ax52), (ax53, ax54)) = plt.subplots(2, 2)
fig5.suptitle(('\u0398\u0305_i = {}'.format(mean_i), '\u0398\u0305_j = {}'.format(mean_i), '\u03C3 = {}'.format(st_dev),
              '\u03C1=variable', '\u03C8={}'.format(psi)), fontsize=12)
ax51.plot(*zip(*list_y_tilde_b), '--g', label='\u0079\u0303(b)')
ax51.plot(*zip(*list_y_star_b), '--b', label='\u0079\u002A(b)')
ax51.plot(*zip(*list_y_tilde), '-g', label='\u0079\u0303')
ax51.plot(*zip(*list_y_star), '-b', label='\u0079\u002A')
ax51.set_title('Thresholds')
leg_1 = ax51.legend()
ax52.plot(*zip(*list_p_ref_b), '--b', label='P(refer under ERP)')
ax52.plot(*zip(*list_p_ref), '-g', label='P(voluntary refer)')
ax52.set_title('Probability of Referral')
leg_2 = ax52.legend()
ax53.plot(*zip(*list_pi_total_vr), '-b', label='\u03A0(VR)')
ax53.plot(*zip(*list_pi_total_erp), '--b', label='\u03A0(ERP)')
ax53.set_title('Firm Profit')
leg_3 = ax53.legend()
ax54.plot(*zip(*list_cost_ref), '-k', label='costs C(\u0398\u0305)')
ax54.plot(*zip(*list_b_star), '-b', label='bonus b\u002A')
ax54.set_title('Referral Costs and Bonus')
leg_4 = ax54.legend()
plt.show()


# Empty lists:
list_cost_ref = []
list_y_tilde = []
list_y_star = []
list_p_ref = []
list_pi_total_vr = []
list_b_star = []
list_y_tilde_b = []
list_y_star_b = []
list_p_ref_b = []
list_pi_total_erp = []


# 6. psi
for i in range(30):
    j = 0.1 + i*(0.89/30)
    B_star = b_star(mean_i, mean_j, st_dev, rho, j)
    list_cost_ref.append((j, c(mean_j)))
    list_y_tilde.append((j, y_tilde(mean_i, mean_j, st_dev, rho, j)))
    list_y_star.append((j, y_star(mean_i, mean_j, st_dev, rho)))
    list_p_ref.append((j, p_ref(mean_i, mean_j, st_dev, rho, j)))
    list_pi_total_vr.append((j, pi_total_vr(mean_i, mean_j, st_dev, rho, j)))
    list_b_star.append((j, B_star))
    list_y_tilde_b.append((j, y_tilde_b(mean_i, mean_j, st_dev, rho, j, B_star)))
    list_y_star_b.append((j, y_star_b(mean_i, mean_j, st_dev, rho, B_star)))
    list_p_ref_b.append((j, p_ref_b(mean_i, mean_j, st_dev, rho, j, B_star)))
    list_pi_total_erp.append((j, pi_total_erp(mean_i, mean_j, st_dev, rho, j, B_star)))

fig6, ((ax61, ax62), (ax63, ax64)) = plt.subplots(2, 2)
fig6.suptitle(('\u0398\u0305_i = {}'.format(mean_i), '\u0398\u0305_j = {}'.format(mean_j), '\u03C3 = {}'.format(st_dev),
              '\u03C1={}'.format(rho), '\u03C8 = variable'), fontsize=12)
ax61.plot(*zip(*list_y_tilde_b), '--g', label='\u0079\u0303(b)')
ax61.plot(*zip(*list_y_star_b), '--b', label='\u0079\u002A(b)')
ax61.plot(*zip(*list_y_tilde), '-g', label='\u0079\u0303')
ax61.plot(*zip(*list_y_star), '-b', label='\u0079\u002A')
ax61.set_title('Thresholds')
leg_1 = ax61.legend()
ax62.plot(*zip(*list_p_ref_b), '--b', label='P(refer under ERP)')
ax62.plot(*zip(*list_p_ref), '-g', label='P(voluntary refer)')
ax62.set_title('Probability of Referral')
leg_2 = ax62.legend()
ax63.plot(*zip(*list_pi_total_vr), '-b', label='\u03A0(VR)')
ax63.plot(*zip(*list_pi_total_erp), '--b', label='\u03A0(ERP)')
ax63.set_title('Firm Profit')
leg_3 = ax63.legend()
ax64.plot(*zip(*list_cost_ref), '-k', label='costs C(\u0398\u0305)')
ax64.plot(*zip(*list_b_star), '-b', label='bonus b\u002A')
ax64.set_title('Referral Costs and Bonus')
leg_4 = ax64.legend()
plt.show()
