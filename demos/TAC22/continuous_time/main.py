import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
import numpy as np
font = {
	# 'font.family' : 'Helvetica',
    'font.size'   : 16,
    'text.usetex' : True,
}
plt.rcParams.update(**font)


def compute_pstar(tau, L, Tf, v):
	inf_xv = tau * (np.exp(-L / (v*tau)) - 1)
	pstar = (tau - inf_xv) * np.exp(Tf/tau) - tau
	return pstar, inf_xv


def vp(x, tau, p, L, Tf, v):
	"""
	Input:
		x: np.ndarray of shape (N,)
		tau: np.ndarray of shape (T,), representing values of the discount factor
		p: np.ndarray of shape (T, P), where p[i, :] is the penalties for which V_p is computed with discounting tau[i]
		L: float
		Tf: float
		v: float
	Output:
		V: np.ndarray of shape (T, P, N)
		x_viable: np.ndarray of the points in x that are viable
		x_unviable: np.ndarray of the points in x that are unviable
	"""
	# tau is a 1D array
	# p is a 2D array: tau.shape[0] x number of penalties / tau
	# The output is a 3D array: tau.shape[0] x p.shape[1] x x.shape[0]

	x_viable = x >= 0
	x_unviable = np.logical_not(x_viable)

	tau = tau[:, np.newaxis, np.newaxis]
	p = p[:, :, np.newaxis]
	x = x[np.newaxis, np.newaxis, :]

	V_xv_safe = tau * (np.exp((x-L)/(v*tau)) - 1)
	V_xv_safe = np.tile(V_xv_safe, (1, p.shape[1], 1)) # shape (tau, p, x)
	V_xv_unsafe = tau * (np.exp(-x/(v*tau)) - 1) + tau * (1 - np.exp(-Tf/tau))*np.exp(-x/(v*tau)) - p * np.exp((Tf + (x/v))/tau)

	V_xv = np.maximum(V_xv_safe, V_xv_unsafe)

	V_xu = tau * (1 - np.exp(-Tf/tau * (1 + x/L)))- p * np.exp(-Tf/tau * (1 + x / L))

	V = V_xv
	V[:, :, x_unviable] = V_xu[:, :, x_unviable]

	return V, x_viable, x_unviable


def plot_for_tau(chosen_tau=1., chosen_Tf=1., plot_only_top=False):
	"""
	Input:
		chosen_tau: float: the discount rate used to plot the value function (top plot)
		chosen_Tf: float: the maximum time-to-failure used to plot the value function (top plot)
	"""
	L = 1.
	v = .2
	tau = np.linspace(0.4, 20, 1000)
	i_chosen_tau = np.argmin(np.abs(tau - chosen_tau))
	Tf = np.linspace(0, 5, 1000)
	pstar, inf_xv_for_pstar = compute_pstar(tau, L, chosen_Tf, v) # pstar as a function of tau (top and middle plots)
	pstar_Tf, _ = compute_pstar(tau[i_chosen_tau], L, Tf, v) # pstar as a function of Tf (bottom plot)

	IND_PSTAR = 1
	IND_P_SUP_PSTAR = 2
	p = np.vstack((0*pstar, 1.*pstar, 2*pstar)).T # The penalties used in the top plot are 0, pstar, and 2*pstar
	x = np.linspace(-1, 1, 1001)


	value, x_viable, x_unviable = vp(x, tau, p, L, chosen_Tf, v)
	value = value - 1 # So everything is negative and fits on a log scale
	# Check that the values don't depend on p for p > pstar
	assert np.isclose(value[:, IND_PSTAR, x_viable], value[:, IND_P_SUP_PSTAR, x_viable]).all(), \
		"The values in the viability kernel for p = p* and for p = 2 p* are different"


	alpha_inf = value[i_chosen_tau, IND_P_SUP_PSTAR, x_unviable].max()
	alpha_sup = value[i_chosen_tau, IND_P_SUP_PSTAR, x_viable].min()

	##############
	## Plotting ##
	##############

	if plot_only_top:
		fig = plt.figure(figsize=(8, 3.3), constrained_layout=True)
		gs = gridspec.GridSpec(1, 1, figure=fig)
	else:
		fig = plt.figure(figsize=(8, 10), constrained_layout=True)
		gs = gridspec.GridSpec(3, 1, figure=fig)
	ax0 = fig.add_subplot(gs[0, 0])
	for i_p in range(p.shape[1]):
		lab = r'$p = 2\cdot p^\star$' if i_p == IND_P_SUP_PSTAR else r'$p = p^\star$' if i_p == IND_PSTAR else r'$p = 0$'
		ax0.plot(x, value[i_chosen_tau, i_p, :], label=lab)

	ax0.set_yscale('symlog', linthresh= 0.1, subs=[2, 3, 4, 5, 6, 7, 8, 9])
	ax0.axhline(alpha_inf, linestyle='dashed', linewidth=.5, color=(0, 0, 0))
	ax0.axhline(alpha_sup, linestyle='dashed', linewidth=.5, color=(0, 0, 0))
	ax0.legend(loc='lower right')
	ax0.set_xlabel(r'$x$')
	ax0.set_ylabel(r'$V_p(x) - 1$')
	## Adding the y axis labels of alpha inf and sup
	# From Github
	trans = transforms.blended_transform_factory(
		ax0.get_yticklabels()[0].get_transform(), ax0.transData)
	ax0.text(0.02, alpha_inf, r'$\alpha_{\mathrm{inf}}-1$', color="black", transform=trans, ha="left", va="bottom")
	ax0.text(0.02, alpha_sup, r'$\alpha_{\mathrm{sup}}-1$', color="black", transform=trans, ha="left", va="bottom")

	if not plot_only_top:
		ax1 = fig.add_subplot(gs[1, 0])
		ax1.plot(tau, pstar, label=r'$p^*(\tau)$')
		ax1.plot(tau, -inf_xv_for_pstar, alpha=0.5, label=r'$-\inf_{\mathcal{X}_V}V$')
		ax1.plot(tau, np.exp(chosen_Tf/tau), alpha=0.5, label=r'$\exp\left(\frac{T_f}{\tau}\right)$')
		ax1.set_xlabel(r'$\tau$')
		ax1.set_ylabel(r'$p^*(\tau)$')
		# Plot the arrow
		M = max(pstar.max(), (-inf_xv_for_pstar).max(), np.exp(chosen_Tf/tau).max())
		m = min(pstar.min(), (-inf_xv_for_pstar).min(), np.exp(chosen_Tf/tau).min())
		ax1.annotate("",
					xy=(tau[i_chosen_tau], pstar[i_chosen_tau]), xycoords='data',
					xytext=(tau[i_chosen_tau], M+ (M-m)*.14), textcoords='data',
					arrowprops=dict(arrowstyle="<-",
									connectionstyle="arc3", color='gray', lw=.5, linestyle='dashed'),
					)
		ax1.legend(loc='upper right')

		ax2 = fig.add_subplot(gs[2, 0])
		ax2.semilogy(Tf, pstar_Tf)
		ax2.set_xlabel(r'$T_f$')
		ax2.set_ylabel(r'$p^*(T_f)$')

	plt.savefig(f'CONTINUOUS_TIME_{chosen_tau}_{chosen_Tf}.pdf', format='pdf', dpi=300)



# Change this line to plot the value function for a different value of tau/Tf in the top plot
# Supported values are should be respectively in [0.4, 20] and (0, 5)
# The value function can be computed outside of these bounds, but the plots may look weird
PLOT_ONLY_TOP = False 
plot_for_tau(chosen_tau=.5, chosen_Tf=1., plot_only_top=PLOT_ONLY_TOP)
plot_for_tau(chosen_tau=1., chosen_Tf=1., plot_only_top=PLOT_ONLY_TOP)
plot_for_tau(chosen_tau=2., chosen_Tf=1., plot_only_top=PLOT_ONLY_TOP)
plot_for_tau(chosen_tau=4., chosen_Tf=1., plot_only_top=PLOT_ONLY_TOP)
