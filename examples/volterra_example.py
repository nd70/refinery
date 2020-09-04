import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('gruvbox')
import library as lib
import numpy as np
import scipy.linalg as sl


#-------------------------------------------------------------------------------
#                                Generate Data
#-------------------------------------------------------------------------------
"""
Given
-----
    d(true_sig, wit1, wit2)
    wit1
    wit2

Find
----
    f(wit1, wit2)

such that

    d(true_sig, wit1, wit2) - f(wit1, wit2) = true_sig

"""

dur, fs = 128, 512
t = np.linspace(0, dur, dur * fs)
true_sig = np.random.rand(t.size) * np.sin(2 * np.pi * 5 * t)
wit1 = np.random.rand(t.size)
wit2 = np.random.rand(t.size) * np.cos(2 * np.pi * 0.5 * t)
d = true_sig + 0.5 * wit1 * wit2
M=1

#-------------------------------------------------------------------------------
#                               Run the Filter
#-------------------------------------------------------------------------------
print('calculating 3-point correlation')
P = lib.multi_corr(d, wit1, wit2, M=M)

print('calculating 4-point correlation')
out = lib.four_point_corr(wit1, wit2, wit1, wit2, M)
vc = out.reshape(((M+1)**2, (M+1)**2)).T

print('solving for the filter weights')
weights = sl.solve(vc, P)

print('applying the weights to the aux channels')
est = lib.apply_weights_2d(wit1, wit2, weights.reshape(((M+1), (M+1))))

print('cleaning the system signal')
clean = d - est
print('MSE = ', np.mean((clean-true_sig)**2))

#-------------------------------------------------------------------------------
#                                  Visualize
#-------------------------------------------------------------------------------
print('making plots')
vis=64
plt.plot(t[-vis:], d[-vis:], label='System Output', alpha=0.8)
plt.plot(t[-vis:], true_sig[-vis:], label='True Signal')
plt.plot(t[-vis:], clean[-vis:], label='Cleaned', ls='-.')
plt.legend()
plt.grid(True)
plt.title('Second Order Volterra Filter')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.savefig('volterra.png', dpi=300)
plt.close()
