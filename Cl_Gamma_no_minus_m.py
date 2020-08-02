import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from pyshtools.expand import spharm_lm, spharm
import healpy as hp
from time import time
from scipy.linalg import solve_triangular
from scipy import interpolate


powspect = np.loadtxt(
    '/home/wudl/project/project1_fisher/data/CMB_r/r_0/test_totCls.dat')
ell_tt = powspect[:, 0]
Dl_tt = powspect[:, 1]
Cl_tt = Dl_tt*2*np.pi/(ell_tt*(ell_tt+1))
theta = np.radians(80)
m_len = ell_tt.size
ell_len = m_len
lmax = len(ell_tt)

def plm_matrix():
    '''
    ell start from 2, delete the column of ell=0 and ell=1
    '''
    matrix_plm = spharm(lmax-1, theta, 0, normalization='ortho',
                        degrees=False, kind='complex', csphase=1)[0].real.T
    return matrix_plm


def Gamma2Cl(mat_plm, B, gamma):
    return np.matmul(np.matmul(B, np.linalg.pinv(np.matmul(mat_plm, B))), gamma)


def Cl2Gamma(mat_plm, Cl_plt):
    return np.matmul(mat_plm, Cl_plt)


s_time = time()

#smooth bl to 7 arcmin
# bl = hp.gauss_beam(np.radians(7/60), lmax=lmax)[2:]
# bl = hp.gauss_beam(np.radians(7/60), lmax=lmax)[1:]
bl = 1
mat_plm_in = plm_matrix()**2
print(np.shape(mat_plm_in))
Gamma_m = Cl2Gamma(mat_plm_in, Cl_tt*bl**2)

# bin_matrix B
q = 1.02
rt = 3
k = [round(rt*q**i) for i in range(lmax)]
w = np.cumsum(k)
idx = abs(w-lmax).argmin()
if w[idx] < lmax:
    bi = idx + 1
elif w[idx] >= lmax:
    bi = idx

bin_mat = np.zeros((lmax, bi+1))
h = np.arange(ell_tt.size).tolist()
a = 0
b = []
for i in range(bi+1):
    s = round(rt*q**i)
    bin_mat[a:a+s, i] = 1
    b.append(int(np.mean(h[a:a+s])))
    a += s

ell_bin = np.sum(bin_mat*b, axis=1)
np.save('B', bin_mat)
print(np.shape(bin_mat))

# Gamma_m to Cl
Cl_r = Gamma2Cl(mat_plm_in, bin_mat, Gamma_m)
Dl_r = ell_bin*(ell_bin+1)/2/np.pi*Cl_r
ell_r = np.arange(Cl_r.size)
plt.figure(figsize=(14, 8))


#fitting
Dl_temp = [Dl_r[i] for i in b]
# Dl_interp = interpolate.interp1d(
#     b, Dl_temp, kind='quadratic', bounds_error=False, fill_value=(Dl_r[0],Dl_r[-1]))(ell_r)
order = 50
Dl_fit = np.polyfit(b[3:-3], Dl_temp[3:-3], order)
Dl_interp = np.poly1d(Dl_fit)(ell_r)

e_time = time()
print('time cost (second)>>> ', (e_time-s_time))

#plot Gamma2Cl
plt.plot(ell_r, Dl_interp, 'y', label='polynomial fitting')
plt.scatter(ell_r, Dl_r, s=1, c='r', label='Dl_bin')
# plt.scatter(b, Dl_temp, s=1, c='r',label='Dl_bin')
# plt.plot(b,Dl_temp,'r')
plt.plot(ell_r, Dl_tt, 'b', label='Dl')
# plt.plot(ell_bin,Dl_r)
plt.ylim(-1000, 6500)
plt.xlim(-10, 2000)
plt.legend()

#plot Cl2Gamma
m_tt = np.arange(lmax)
fig, ax = plt.subplots(2, 1, figsize=(14, 8))
ax[1].semilogx(m_tt, np.sqrt(2*m_tt*Gamma_m), label='$\Gamma_m$')
ax[1].set_xlabel('$m$')
ax[1].set_ylabel('$(2m\Gamma_m)^{1/2}$')
ax[0].semilogx(ell_tt, np.sqrt(Dl_tt/2*bl**2), label='Cl')
ax[0].set_xlabel('$\ell$')
ax[0].set_ylabel('$(\ell(2\ell+1)C_\ell/4\pi)^{1/2}$')

plt.legend()
plt.show()
