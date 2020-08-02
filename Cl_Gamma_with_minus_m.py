from time import time
import healpy as hp
from scipy import special
import matplotlib.pyplot as plt
import numpy as np
from pyshtools.expand import spharm_lm, spharm

Gamma_m = np.loadtxt('Gamma_m.txt')
powspect = np.loadtxt(
    '/home/wudl/project/project1_fisher/data/CMB_r/r_0/test_totCls.dat')
ell_tt = powspect[:, 0]
Dl_tt = powspect[:, 1]
Cl_tt = Dl_tt*2*np.pi/(ell_tt*(ell_tt+1))
lmax = Cl_tt.size
Cl_tt = Cl_tt[:lmax]
ell_tt = ell_tt[:lmax]
Dl_tt = Dl_tt[:lmax]
theta = np.radians(70)

def plm_matrix():
    '''
    ell start from 2, delete the column of ell=0 and ell=1
    '''
    matrix_plm = spharm(lmax-1, theta, 0, normalization='ortho',
                        degrees=False, kind='complex', csphase=-1).real
    return matrix_plm

def Gamma2Cl(mat_plm, B, gamma):
    return np.matmul(np.matmul(B, np.linalg.pinv(np.matmul(mat_plm, B))), gamma)

def Cl2Gamma(mat_plm, Cl_plt):
    return np.matmul(mat_plm, Cl_plt)


s_time = time()

#smooth bl to 7 arcmin
# bl = hp.gauss_beam(np.radians(7/60), lmax=lmax-1)
bl = 1
plm_mat = plm_matrix()
mat_plm_plus, mat_plm_minus = np.power(
    plm_mat[0].T, 2), np.power(plm_mat[1].T, 2)

mat_plm_plus = np.delete(mat_plm_plus, 0, 0)
mat_plm_minus = mat_plm_minus[::-1]
mat_plm = np.row_stack((mat_plm_minus, mat_plm_plus))
Gamma_m = Cl2Gamma(mat_plm, Cl_tt*bl**2)

# bin_matrix B
q = 1.02
rt = 3
k = [round(rt*q**i) for i in range(lmax)]
w = np.cumsum(k)
idx = abs(w-lmax).argmin()
if w[idx] < lmax:
    bi = idx + 1
else:
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
# np.save('B', bin_mat)
print(np.shape(bin_mat))

#compute Gamma_m to Cl
Cl_r = Gamma2Cl(mat_plm, bin_mat, Gamma_m)
Dl_r = ell_bin*(ell_bin+1)/2/np.pi*Cl_r
ell_r = np.arange(2,Cl_r.size)

#plot Gamma2Cl
plt.figure(figsize=(14, 8))
plt.scatter(ell_r, Dl_r[2:], s=1, c='r', label='Dl_bin')
plt.plot(ell_tt, Dl_tt, 'b', label='Dl')
plt.ylim(-1000, 6500)
plt.xlim(-10, 2000)

#plot Cl2Gamma
m_tt = np.arange(lmax)
fig, ax = plt.subplots(2, 1, figsize=(14, 8))
ax[1].semilogx(m_tt, np.sqrt(2*m_tt*Gamma_m[m_tt.size-1:]),
               label='$\Gamma_m_plus$')
ax[1].set_xlabel('$m$')
ax[1].set_ylabel('$(2m\Gamma_m)^{1/2}$')
ax[1].legend()
ax[0].semilogx(ell_tt, np.sqrt(Dl_tt/2*bl**2), label='Cl')
ax[0].set_xlabel('$\ell$')
ax[0].set_ylabel('$(\ell(2\ell+1)C_\ell/4\pi)^{1/2}$')

plt.legend()
plt.show()
