import numpy as np
import numba
import matplotlib.pyplot as plt
import argparse, os
import tomli


parser = argparse.ArgumentParser()
parser.add_argument("--refine", type=int, default=1)
parser.add_argument("--config", type=str, default="config.toml")
args = parser.parse_args()
refine = args.refine
config = args.config
if not os.path.exists(config):
    raise FileNotFoundError(f"config file {config} not found")
with open(config, "rb") as f: config = tomli.load(f)

predir = f"{config['title']}-{refine}"
os.makedirs(predir, exist_ok=True)

lx = config["geometry"]["lx"]
ly = config["geometry"]["ly"]
nx = config["geometry"]["nx"]*refine
ny = config["geometry"]["ny"]*refine
hx = lx/nx
hy = ly/ny
x = np.array([ (i+0.5)*hx for i in range(nx) ])
y = np.array([ (j+0.5)*hy for j in range(ny) ])
keff_ref = config["solution"]["k_eff"]

material = [{}]
for i in range(config["material"]["total"]):
    material.append(config["material"][f"mat_{i+1}"])
D1, D2 = np.zeros(nx*ny), np.zeros(nx*ny)
a1, a2 = np.zeros(nx*ny), np.zeros(nx*ny)
nf1, nf2 = np.zeros(nx*ny), np.zeros(nx*ny)
s12 = np.zeros(nx*ny)
gx, gy = config["geometry"]["gx"], config["geometry"]["gy"]
gnx, gny = config["geometry"]["nx"], config["geometry"]["ny"]
content = np.fromstring(config["geometry"]["content"], dtype=int, sep="\n").reshape(gnx, gny)
ext_dist = config["boundary"]["ext_dist"]

@numba.jit(nopython=True) 
def in_rect(x, y, x1, y1, w, h): return x1 <= x <= x1+w and y1 <= y <= y1+h
@numba.jit(nopython=True)
def outside(x, y):
    # 未在矩形区域内或者在矩形区域内但材料为0
    return (not in_rect(x, y, 0, 0, lx, ly)) \
        or (content[int(x//gx), int(y//gy)] == 0)

# 曲率修正
Bz2 = config["material"]["Bz_sqr"]
for i in range(nx):
    for j in range(ny):
        xx, yy = x[i], y[j]
        if outside(xx, yy): continue
        mymat = material[content[int(xx//gx), int(yy//gy)]]
        idx = i+j*nx
        D1[idx], D2[idx] = mymat["d1"], mymat["d2"]
        a1[idx], a2[idx] = mymat["a1"], mymat["a2"]
        nf1[idx], nf2[idx] = mymat["nf1"], mymat["nf2"]
        s12[idx] = mymat["s12"]
        a1[idx] += D1[idx]*Bz2
        a2[idx] += D2[idx]*Bz2

def gen_laplacian():
    @numba.jit(nopython=True)
    def idx_A(i, j): return i+j*nx
    @numba.jit(nopython=True)
    def idx_B(i, j): return idx_A(i, j)+nx*ny
    @numba.jit(nopython=True)
    def gen_sub():
        # 生成一个满的拉普拉斯矩阵(-DΔ(u,v))
        triple_A = []
        for i in range(nx):
            for j in range(ny):
                if outside(x[i], y[j]):
                    # 设置 phi=0
                    triple_A.append((idx_A(i, j), idx_A(i, j), 1))
                    triple_A.append((idx_B(i, j), idx_B(i, j), 1))
                    continue

                # 1) 设置phi1
                temp = 0
                top, bot, lef, rig  = -1, -1, -1, -1
                val_up, val_rig = 1, 1
                d = D1[idx_A(i, j)] * ext_dist

                D1_top = 2/(1/D1[idx_A(i, j)] + 1/D1[idx_A(i, j+1)]) if not outside(x[i], y[j]+hy) else D1[idx_A(i, j)]
                D1_bot = 2/(1/D1[idx_A(i, j)] + 1/D1[idx_A(i, j-1)]) if not outside(x[i], y[j]-hy) else D1[idx_A(i, j)]
                D1_lef = 2/(1/D1[idx_A(i, j)] + 1/D1[idx_A(i-1, j)]) if not outside(x[i]-hx, y[j]) else D1[idx_A(i, j)]
                D1_rig = 2/(1/D1[idx_A(i, j)] + 1/D1[idx_A(i+1, j)]) if not outside(x[i]+hx, y[j]) else D1[idx_A(i, j)]

                # ∂n=0
                if i==0: lef = 0
                if j==0: bot = 0
                # phi=0
                if outside(x[i]+hx, y[j]):
                    rig = 0
                    val_rig = hx/(hx/2+d)
                if outside(x[i], y[j]+hy):
                    top = 0
                    val_up = hy/(hy/2+d)
                # update triple
                if top != 0: triple_A.append((idx_A(i, j), idx_A(i, j+1), top*D1_top/(hy**2)))
                if bot != 0: triple_A.append((idx_A(i, j), idx_A(i, j-1), bot*D1_bot/(hy**2)))
                if lef != 0: triple_A.append((idx_A(i, j), idx_A(i-1, j), lef*D1_lef/(hx**2)))
                if rig != 0: triple_A.append((idx_A(i, j), idx_A(i+1, j), rig*D1_rig/(hx**2)))
                
                temp += val_up*D1_top/(hy**2)
                temp += D1_bot/(hy**2) if bot != 0 else 0
                temp += D1_lef/(hx**2) if lef != 0 else 0
                temp += val_rig*D1_rig/(hx**2)
                triple_A.append((idx_A(i, j), idx_A(i, j), temp))
                
                # 2) 设置phi2
                temp = 0
                top, bot, lef, rig = -1, -1, -1, -1
                val_up, val_rig = 1, 1
                d = D2[idx_A(i, j)] * ext_dist

                D2_top = 2/(1/D2[idx_A(i, j)] + 1/D2[idx_A(i, j+1)]) if not outside(x[i], y[j]+hy) else D2[idx_A(i, j)]
                D2_bot = 2/(1/D2[idx_A(i, j)] + 1/D2[idx_A(i, j-1)]) if not outside(x[i], y[j]-hy) else D2[idx_A(i, j)]
                D2_lef = 2/(1/D2[idx_A(i, j)] + 1/D2[idx_A(i-1, j)]) if not outside(x[i]-hx, y[j]) else D2[idx_A(i, j)]
                D2_rig = 2/(1/D2[idx_A(i, j)] + 1/D2[idx_A(i+1, j)]) if not outside(x[i]+hx, y[j]) else D2[idx_A(i, j)]
                # ∂n=0
                if i==0: lef = 0
                if j==0: bot = 0
                # phi=0
                if outside(x[i]+hx, y[j]):
                    rig = 0
                    val_rig = hx/(hx/2+d)
                if outside(x[i], y[j]+hy):
                    top = 0
                    val_up = hy/(hy/2+d)
                # update triple
                if top != 0: triple_A.append((idx_B(i, j), idx_B(i, j+1), top*D2_top/(hy**2)))
                if bot != 0: triple_A.append((idx_B(i, j), idx_B(i, j-1), bot*D2_bot/(hy**2)))
                if lef != 0: triple_A.append((idx_B(i, j), idx_B(i-1, j), lef*D2_lef/(hx**2)))
                if rig != 0: triple_A.append((idx_B(i, j), idx_B(i+1, j), rig*D2_rig/(hx**2)))
                
                temp += val_up*D2_top/(hy**2)
                temp += D2_bot/(hy**2) if bot != 0 else 0
                temp += D2_lef/(hx**2) if lef != 0 else 0
                temp += val_rig*D2_rig/(hx**2)
                triple_A.append((idx_B(i, j), idx_B(i, j), temp))
        return triple_A

    triple_A = gen_sub()
    from scipy.sparse import csr_matrix
    row, col, val = [], [], []
    for r, c, v in triple_A: row.append(r); col.append(c); val.append(v)
    A = csr_matrix((val, (row, col)), shape=(nx*ny*2, nx*ny*2))
    return A

lap = gen_laplacian()

# -D1Δu + (∑a1+∑1→2) u = λ (ν∑f1u + ν∑f2v)
# -D2Δv + (∑a2) v = ∑1→2 u


# linearoperator
from scipy.sparse.linalg import LinearOperator
def mat_A(phi):
    res = lap*phi
    res[:nx*ny] += (a1+s12)*phi[:nx*ny]
    res[nx*ny:] += a2*phi[nx*ny:]-s12*phi[:nx*ny]
    return res
A = LinearOperator((nx*ny*2, nx*ny*2), mat_A)
def mat_B(phi):
    res = np.zeros(nx*ny*2)
    res[:nx*ny] = nf1*phi[:nx*ny]+nf2*phi[nx*ny:]
    return res
B = LinearOperator((nx*ny*2, nx*ny*2), mat_B)


# 源迭代
from scipy.sparse.linalg import gmres
def step(phi, keff):
    nxt_phi, _ = gmres(A, B*phi/keff)
    keff = keff * np.sum(B*nxt_phi) / (np.sum(B*phi))
    return nxt_phi, keff
phi = np.ones(nx*ny*2)
phi /= np.sqrt(np.sum(phi**2))
keff = 1
keff_his = []
for i in range(100):
    phi, keff = step(phi, keff)
    keff_his.append(keff)
    print(f"iter {i}, {keff=:.5f}")
print(f"finish: {keff=:.5f}")
np.save(f'{predir}/phi.npy', phi)
keff_his = np.array(keff_his)
np.save(f'{predir}/keff.npy', keff_his)



# 画图
xarr = [0, 10, 70, 90, 130, 150, 170]
yarr = xarr
def plot_phi(phi):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    phi1 = phi[:nx*ny]
    phi2 = phi[nx*ny:]
    phi1 = phi1.reshape(ny, nx)
    phi2 = phi2.reshape(ny, nx)
    phi1max = np.max(np.abs(phi1))
    im = ax[0].imshow(phi1, cmap='jet', vmin=0, vmax=phi1max, origin='lower', extent=(0, lx, 0, ly))
    fig.colorbar(im, ax=ax[0], orientation='vertical', fraction=0.046, pad=0.04)
    ax[0].set_title("$\phi_1$")
    ax[0].grid(True, which='both', linestyle='--', linewidth=1, color='white')
    ax[0].set_xticks(xarr)
    ax[0].set_yticks(yarr)
    phi2max = np.max(np.abs(phi2))
    im = ax[1].imshow(phi2, cmap='jet', vmin=0, vmax=phi2max, extent=(0, lx, 0, ly), origin='lower')
    fig.colorbar(im, ax=ax[1], orientation='vertical', fraction=0.046, pad=0.04)
    ax[1].set_title("$\phi_2$")
    ax[1].grid(True, which='both', linestyle='--', linewidth=1, color='white')
    ax[1].set_xticks(xarr)
    ax[1].set_yticks(yarr)
    plt.tight_layout()
    plt.savefig(f'{predir}/phi.png')
    plt.close()
plot_phi(phi)

def plot_keff():
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(keff_his, label='$k_{eff}$='+f"{keff_his[-1]:.5f}")
    ax[0].plot([0, len(keff_his)], [keff_ref, keff_ref], 'r--', label="$k_{eff,ref}=$"f'{keff_ref}')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('keff')
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].legend()
    ax[1].plot(np.abs(keff_his-keff_ref)/keff_ref)
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Relative error of keff')
    ax[1].set_yscale('log')
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'{predir}/keff.png')
    plt.close()
plot_keff()
