"""
gpe_solver.py —— 二维含时 Gross-Pitaevskii 方程的分步傅里叶法求解器

求解的无纲化 GP 方程：
    i ∂ψ/∂t = -0.5 * ∇²ψ + V * ψ + g * |ψ|² * ψ

数值方法：对称分步傅里叶法 (Symmetric Split-Step Fourier Method)
    每一步演化顺序：动能半步(傅里叶空间) → 势能+非线性整步(实空间) → 动能半步(傅里叶空间)
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq


class GPESolver:
    """二维 Gross-Pitaevskii 方程数值求解器

    使用对称分步傅里叶法 (Strang splitting) 演化波函数。
    演化算符 U(dt) = exp(-iK*dt/2) * exp(-i(V+g|ψ|²)*dt) * exp(-iK*dt/2)
    """

    def __init__(self, Nx, Ny, Lx, Ly, dt, g):
        """
        Parameters
        ----------
        Nx, Ny : int
            x, y 方向的网格点数
        Lx, Ly : float
            x, y 方向的空间范围，坐标区间为 [-Lx/2, Lx/2), [-Ly/2, Ly/2)
        dt : float
            时间步长
        g : float
            非线性相互作用强度
        """
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
        self.g = g

        # 空间网格
        self.dx = Lx / Nx
        self.dy = Ly / Ny

        # 实空间坐标（用于设置初始条件和外势）
        self.x = (np.arange(Nx) - Nx / 2.0) * self.dx
        self.y = (np.arange(Ny) - Ny / 2.0) * self.dy
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # 傅里叶空间波数（圆频率角波数 k = 2π * frequency）
        # fftfreq 返回标准化频率，乘以 2π / dx 得到物理波数
        self.kx = 2.0 * np.pi * fftfreq(Nx, d=self.dx)
        self.ky = 2.0 * np.pi * fftfreq(Ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')

        # 动能项传播子（傅里叶空间）：exp(-i * (kx^2 + ky^2) * dt / 4)
        # 注意：半步演化因子为 exp(-i * K * dt / 2)，对应 dt → dt/2
        #       其中 K = (kx² + ky²) / 2，所以：
        #       exp(-i * (kx² + ky²) / 2 * dt / 2) = exp(-i * (kx²+ky²) * dt / 4)
        self._kinetic_prop_half = np.exp(
            -1j * (self.KX**2 + self.KY**2) * dt / 4.0
        )

        # 存储外势 V(x,y)，默认为零
        self.V = np.zeros((Nx, Ny))

        # 波函数 psi
        self.psi = np.zeros((Nx, Ny), dtype=complex)

    def set_initial_cond(self, psi0):
        """设置初始波函数

        Parameters
        ----------
        psi0 : ndarray of shape (Nx, Ny), dtype complex
            初始时刻的波函数
        """
        if psi0.shape != (self.Nx, self.Ny):
            raise ValueError(
                f"初始波函数形状必须为 ({self.Nx}, {self.Ny})，"
                f"当前为 {psi0.shape}"
            )
        self.psi = psi0.astype(complex).copy()

    def set_potential(self, V):
        """设置外部势 V(x, y)

        Parameters
        ----------
        V : ndarray of shape (Nx, Ny)
            外部势在空间网格上的取值
        """
        if V.shape != (self.Nx, self.Ny):
            raise ValueError(
                f"外势形状必须为 ({self.Nx}, {self.Ny})，当前为 {V.shape}"
            )
        self.V = V.copy()

    def step(self):
        """执行一步时间演化，采用对称分步傅里叶法

        演化顺序：
            1. 动能半步演化（傅里叶空间）
            2. 势能 + 非线性项整步演化（实空间）
            3. 动能半步演化（傅里叶空间）

        Returns
        -------
        psi : ndarray of shape (Nx, Ny), dtype complex
            演化一步后的波函数
        """
        # ---- 第一步：动能项半步演化（傅里叶空间） ----
        psi_k = fft2(self.psi)
        psi_k *= self._kinetic_prop_half  # exp(-i * k² * dt / 4)
        psi = ifft2(psi_k)

        # ---- 第二步：势能 + 非线性项整步演化（实空间） ----
        # 非线性项传播子 exp(-i * (V + g*|ψ|²) * dt)
        density = np.abs(psi)**2
        psi *= np.exp(-1j * (self.V + self.g * density) * self.dt)

        # ---- 第三步：动能项半步演化（傅里叶空间） ----
        psi_k = fft2(psi)
        psi_k *= self._kinetic_prop_half  # exp(-i * k² * dt / 4)
        self.psi = ifft2(psi_k)

        return self.psi

    def get_density(self):
        """返回粒子数密度 |ψ|²

        Returns
        -------
        density : ndarray of shape (Nx, Ny)
        """
        return np.abs(self.psi)**2

    def get_phase(self):
        """返回波函数的相位 angle(ψ)

        Returns
        -------
        phase : ndarray of shape (Nx, Ny)
            取值范围 (-π, π]
        """
        return np.angle(self.psi)


if __name__ == "__main__":
    # ——— 测试：高斯波包在自由空间中的扩散 ———
    import matplotlib.pyplot as plt

    # 网格与时间参数
    Nx, Ny = 256, 256
    Lx, Ly = 20.0, 20.0
    dt = 0.001
    g = 0.0  # 自由扩散，无非线性

    solver = GPESolver(Nx, Ny, Lx, Ly, dt, g)

    # 高斯波包：ψ(x, y, t=0) = exp(-(x² + y²) / (2σ²))
    sigma = 1.5
    psi0 = np.exp(-(solver.X**2 + solver.Y**2) / (2.0 * sigma**2))
    # 归一化到粒子数为 1
    psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * solver.dx * solver.dy)

    solver.set_initial_cond(psi0)
    solver.set_potential(np.zeros((Nx, Ny)))

    # 演化参数
    num_steps = 500            # 每 500 步保存一帧
    num_frames = 5             # 共 5 帧
    save_interval = num_steps

    densities = []
    times = []

    for frame in range(num_frames):
        for _ in range(save_interval):
            solver.step()
        t = (frame + 1) * save_interval * dt
        densities.append(solver.get_density())
        times.append(t)

    # 绘图
    fig, axes = plt.subplots(1, num_frames, figsize=(5 * num_frames, 4.5))
    for idx, (dens, t) in enumerate(zip(densities, times)):
        im = axes[idx].imshow(
            dens.T, origin='lower', extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
            cmap='inferno'
        )
        axes[idx].set_title(f"t = {t:.2f}")
        axes[idx].set_xlabel("x")
        axes[idx].set_ylabel("y")
        plt.colorbar(im, ax=axes[idx])

    fig.suptitle("Gaussian wave packet free expansion (Split-Step Fourier)", fontsize=14)
    plt.tight_layout()
    plt.show()
