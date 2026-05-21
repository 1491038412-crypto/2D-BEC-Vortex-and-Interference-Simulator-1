"""
app.py —— Cold Atom Lab: 基于 Gross-Pitaevskii 方程的交互式模拟

运行方式：
    streamlit run app.py
"""

import base64
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from gpe_solver import GPESolver


# ---- 页面配置 ----
st.set_page_config(
    page_title="Cold Atom Lab",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_global_css():
    """注入全局样式，只修改视觉，不改模拟逻辑。"""
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5f5f7;
            --card: #ffffff;
            --text: #1d1d1f;
            --muted: #86868b;
            --line: #e0e0e0;
            --blue: #0071e3;
            --blue-hover: #0077ed;
            --shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
            --font-stack: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                          Helvetica, Arial, sans-serif;
        }

        html, body, [class*="css"], [data-testid="stAppViewContainer"],
        [data-testid="stSidebar"], [data-testid="stMarkdownContainer"],
        .stButton, .stSlider, .stRadio, .stSelectbox, .stCaption {
            font-family: var(--font-stack);
        }

        body, .stApp {
            background: var(--bg);
            color: var(--text);
        }

        [data-testid="stAppViewContainer"] {
            background: var(--bg);
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] > div:first-child {
            background: #ffffff;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 6rem;
        }

        #MainMenu,
        footer,
        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {
            display: none !important;
        }

        .sidebar-brand {
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
            margin-bottom: 1.2rem;
        }

        .sidebar-logo {
            font-size: 1.4rem;
            line-height: 1;
            color: var(--text);
        }

        .sidebar-title {
            font-size: 1.45rem;
            font-weight: 700;
            color: var(--text);
            letter-spacing: -0.02em;
        }

        .topbar {
            margin-bottom: 1.4rem;
        }

        .hero-title {
            font-size: 2.5rem;
            line-height: 1.1;
            font-weight: 700;
            color: var(--text);
            letter-spacing: -0.04em;
            margin: 0 0 1rem 0;
        }

        .topbar-rule {
            width: 100%;
            height: 1px;
            background: var(--line);
        }

        .mode-title {
            font-size: 1.15rem;
            line-height: 1.4;
            font-weight: 500;
            color: var(--muted);
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin: 0.85rem 0 1.5rem 0;
        }

        div[data-testid="stRadio"] label,
        div[data-testid="stSlider"] label {
            color: var(--text);
        }

        .stButton > button {
            width: 100%;
            min-height: 2.75rem;
            border-radius: 8px;
            border: none;
            background: var(--blue);
            color: #ffffff;
            font-weight: 600;
            box-shadow: none;
            transition: background-color 0.2s ease, transform 0.2s ease,
                        opacity 0.2s ease;
        }

        .stButton > button:hover {
            background: var(--blue-hover);
            color: #ffffff;
            transform: translateY(-1px);
        }

        .stButton > button:focus {
            outline: none;
            box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.18);
        }

        .stButton > button[kind="secondary"] {
            background: #f0f4f9;
            color: var(--text);
        }

        .plot-card {
            background: var(--card);
            border-radius: 12px;
            box-shadow: var(--shadow);
            padding: 1.1rem 1.1rem 0.8rem 1.1rem;
            min-height: 100%;
        }

        .card-title {
            font-size: 1.1rem;
            line-height: 1.4;
            font-weight: 500;
            color: var(--muted);
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-bottom: 0.9rem;
        }

        .plot-image {
            display: block;
            width: 100%;
            height: auto;
            border: none;
            background: transparent;
            border-radius: 10px;
        }

        .info-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(224, 224, 224, 0.8);
            border-radius: 12px;
            box-shadow: var(--shadow);
            padding: 1.2rem 1.25rem;
            color: var(--text);
        }

        .info-card p,
        .info-card li {
            color: var(--text);
            font-size: 0.98rem;
            line-height: 1.7;
            margin: 0;
        }

        .info-card ul {
            margin: 0.85rem 0 0 1.1rem;
            padding: 0;
        }

        .status-bar {
            position: fixed;
            left: 50%;
            bottom: 1.25rem;
            transform: translateX(-50%);
            z-index: 999;
            display: inline-flex;
            align-items: center;
            gap: 0.75rem;
            max-width: calc(100vw - 3rem);
            padding: 0.8rem 1.1rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid rgba(224, 224, 224, 0.9);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            color: var(--muted);
            font-size: 0.85rem;
            line-height: 1;
            white-space: nowrap;
        }

        .status-strong {
            color: var(--text);
            font-weight: 600;
        }

        .status-divider {
            color: #c7c7cc;
        }

        .sidebar-footer {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.65;
        }

        [data-testid="stAlert"] {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def figure_to_base64(fig):
    """把 matplotlib 图像转成 base64，便于放入自定义卡片。"""
    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=180,
        bbox_inches="tight",
        transparent=True,
    )
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    plt.close(fig)
    return encoded


def build_plot_image(data, extent, cmap, xlabel, ylabel, vmin=None, vmax=None):
    """生成图像卡片中的 matplotlib 图片。"""
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    image = ax.imshow(
        data.T,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )

    ax.set_xlabel(xlabel, color="#6e6e73")
    ax.set_ylabel(ylabel, color="#6e6e73")
    ax.tick_params(colors="#6e6e73", labelsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors="#6e6e73", labelsize=9)

    return figure_to_base64(fig)


inject_global_css()


# ---- 会话状态初始化 ----
DEFAULTS = {
    "running": False,
    "initialized": False,
    "time_elapsed": 0.0,
    "step_count": 0,
    "solver_params": None,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---- 缓存的求解器工厂 ----
@st.cache_resource
def get_solver(_N, _L, _dt, _g):
    """使用 @st.cache_resource 缓存 GPESolver 实例。"""
    return GPESolver(_N, _N, _L, _L, _dt, _g)


# ================================================================
# 侧边栏 —— 控制面板
# ================================================================
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-logo">⚛️</div>
        <div class="sidebar-title">Cold Atom Lab</div>
    </div>
    """,
    unsafe_allow_html=True,
)

mode = st.sidebar.radio(
    "Simulation Mode",
    ["Interference", "Vortex", "Free Expansion"],
    help="Choose an initial state and external potential.",
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

N = st.sidebar.slider(
    "Grid Size N",
    min_value=128,
    max_value=512,
    value=256,
    step=32,
    help="Higher values improve resolution but increase computational cost.",
    key="slider_N",
)
L = st.sidebar.slider(
    "Spatial Extent L",
    min_value=10.0,
    max_value=50.0,
    value=20.0,
    step=1.0,
    help="Sets the physical size of the simulation window.",
    key="slider_L",
)
g = st.sidebar.slider(
    "Nonlinearity g",
    min_value=0.0,
    max_value=1000.0,
    value=0.0,
    step=1.0,
    help="Controls the strength of the mean-field interaction.",
    key="slider_g",
)
dt = st.sidebar.slider(
    "Time Step dt",
    min_value=0.001,
    max_value=0.01,
    value=0.001,
    step=0.001,
    help="Smaller values improve stability at the cost of speed.",
    key="slider_dt",
)

st.session_state["_N"] = N
st.session_state["_L"] = L
st.session_state["_dt"] = dt
st.session_state["_g"] = g

current_params = (N, L, dt, g)
if st.session_state.solver_params != current_params:
    st.session_state.initialized = False
    st.session_state.running = False
    st.session_state.time_elapsed = 0.0
    st.session_state.step_count = 0
    st.session_state.solver_params = current_params

st.sidebar.markdown("<br>", unsafe_allow_html=True)

btn_col1, btn_col2, btn_col3 = st.sidebar.columns(3)

if "init_clicked" not in st.session_state:
    st.session_state.init_clicked = False
if "toggle_clicked" not in st.session_state:
    st.session_state.toggle_clicked = False
if "reset_clicked" not in st.session_state:
    st.session_state.reset_clicked = False

with btn_col1:
    if st.button("Initialize", use_container_width=True, key="btn_init"):
        st.session_state.init_clicked = True
        st.session_state.toggle_clicked = False
        st.session_state.reset_clicked = False

with btn_col2:
    button_label = "Pause" if st.session_state.running else "Start"
    if st.button(button_label, use_container_width=True, key="btn_toggle"):
        st.session_state.toggle_clicked = True
        st.session_state.init_clicked = False
        st.session_state.reset_clicked = False

with btn_col3:
    if st.button("Reset", use_container_width=True, key="btn_reset"):
        st.session_state.reset_clicked = True
        st.session_state.init_clicked = False
        st.session_state.toggle_clicked = False


# ================================================================
# 按钮逻辑处理
# ================================================================
if st.session_state.init_clicked:
    st.session_state.init_clicked = False
    try:
        solver = get_solver(N, L, dt, g)
    except MemoryError:
        st.sidebar.error("Insufficient memory. Please reduce Grid Size N and try again.")
    except Exception as exc:
        st.sidebar.error(f"Failed to create solver: {exc}")
    else:
        X, Y = solver.X, solver.Y

        if mode == "Interference":
            d = 3.5
            w = 1.5
            psi0 = (
                np.exp(-((X + d / 2) ** 2 + Y**2) / w**2)
                + np.exp(-((X - d / 2) ** 2 + Y**2) / w**2)
            )
            V = np.zeros_like(X)

        elif mode == "Vortex":
            w = 1.5
            gauss = np.exp(-(X**2 + Y**2) / w**2)
            phase = np.arctan2(Y, X)
            psi0 = gauss * np.exp(1j * phase)
            V = 0.5 * (X**2 + Y**2)

        else:  # Free Expansion
            w = 1.5
            psi0 = np.exp(-(X**2 + Y**2) / w**2)
            V = np.zeros_like(X)

        norm = np.sqrt(np.sum(np.abs(psi0) ** 2) * solver.dx * solver.dy)
        psi0 /= norm

        solver.set_initial_cond(psi0)
        solver.set_potential(V)

        st.session_state.initialized = True
        st.session_state.running = False
        st.session_state.time_elapsed = 0.0
        st.session_state.step_count = 0
        st.sidebar.success(f"✅ {mode} mode initialized (N={N}, L={L})")

if st.session_state.toggle_clicked:
    st.session_state.toggle_clicked = False
    if st.session_state.initialized:
        st.session_state.running = not st.session_state.running
    else:
        st.sidebar.warning("Please initialize the simulation first")

if st.session_state.reset_clicked:
    st.session_state.reset_clicked = False
    st.session_state.running = False
    st.session_state.initialized = False
    st.session_state.time_elapsed = 0.0
    st.session_state.step_count = 0
    get_solver.clear()
    st.sidebar.info("Reset complete. Please initialize again.")


# ================================================================
# 主界面
# ================================================================
st.markdown(
    """
    <div class="topbar">
        <div class="hero-title">Cold Atom Lab</div>
        <div class="topbar-rule"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="mode-title">{mode} Mode</div>',
    unsafe_allow_html=True,
)

solver = None
if st.session_state.initialized:
    try:
        solver = get_solver(N, L, dt, g)
    except MemoryError:
        st.error("Insufficient memory. Grid Size N is too large. Please reduce it and initialize again.")
        st.stop()
    except Exception as exc:
        st.error(f"Failed to create solver: {exc}")
        st.stop()

    extent = [-L / 2, L / 2, -L / 2, L / 2]
    density_image = build_plot_image(
        solver.get_density(),
        extent=extent,
        cmap="inferno",
        xlabel="x",
        ylabel="y",
    )
    phase_image = build_plot_image(
        solver.get_phase(),
        extent=extent,
        cmap="twilight",
        xlabel="x",
        ylabel="y",
        vmin=-np.pi,
        vmax=np.pi,
    )

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown(
            f"""
            <div class="plot-card">
                <div class="card-title">Density |ψ|²</div>
                <img class="plot-image" src="data:image/png;base64,{density_image}" alt="Density plot" />
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            f"""
            <div class="plot-card">
                <div class="card-title">Phase arg(ψ)</div>
                <img class="plot-image" src="data:image/png;base64,{phase_image}" alt="Phase plot" />
            </div>
            """,
            unsafe_allow_html=True,
        )

    run_status = "Running" if st.session_state.running else "Paused"
    st.markdown(
        f"""
        <div class="status-bar">
            <span class="status-strong">{run_status}</span>
            <span class="status-divider">|</span>
            <span>t = {st.session_state.time_elapsed:.4f}</span>
            <span class="status-divider">|</span>
            <span>Step = {st.session_state.step_count}</span>
            <span class="status-divider">|</span>
            <span>Mode = {mode}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.running:
        STEPS_PER_FRAME = 10
        for _ in range(STEPS_PER_FRAME):
            if mode == "Vortex":
                t_cur = st.session_state.time_elapsed
                omega = 2.0
                r0 = L * 0.2
                A_obs = 20.0
                w_obs = 1.5
                V_trap = 0.5 * (solver.X**2 + solver.Y**2)
                x0 = r0 * np.cos(omega * t_cur)
                y0 = r0 * np.sin(omega * t_cur)
                V_obstacle = A_obs * np.exp(
                    -((solver.X - x0) ** 2 + (solver.Y - y0) ** 2) / (2.0 * w_obs**2)
                )
                solver.V = V_trap + V_obstacle

            solver.step()
            st.session_state.time_elapsed += dt
            st.session_state.step_count += 1

        st.rerun()

else:
    st.markdown(
        """
        <div class="info-card">
            <p>← Set parameters and click <strong>Initialize</strong> to begin.</p>
            <ul>
                <li><strong>Interference</strong>: Two offset Gaussian wave packets overlap, showing interference fringes.</li>
                <li><strong>Vortex</strong>: A phase-imprinted condensate in a harmonic trap with a rotating obstacle.</li>
                <li><strong>Free Expansion</strong>: A Gaussian wave packet expanding freely in space.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.divider()
st.sidebar.markdown(
    """
    <div class="sidebar-footer">
        Based on a split-step Fourier solver for the Gross-Pitaevskii equation<br>
        <span>$i\\partial_t\\psi = -\\frac{1}{2}\\nabla^2\\psi + V\\psi + g|\\psi|^2\\psi$</span>
    </div>
    """,
    unsafe_allow_html=True,
)
