"""
=============================================================================
Forward Dunkle Model — Single Slope Solar Still (SYNTHETIC DATA GENERATION)
=============================================================================
Project : Data-Driven Discovery of the Dunkle Model

⚠️  SCOPE: SYNTHETIC DATA ONLY
    This script generates PUBLIC synthetic thermal data for testing and
    validation. Experimental data is RESTRICTED and available upon request.
    See DATA.md for full data access policy.

Description
-----------
Simulates the thermal behaviour of a single-slope passive solar still using
Dunkle's (1961) heat and mass transfer correlations. Synthetic climatic
forcing is generated to approximate Beirut, Lebanon conditions in May
(spring). The simulation produces time series of:
    - Water temperature            Tw   [°C]
    - Inner glass temperature      Tgi  [°C]
    - Convective HTC               hc   [W/m²K]
    - Evaporative HTC              he   [W/m²K]
    - Yield                        m_dot [kg/m²s]

The output dataset (clean + noisy) is saved as a CSV file for use in the
subsequent ML phases (SINDy, pySR, PINNs).

Output Location
---------------
Generated files are saved to: data/processed/
    - dunkle_clean.csv     (public, no noise)
    - dunkle_noisy.csv     (public, with Gaussian noise)

References
----------
Dunkle, R.V. (1961). Solar water distillation.
Tiwari, A.K. & Tiwari, G.N. (2006). Desalination, 195(1–3), 78–94.
Tiwari, G.N. & Sahota, L. (2017). Advanced Solar-Distillation Systems.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# =============================================================================
# 0. PHYSICAL CONSTANTS
# =============================================================================

SIGMA   = 5.67e-8       # Stefan-Boltzmann constant          [W/m²K⁴]
C_W     = 4186.0        # Specific heat of water             [J/kgK]
H_FG    = 2.26e6        # Latent heat of vaporisation        [J/kg]
RHO_W   = 1000.0        # Density of water                   [kg/m³]


# =============================================================================
# 1. SYSTEM PARAMETERS — Single-slope passive solar still
# =============================================================================

class SolarStillParams:
    """
    Physical and optical parameters for a single-slope solar still.
    Default values are representative of a standard lab-scale unit.
    """
    def __init__(self):
        # --- Geometry ---
        self.A       = 1.0          # Basin area                      [m²]
        self.d_w     = 0.02       # Water depth                     [m] # (depth is too low?)
        self.M_w     = RHO_W * self.d_w * self.A   # Mass of water   [kg]
        self.L_g     = 0.004        # Glass cover thickness          [m]
        self.K_g     = 0.20         # Thermal conducivity of glass   [W/m K]
        self.L_ins   = 0.005        # Insulation thickness           [m]
        self.K_ins   = 0.04         # Thermal conducivity of glass   [W/m K]


        # --- Optical properties ---
        self.alpha_w = 0.015         # Absorptivity of water           [-]
        self.tau_w   = 0.945         # Transmissivity of water         [-]
        self.r_w     = 0.04          # Reflectivity of water           [-]
        self.alpha_g = 0.05          # Absorptivity of glass           [-]
        self.tau_g   = 0.90          # Transmissivity of glass         [-]
        self.r_g     = 0.05          # Reflectivity of glass           [-]
        self.alpha_b = 0.90          # Absorptivity of basin           [-]

        # --- Emissivities ---
        self.eps_w   = 0.90         # Emissivity of water surface     [-]
        self.eps_g   = 0.90         # Emissivity of glass cover       [-]

        # Effective emissivity (water–glass cavity)
        self.eps_eff = 1.0 / (1.0/self.eps_w + 1.0/self.eps_g - 1.0)


# =============================================================================
# 2. SYNTHETIC CLIMATIC FORCING — Beirut, May (spring)
# =============================================================================

def generate_forcing(dt_s=60, days=1):
    """
    Generate synthetic 24-hour climatic forcing representative of
    Beirut, Lebanon in May.

    Beirut May climate (approximate):
        - Peak solar irradiance  : ~850 W/m²
        - Ambient temperature    : 22–30 °C (dawn to afternoon peak)
        - Wind speed             : 2–4 m/s (sea breeze pattern)

    Parameters
    ----------
    dt_s  : timestep in seconds (default 60 s = 1 min)
    days  : number of simulation days (default 1)

    Returns
    -------
    t     : time array [s]
    I     : solar irradiance [W/m²]
    T_a   : ambient temperature [°C]
    v     : wind speed [m/s]
    """
    # Total simulation time and time array
    t_end  = days * 24 * 3600          # seconds
    t      = np.arange(0, t_end, dt_s) # [s]
    t_hr   = t / 3600.0                # [hours] for convenience

    # --- Solar irradiance I(t) ---
    # Sinusoidal between sunrise (6h) and sunset (19h), zero otherwise
    # Beirut May: sunrise ~05:50, sunset ~19:30 → use 6 and 19 for simplicity
    t_rise = 6.0    # hours
    t_set  = 19.0   # hours
    I_max  = 850.0  # W/m²  (peak, May Beirut)

    I = np.where(
        (t_hr % 24 >= t_rise) & (t_hr % 24 <= t_set),
        I_max * np.sin(np.pi * (t_hr % 24 - t_rise) / (t_set - t_rise)),
        0.0
    )

    # --- Ambient temperature T_a(t) ---
    # Sinusoidal: min at dawn (~22 °C), max in early afternoon (~30 °C)
    # Phase offset so peak occurs ~14:00
    T_a_min  = 22.0   # °C  (dawn minimum, May Beirut)
    T_a_max  = 30.0   # °C  (afternoon maximum)
    T_a_mean = (T_a_min + T_a_max) / 2.0
    T_a_amp  = (T_a_max - T_a_min) / 2.0
    # Peak at 14:00 → phase = 14h in a 24h cycle
    T_a = T_a_mean + T_a_amp * np.sin(
        2 * np.pi * (t_hr % 24 - 8.0) / 24.0
    )

    # --- Wind speed v(t) ---
    # Beirut has a characteristic afternoon sea breeze
    # Mild in morning (~2 m/s), peaking in afternoon (~4 m/s)
    v_mean = 3.0    # m/s
    v_amp  = 1.0    # m/s
    v = v_mean + v_amp * np.sin(
        2 * np.pi * (t_hr % 24 - 6.0) / 24.0
    )
    v = np.clip(v, 0.5, None)   # physical lower bound

    return t, I, T_a, v


# =============================================================================
# 3. THERMOPHYSICAL FUNCTIONS
# =============================================================================

def vapor_pressure(T):
    """
    Saturation vapor pressure using Antoine-type expression.

    Parameters
    ----------
    T : temperature [°C]

    Returns
    -------
    P : vapor pressure [N/m²]
    """
    return np.exp(25.317 - 5144.0 / (T + 273.15))


def h_rad_water_glass(Tw, Tgi, eps_eff): #  THIS CORRESPONDS TO RADIATION COVER-SKY??
    """
    Radiative heat transfer coefficient between water surface and
    inner glass cover.

    Parameters
    ----------
    Tw      : water temperature        [°C]
    Tgi     : inner glass temperature  [°C]
    eps_eff : effective emissivity     [-]

    Returns
    -------
    h_rw : radiative HTC [W/m²K]
    """
    Tw_K  = Tw  + 273.15
    Tgi_K = Tgi + 273.15
    dT    = Tw - Tgi

    if abs(dT) < 0.01:
        # Linearised approximation when temperatures nearly equal
        T_avg = (Tw_K + Tgi_K) / 2.0
        return 4.0 * eps_eff * SIGMA * T_avg**3
    else:
        return eps_eff * SIGMA * (Tw_K**4 - Tgi_K**4) / dT


def h_rad_glass_sky(Tgi, Ta, eps_g):
    """
    Radiative heat transfer coefficient from outer glass to sky.

    Parameters
    ----------
    Tgi   : inner glass temperature  [°C]
    Ta    : ambient temperature      [°C]
    eps_g : glass emissivity         [-]

    Returns
    -------
    h_rg : radiative HTC [W/m²K]
    """
    Tgi_K   = Tgi + 273.15
    T_sky   = Ta - 6.0              # Sky temperature approximation [°C]
    T_sky_K = T_sky + 273.15
    dT      = Tgi - Ta

    if abs(dT) < 0.01:
        T_avg = (Tgi_K + T_sky_K) / 2.0
        return 4.0 * eps_g * SIGMA * T_avg**3
    else:
        return eps_g * SIGMA * (Tgi_K**4 - T_sky_K**4) / dT


# =============================================================================
# 4. DUNKLE'S INTERNAL HEAT TRANSFER COEFFICIENTS
# =============================================================================

def dunkle_coefficients(Tw, Tgi):
    """
    Compute Dunkle's (1961) convective and evaporative heat transfer
    coefficients between the water surface and inner glass cover.

    Equations (from proposal):
        hc = 0.884 * [(Tw - Tgi) + (Pw - Pgi)(Tw + 273.15)
                       / (268.9e3 - Pw)]^(1/3)             [Eq. 3]

        he = 16.273e-3 * hc * (Pw - Pgi) / (Tw - Tgi)     [Eq. 2]

    Parameters
    ----------
    Tw  : water temperature        [°C]
    Tgi : inner glass temperature  [°C]

    Returns
    -------
    h_c : convective HTC           [W/m²K]
    h_e : evaporative HTC          [W/m²K]
    """
    dT = Tw - Tgi

    # Guard: no heat transfer when temperatures are equal
    if abs(dT) < 0.01:
        return 0.0, 0.0

    Pw  = vapor_pressure(Tw)
    Pgi = vapor_pressure(Tgi)
    dP  = Pw - Pgi

    # Inner bracket of hc expression
    denom   = 268.9e3 - Pw
    bracket = dT + dP * (Tw + 273.15) / denom

    # Guard: bracket must be positive (physically required)
    if bracket <= 0.0:
        return 0.0, 0.0

    h_c = 0.884 * bracket**(1.0/3.0)
    h_e = 16.273e-3 * h_c * dP / dT

    # Guard: he must be non-negative
    h_e = max(h_e, 0.0)

    return h_c, h_e


# =============================================================================
# 5. TOTAL HEAT TRANSFER COEFFICIENTS
# =============================================================================

def h_1w_total(Tw, Tgi, eps_eff):
    """
    Total internal heat transfer coefficient from water to inner glass.
    h_1w = h_c + h_e + h_rw
    """
    h_c, h_e = dunkle_coefficients(Tw, Tgi)
    h_rw     = h_rad_water_glass(Tw, Tgi, eps_eff)
    return h_c + h_e + h_rw


def h_1g_total(Tgi, Ta, v, eps_g):
    """
    Total external heat transfer coefficient from outer glass to ambient.
    h_1g = h_cg (wind) + h_rg (radiation to sky)
    """
    h_cg = 2.8 + 3.0 * v                       # Wind convection [W/m²K]
    h_rg = h_rad_glass_sky(Tgi, Ta, eps_g)      # Radiation to sky
    return h_cg + h_rg


# =============================================================================
# 6. QUASI-STEADY GLASS TEMPERATURE
# =============================================================================

def compute_Tgi(Tw, Tgi_prev, Ta, v, I, params):
    """
    Solve the quasi-steady energy balance on the glass cover to get Tgi.

    Energy balance (steady, negligible glass thermal mass):
        alpha_g * I + h_1w * Tw + h_1g * Ta = (h_1w + h_1g) * Tgi

    Rearranged:
        Tgi = (alpha_g * I + h_1w * Tw + h_1g * Ta) / (h_1w + h_1g)

    h_1w and h_1g are evaluated using the PREVIOUS Tgi (lagged scheme).

    Parameters
    ----------
    Tw       : current water temperature       [°C]
    Tgi_prev : glass temperature at t-1        [°C]
    Ta       : ambient temperature             [°C]
    v        : wind speed                      [m/s]
    I        : solar irradiance                [W/m²]
    params   : SolarStillParams instance

    Returns
    -------
    Tgi_new : updated inner glass temperature  [°C]
    """
    H_1w = h_1w_total(Tw, Tgi_prev, params.eps_eff)
    H_1g = h_1g_total(Tgi_prev, Ta, v, params.eps_g)
    alpha_g_prime = params.alpha_g * (1 - params.r_g)
    U_tg = ((params.K_g / params.L_g) * H_1g)/((params.K_g / params.L_g) + H_1g)

    Tgi_new = (alpha_g_prime * I + H_1w * Tw + U_tg * Ta) / (H_1w + U_tg)
    return Tgi_new


# =============================================================================
# 7. ANALYTICAL ADVANCE OF WATER TEMPERATURE
# =============================================================================

def advance_Tw(Tw, Tgi, Ta, v, I, dt, params):
    """
    Advance water temperature by one timestep using the analytical
    solution of the first-order linear ODE:

        M_w C_w dTw/dt + (h_1w + U_b) Tw = alpha_w tau_g I
                                            + h_1w Tgi + U_b Ta

    Which in standard form is:
        dTw/dt + a * Tw = f(t)

    With analytical solution (treating a and f constant over dt):
        Tw(t+dt) = f/a + (Tw(t) - f/a) * exp(-a * dt)

    Parameters
    ----------
    Tw     : current water temperature  [°C]
    Tgi    : current glass temperature  [°C]
    Ta     : ambient temperature        [°C]
    v      : wind speed                 [m/s]
    I      : solar irradiance           [W/m²]
    dt     : timestep                   [s]
    params : SolarStillParams instance

    Returns
    -------
    Tw_new         : water temperature at t+dt                  [°C]
    a              : ODE decay coefficient                      [1/s]
    f              : ODE forcing term                           [°C/s]
    coefficient_solar  : alpha_eff / (M_w * C_w)               [1/(W·s)]
    coefficient_ambient: (U_t + U_b) / (M_w * C_w)             [1/s]
    """
    H_1w = h_1w_total(Tw, Tgi, params.eps_eff)
    H_1g = h_1g_total(Tgi, Ta, v, params.eps_g)
    alpha_w_prime = params.alpha_w * (1 - params.r_w) * (1 - params.r_g) * (1 - params.alpha_g)
    alpha_b_prime = params.alpha_b * (1 - params.alpha_g) * (1 - params.r_g) * (1 - params.alpha_w) * (1 - params.r_w)
    alpha_g_prime = params.alpha_g * (1 - params.r_g)
    h_w = 100 # Convective heat transfer coefficient from the basin to the water     [W/m²K]
    h_b = 1 / (params.L_ins/params.K_ins + 1 / (5.7 + 3.8 * v)) # Conductive and convective heat transfer coefficient from the basin to the ambient     [W/m²K]
    U_tg = ((params.K_g / params.L_g) * H_1g)/((params.K_g / params.L_g) + H_1g)
    U_t = (H_1w * U_tg) / (H_1w + U_tg)      # Top loss coefficient
    U_b = (h_w * h_b) / (h_w + h_b)          # Bottom loss coefficient
    alpha_eff = alpha_w_prime + (alpha_b_prime * h_w)/(h_w + h_b) + (alpha_g_prime * H_1w)/(H_1w + U_tg)



    # ODE coefficients
    a = (U_t + U_b) / (params.M_w * C_W)
    coefficient_solar = alpha_eff / (params.M_w * C_W)
    coefficient_ambient = (U_t + U_b) / (params.M_w * C_W)
    f_solar_term = coefficient_solar * I
    f_ambient_term = coefficient_ambient * Ta
    f = f_solar_term + f_ambient_term

    # Analytical solution
    Tw_new = (f / a) + (Tw - f / a) * np.exp(-a * dt)

    return Tw_new, a, f, coefficient_solar, coefficient_ambient


# =============================================================================
# 8. YIELD CALCULATION
# =============================================================================

def compute_yield(Tw, Tgi):
    """
    Compute instantaneous distillate yield.

        m_dot = h_e * (Tw - Tgi) / h_fg    [kg/m²s]

    Parameters
    ----------
    Tw  : water temperature        [°C]
    Tgi : inner glass temperature  [°C]

    Returns
    -------
    m_dot : yield rate [kg/m²s], clipped to >= 0
    """
    _, h_e = dunkle_coefficients(Tw, Tgi)
    m_dot  = h_e * (Tw - Tgi) / H_FG
    return max(m_dot, 0.0)


# =============================================================================
# 9. MAIN SIMULATION
# =============================================================================

def run_simulation(params, dt_s=60, days=1, noise_std_T=0.5,
                   noise_std_he=0.002, seed=42):
    """
    Run the forward Dunkle model simulation.

    Parameters
    ----------
    params       : SolarStillParams instance
    dt_s         : timestep [s]              (default 300 s)
    days         : simulation duration [days] (default 1)
    noise_std_T  : Gaussian noise std for temperatures [°C]
    noise_std_he : Gaussian noise std for he [W/m²K]
    seed         : random seed for reproducibility

    Returns
    -------
    df_clean : DataFrame with noise-free simulation results
    df_noisy : DataFrame with Gaussian noise added
    """
    np.random.seed(seed)

    # --- Generate forcing ---
    t, I_arr, Ta_arr, v_arr = generate_forcing(dt_s=dt_s, days=days)
    N = len(t)

    # --- Initialise storage arrays ---
    Tw_arr   = np.zeros(N)
    Tgi_arr  = np.zeros(N)
    hc_arr   = np.zeros(N)
    he_arr   = np.zeros(N)
    mdot_arr = np.zeros(N)
    a_arr    = np.zeros(N)
    f_arr    = np.zeros(N)
    coefficient_solar_arr   = np.zeros(N)
    coefficient_ambient_arr = np.zeros(N)

    # --- Initial conditions (pre-dawn, close to ambient minimum) ---
    Tw_arr[0]  = 23.0   # °C — slightly above dawn ambient (22 °C)
    Tgi_arr[0] = 22.5   # °C — just below water at t=0

    # Compute initial coefficients
    hc_arr[0], he_arr[0] = dunkle_coefficients(Tw_arr[0], Tgi_arr[0])
    mdot_arr[0]          = compute_yield(Tw_arr[0], Tgi_arr[0])
    _, a_arr[0], f_arr[0], coefficient_solar_arr[0], coefficient_ambient_arr[0] = advance_Tw(
        Tw_arr[0], Tgi_arr[0], Ta_arr[0], v_arr[0], I_arr[0], dt_s, params
    )

    # --- Time-stepping loop ---
    for i in range(1, N):
        Tw       = Tw_arr[i-1]
        Tgi_prev = Tgi_arr[i-1]
        Ta       = Ta_arr[i]
        I        = I_arr[i]
        v        = v_arr[i]

        # Step 1: Update Tgi quasi-steadily (lagged h coefficients)
        Tgi_new = compute_Tgi(Tw, Tgi_prev, Ta, v, I, params)

        # Step 2: Advance Tw analytically
        Tw_new, a_i, f_i, coefficient_solar_i, coefficient_ambient_i = advance_Tw(Tw, Tgi_new, Ta, v, I, dt_s, params)

        # Step 3: Compute Dunkle coefficients at new state
        h_c, h_e = dunkle_coefficients(Tw_new, Tgi_new)

        # Step 4: Compute yield
        m_dot = compute_yield(Tw_new, Tgi_new)

        # Step 5: Store
        Tgi_arr[i]  = Tgi_new
        Tw_arr[i]   = Tw_new
        hc_arr[i]   = h_c
        he_arr[i]   = h_e
        mdot_arr[i] = m_dot
        a_arr[i]    = a_i
        f_arr[i]    = f_i
        coefficient_solar_arr[i]   = coefficient_solar_i
        coefficient_ambient_arr[i] = coefficient_ambient_i

    # --- Assemble clean dataset ---
    df_clean = pd.DataFrame({
        "time_s"   : t,
        "time_hr"  : t / 3600.0,
        "I"        : I_arr,
        "T_a"      : Ta_arr,
        "v"        : v_arr,
        "Tw"       : Tw_arr,
        "Tgi"      : Tgi_arr,
        "Pw"       : vapor_pressure(Tw_arr),
        "Pgi"      : vapor_pressure(Tgi_arr),
        "hc"       : hc_arr,
        "he"       : he_arr,
        "m_dot"    : mdot_arr,
        "a_coeff"  : a_arr,
        "f_coeff"  : f_arr,
        "coefficient_solar"   : coefficient_solar_arr,
        "coefficient_ambient" : coefficient_ambient_arr,
    })

    # --- Add Gaussian noise for ML training dataset ---
    df_noisy = df_clean.copy()
    n        = len(df_clean)
    rng      = np.random.default_rng(seed)

    df_noisy["Tw"]  = df_clean["Tw"]  + rng.normal(0, noise_std_T,  n)
    df_noisy["Tgi"] = df_clean["Tgi"] + rng.normal(0, noise_std_T,  n)
    df_noisy["he"]  = np.clip(
                        df_clean["he"] + rng.normal(0, noise_std_he, n),
                        0, None)
    df_noisy["Pw"]  = vapor_pressure(df_noisy["Tw"].values)
    df_noisy["Pgi"] = vapor_pressure(df_noisy["Tgi"].values)

    return df_clean, df_noisy


# =============================================================================
# 10. PLOTTING
# =============================================================================

def plot_results(df_clean, df_noisy, save_path=None):
    """
    Generate a comprehensive 5-panel figure of simulation results.
    """
    t = df_clean["time_hr"].values

    fig = plt.figure(figsize=(14, 16))
    fig.suptitle(
        "Dunkle Forward Model — Single Slope Solar Still\n"
        "Synthetic Beirut May Forcing",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(5, 1, hspace=0.45)

    # --- Panel 1: Climatic forcing ---
    ax1 = fig.add_subplot(gs[0])
    ax1b = ax1.twinx()
    ax1.plot(t, df_clean["I"],   color="darkorange",  lw=1.8, label="I(t) [W/m²]")
    ax1b.plot(t, df_clean["T_a"], color="steelblue", lw=1.5,
              linestyle="--", label="Ta(t) [°C]")
    ax1.set_ylabel("Solar Irradiance [W/m²]", color="darkorange")
    ax1b.set_ylabel("Ambient Temp [°C]",      color="steelblue")
    ax1.set_title("Climatic Forcing")
    ax1.legend(loc="upper left",  fontsize=8)
    ax1b.legend(loc="upper right", fontsize=8)
    ax1.set_xlim(0, 24)
    ax1.set_xlabel("Time [hr]")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Temperatures ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t, df_clean["Tw"],  color="crimson",    lw=2.0,
             label="Tw clean")
    ax2.plot(t, df_clean["Tgi"], color="royalblue",  lw=2.0,
             label="Tgi clean")
    ax2.scatter(t[::6], df_noisy["Tw"].values[::6],  color="crimson",
                s=8, alpha=0.4, label="Tw noisy")
    ax2.scatter(t[::6], df_noisy["Tgi"].values[::6], color="royalblue",
                s=8, alpha=0.4, label="Tgi noisy")
    ax2.set_ylabel("Temperature [°C]")
    ax2.set_title("Water and Inner Glass Temperatures")
    ax2.legend(fontsize=8, ncol=2)
    ax2.set_xlim(0, 24)
    ax2.set_xlabel("Time [hr]")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Heat transfer coefficients ---
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(t, df_clean["hc"], color="seagreen",  lw=2.0, label="hc [W/m²K]")
    ax3.plot(t, df_clean["he"], color="purple",    lw=2.0, label="he [W/m²K]")
    ax3.set_ylabel("HTC [W/m²K]")
    ax3.set_title("Dunkle Heat Transfer Coefficients")
    ax3.legend(fontsize=8)
    ax3.set_xlim(0, 24)
    ax3.set_xlabel("Time [hr]")
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Yield rate ---
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(t, df_clean["m_dot"] * 3600,  # convert to kg/m²hr
             color="teal", lw=2.0, label="m_dot [kg/m²hr]")
    ax4.fill_between(t, df_clean["m_dot"] * 3600,
                     alpha=0.15, color="teal")
    ax4.set_ylabel("Yield [kg/m²hr]")
    ax4.set_title("Distillate Yield Rate")
    ax4.legend(fontsize=8)
    ax4.set_xlim(0, 24)
    ax4.set_xlabel("Time [hr]")
    ax4.grid(True, alpha=0.3)

    # --- Panel 5: Temperature difference ---
    ax5 = fig.add_subplot(gs[4])
    dT = df_clean["Tw"] - df_clean["Tgi"]
    ax5.plot(t, dT, color="darkorchid", lw=2.0,
             label="ΔT = Tw − Tgi [°C]")
    ax5.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax5.set_ylabel("ΔT [°C]")
    ax5.set_title("Temperature Difference (Driving Force for Evaporation)")
    ax5.legend(fontsize=8)
    ax5.set_xlim(0, 24)
    ax5.set_xlabel("Time [hr]")
    ax5.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


# =============================================================================
# 11. SUMMARY STATISTICS
# =============================================================================

def print_summary(df_clean):
    """Print key simulation statistics."""
    dt_hr    = df_clean["time_hr"].diff().median()
    daily_yield = (df_clean["m_dot"] * dt_hr * 3600).sum()  # kg/m²/day

    print("=" * 55)
    print("  SIMULATION SUMMARY — Dunkle Forward Model")
    print("=" * 55)
    print(f"  Timestep                 : {df_clean['time_s'].diff().median():.0f} s")
    print(f"  Peak solar irradiance    : {df_clean['I'].max():.1f} W/m²")
    print(f"  Ambient temp range       : {df_clean['T_a'].min():.1f} – "
          f"{df_clean['T_a'].max():.1f} °C")
    print(f"  Peak water temp (Tw)     : {df_clean['Tw'].max():.2f} °C")
    print(f"  Peak glass temp (Tgi)    : {df_clean['Tgi'].max():.2f} °C")
    print(f"  Peak ΔT = Tw − Tgi       : "
          f"{(df_clean['Tw'] - df_clean['Tgi']).max():.2f} °C")
    print(f"  Peak hc                  : {df_clean['hc'].max():.4f} W/m²K")
    print(f"  Peak he                  : {df_clean['he'].max():.4f} W/m²K")
    print(f"  Peak yield rate          : "
          f"{df_clean['m_dot'].max()*3600:.4f} kg/m²hr")
    print(f"  Daily yield              : {daily_yield:.4f} kg/m²/day")
    print("=" * 55)


# =============================================================================
# 12. ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # Output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(os.path.dirname(script_dir), "data", "processed")

    # Initialise parameters
    params = SolarStillParams()

    print("Running Dunkle forward model simulation...")
    print(f"  Water depth   : {params.d_w*100:.0f} cm")
    print(f"  Water mass    : {params.M_w:.1f} kg")
    print(f"  Basin area    : {params.A:.1f} m²")
    #print(f"  U_b           : {params.U_b:.1f} W/m²K")
    print()

    # Run simulation
    df_clean, df_noisy = run_simulation(
        params,
        dt_s        = 60,      # 5-minute timestep
        days        = 1,
        noise_std_T = 0.5,      # ±0.5 °C temperature noise
        noise_std_he= 0.002,    # small noise on he
        seed        = 42
    )

    # Print summary
    print_summary(df_clean)

    # Save datasets
    clean_path = os.path.join(out_dir, "dunkle_clean.csv")
    noisy_path = os.path.join(out_dir, "dunkle_noisy.csv")
    df_clean.to_csv(clean_path, index=False)
    df_noisy.to_csv(noisy_path, index=False)
    print(f"\nClean dataset saved : {clean_path}")
    print(f"Noisy dataset saved : {noisy_path}")

    # Plot and save figure
    fig_path = os.path.join(out_dir, "dunkle_simulation.png")
    fig = plot_results(df_clean, df_noisy, save_path=fig_path)
    plt.show()
