"""
Figure generation for the course report
"Reliability of Off-Line LED Drivers: Eliminating or Monitoring the Electrolytic Capacitor?"

Author : Christian E. Bamogo (HIT, ID 24SF04254)
Course : Introduction to LED Lighting - Prof. J. Marcos Alonso Alvarez

All figures are computed from two sources only:

  (a) The design equations of the course notes, Module 4 (off-line LED drivers)
      and Module 2 (LED electrical model), applied to the worked example given
      in Module 4, slides 29-30:

          Mains        : 230 V / 50 Hz   -> output ripple at 2*fL = 100 Hz
          LED lamp     : 200 V / 0.35 A, V_gamma = 170 V, R_gamma = 87 Ohm
          Lamp power   : Pg = 70 W
          Ripple       : I_hat = (Pg/Vo) / (4*pi*fL*R_gamma*Co)

  (b) IEEE Std 1789-2015 percent-modulation boundaries:

          %Mod = 100 * (Max - Min) / (Max + Min)

          low risk           : 0.025*f   for f <  90 Hz
                               0.080*f   for 90 <= f <= 1250 Hz
          no observable eff. : 0.010*f   for f <  90 Hz
                               0.0333*f  for 90 <= f <= 3000 Hz

Nothing is fitted, simulated or taken from external data. Every number printed
at the bottom of this script is a direct evaluation of the formulas above.

Usage:  python figures.py
Output: ./figures/*.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})
C1, C2, C3, C4 = "#1f4e79", "#c0392b", "#27ae60", "#e67e22"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
save = lambda fig, name: (fig.savefig(os.path.join(OUT, name)), plt.close(fig))

# --------------------------------------------------------------------------
# Case study parameters - course notes, Module 4, slides 29-30
# --------------------------------------------------------------------------
Vg_rms = 230.0      # mains RMS voltage                      [V]
fL     = 50.0       # line frequency                         [Hz]
Pg     = 70.0       # lamp power                             [W]
Vo     = 200.0      # LED string voltage                     [V]
Idc    = 0.35       # LED average current                    [A]
Vgam   = 170.0      # LED threshold voltage                  [V]
Rgam   = 87.0       # LED dynamic resistance                 [Ohm]
Rg     = 755.7      # equivalent input resistance (Vg^2/Pg)  [Ohm]

fR = 2 * fL         # output ripple frequency = 100 Hz


def ripple_amplitude(Co):
    """LED current ripple amplitude [A] for a bus capacitance Co [F]."""
    return (Pg / Vo) / (4 * np.pi * fL * Rgam * Co)


def capacitance_for(I_hat):
    """Bus capacitance [F] required for a target ripple amplitude I_hat [A]."""
    return (Pg / Vo) / (4 * np.pi * fL * Rgam * I_hat)


def percent_modulation(I_hat):
    """IEEE 1789 percent modulation for a symmetric ripple about Idc."""
    return 100.0 * (2 * I_hat) / (2 * Idc)


def ieee1789_limits(f):
    """Return (no_observable_effect, low_risk) percent-modulation limits."""
    noel = np.where(f < 90, 0.01 * f, np.where(f <= 3000, 0.0333 * f, np.inf))
    lowr = np.where(f < 90, 0.025 * f, np.where(f <= 1250, 0.080 * f, np.inf))
    return noel, lowr


# ==========================================================================
# Fig. 1 - Pulsating input power under unity power factor
# ==========================================================================
def fig1():
    t = np.linspace(0, 0.04, 2000)
    vg = Vg_rms * np.sqrt(2) * np.sin(2 * np.pi * fL * t)
    ig = (Vg_rms * np.sqrt(2) / Rg) * np.sin(2 * np.pi * fL * t)
    pg = vg * ig

    fig, ax = plt.subplots(2, 1, figsize=(6.2, 4.4), sharex=True)
    ax[0].plot(t * 1e3, vg, color=C1, lw=1.4)
    axb = ax[0].twinx(); axb.plot(t * 1e3, ig, color=C4, lw=1.4)
    ax[0].set_ylabel("Line voltage (V)", color=C1)
    axb.set_ylabel("Line current (A)", color=C4)
    axb.grid(False); axb.spines["top"].set_visible(False)
    ax[0].set_title("Unity-power-factor input: voltage and current in phase", fontsize=9.5)

    ax[1].plot(t * 1e3, pg, color=C2, lw=1.5, label=r"$p_g(t)=v_g i_g$")
    ax[1].axhline(Pg, color=C1, ls="--", lw=1.3, label=r"$P_g$ = 70 W (average)")
    ax[1].fill_between(t * 1e3, Pg, pg, where=(pg > Pg), color=C2, alpha=0.15)
    ax[1].fill_between(t * 1e3, Pg, pg, where=(pg < Pg), color=C1, alpha=0.15)
    ax[1].set_xlabel("Time (ms)"); ax[1].set_ylabel("Input power (W)")
    ax[1].legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax[1].set_title(r"Instantaneous power pulsates at $2f_L$ = 100 Hz "
                    r"$\rightarrow$ energy buffer required", fontsize=9.5)
    fig.tight_layout(); save(fig, "fig1_pulsating_power.png")


# ==========================================================================
# Fig. 2 - Required bus capacitance vs LED current ripple
# ==========================================================================
def fig2():
    Co = np.linspace(20e-6, 800e-6, 800)
    ripple_pct = 100 * (2 * ripple_amplitude(Co)) / Idc   # peak-to-peak, % of Idc

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(Co * 1e6, ripple_pct, color=C1, lw=2)
    ax.axhline(30, color=C2, ls="--", lw=1.2)
    ax.plot(130, 30, "o", color=C2, ms=7, zorder=5)
    ax.annotate("Course design point\n130 µF → 30 % ripple", xy=(130, 30), xytext=(260, 62),
                fontsize=8.5, color=C2, arrowprops=dict(arrowstyle="->", color=C2, lw=1.1))
    ax.set_xlabel("Bus capacitance $C_o$ (µF)")
    ax.set_ylabel("LED current ripple, peak-to-peak (% of $I_{dc}$)")
    ax.set_title(r"$\hat{I}_{LED}=\dfrac{P_g/V_o}{4\pi f_L R_\gamma C_o}$"
                 r"   —  ripple is inversely proportional to $C_o$", fontsize=9.5)
    ax.set_ylim(0, 120); ax.set_xlim(0, 800)
    fig.tight_layout(); save(fig, "fig2_capacitance_vs_ripple.png")


# ==========================================================================
# Fig. 3 - Electrolytic capacitor wear-out
#          Lifetime model: L = L0 * (V/V0)^-n * 2^((T0-T)/10)
#          (Wang & Blaabjerg, IEEE Trans. Ind. Appl. 50(5), 2014)
# ==========================================================================
def fig3():
    L0, T0, n = 5000.0, 105.0, 4.0     # n typically 3-5 for aluminium electrolytics
    TA = np.linspace(40, 105, 400)

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.9))
    a = ax[0]
    for vr, col in [(1.0, C2), (0.8, C4), (0.6, C3)]:
        a.semilogy(TA, L0 * vr ** (-n) * 2 ** ((T0 - TA) / 10.0), lw=1.9, color=col,
                   label=fr"$V/V_0$ = {vr:.1f}")
    a.axhline(1e5, color="#34495e", ls="--", lw=1.3)
    a.text(41, 1.2e5, "LED lamp target: 100 000 h", color="#34495e", fontsize=8.2)
    a.set_xlabel("Capacitor core temperature $T_A$ (°C)")
    a.set_ylabel("Useful life (h, log scale)")
    a.set_title(r"$L = L_0\,(V/V_0)^{-n}\,2^{(T_0-T)/10}$,   $n$ = 4 (Al-cap)", fontsize=9.5)
    a.legend(fontsize=8, title="Voltage derating", title_fontsize=8, loc="upper right")
    a.grid(alpha=0.25, which="both")

    b = ax[1]
    vr = np.linspace(0.5, 1.0, 300)
    for T, col in [(85, C2), (70, C4), (55, C3)]:
        b.semilogy(vr * 100, L0 * vr ** (-n) * 2 ** ((T0 - T) / 10.0), lw=1.9, color=col,
                   label=f"$T_A$ = {T} °C")
    b.axhline(1e5, color="#34495e", ls="--", lw=1.3)
    b.set_xlabel("Applied voltage as % of rated $V_0$")
    b.set_ylabel("Useful life (h, log scale)")
    b.set_title("Voltage derating is the second lever", fontsize=9.5)
    b.legend(fontsize=8, loc="upper right"); b.grid(alpha=0.25, which="both")
    fig.suptitle("Electrolytic capacitor wear-out: two design levers, "
                 "both bounded by the 100 000 h target", fontsize=10.5, y=1.03)
    fig.tight_layout(); save(fig, "fig3_capacitor_lifetime.png")


# ==========================================================================
# Fig. 4 - Electrolytic vs film capacitor (qualitative, course notes Module 4)
# ==========================================================================
def fig4():
    cats = ["Lifetime\n(h)", "Volume", "Cost", "ESR / losses"]
    elec = [1, 1.0, 1.0, 1.0]          # normalised to the electrolytic case
    film = [10, 3.2, 4.0, 0.25]

    x = np.arange(len(cats)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.bar(x - w / 2, elec, w, color=C2, label="Electrolytic", alpha=0.85)
    ax.bar(x + w / 2, film, w, color=C1, label="Film (polypropylene)", alpha=0.85)
    for i, (e, f_) in enumerate(zip(elec, film)):
        ax.text(i - w / 2, e + 0.12, f"{e:g}×", ha="center", fontsize=8, color=C2)
        ax.text(i + w / 2, f_ + 0.12, f"{f_:g}×", ha="center", fontsize=8, color=C1)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("Relative to electrolytic (×)")
    ax.set_title("The reliability trade-off: film wins on life and losses,\n"
                 "loses on volume and cost — hence the drive to reduce $C$", fontsize=9.5)
    ax.legend(fontsize=8.5); ax.set_ylim(0, 11.5)
    fig.tight_layout(); save(fig, "fig4_electrolytic_vs_film.png")


# ==========================================================================
# Fig. 5 - IEEE 1789-2015 limits vs required bus capacitance   [KEY FIGURE]
#          Reference design points from Spode et al., IEEE TIA 59(4), 2023
# ==========================================================================
def fig5():
    f = np.logspace(np.log10(10), np.log10(5000), 2000)
    noel, lowr = ieee1789_limits(f)
    noel = np.where(np.isinf(noel), 1e3, noel)
    lowr = np.where(np.isinf(lowr), 1e3, lowr)

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0))

    a = ax[0]
    a.fill_between(f, 0.1, noel, color=C3, alpha=0.22, label="No observable effect")
    a.fill_between(f, noel, lowr, color=C4, alpha=0.20, label="Low risk")
    a.fill_between(f, lowr, 200, color=C2, alpha=0.12, label="Not recommended")
    a.plot(f, noel, color=C3, lw=1.4); a.plot(f, lowr, color=C4, lw=1.4)
    a.axvline(1250, color=C4, ls=":", lw=1.0); a.axvline(3000, color=C3, ls=":", lw=1.0)
    a.text(1290, 0.62, "1250 Hz", color=C4, fontsize=7.2, rotation=90, va="bottom")
    a.text(3100, 0.62, "3000 Hz", color=C3, fontsize=7.2, rotation=90, va="bottom")

    a.plot(100, percent_modulation(ripple_amplitude(130e-6)), "o", color=C2,
           ms=8, mec="white", mew=1.2, zorder=6)
    a.plot(120, 8.05, "D", color=C1, ms=7.5, mec="white", mew=1.2, zorder=6)
    a.plot(120, 27.55, "X", color=C2, ms=8.5, mec="white", mew=1.2, zorder=6)
    a.annotate("Course example\n130 µF, 100 Hz\n14.1 %", xy=(100, 14.1), xytext=(21, 33),
               fontsize=7.6, color=C2, arrowprops=dict(arrowstyle="->", color=C2, lw=0.9))
    a.annotate("Spode/Alonso 2023\nproposed: 8.05 %", xy=(120, 8.05), xytext=(300, 3.0),
               fontsize=7.6, color=C1, arrowprops=dict(arrowstyle="->", color=C1, lw=0.9))
    a.annotate("conventional: 27.55 %", xy=(120, 27.55), xytext=(330, 46),
               fontsize=7.6, color=C2, arrowprops=dict(arrowstyle="->", color=C2, lw=0.9))
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(10, 5000); a.set_ylim(0.5, 150)
    a.set_xlabel("Modulation frequency (Hz)"); a.set_ylabel("Percent modulation (%)")
    a.set_title("IEEE 1789-2015 risk regions", fontsize=9.5)
    a.legend(fontsize=7.3, loc="lower right", framealpha=0.95)

    b = ax[1]
    Co = np.linspace(60e-6, 800e-6, 700)
    b.plot(Co * 1e6, percent_modulation(ripple_amplitude(Co)), color=C1, lw=2.2)
    b.axhline(8.0, color=C4, ls="--", lw=1.4); b.axhline(3.33, color=C3, ls="--", lw=1.4)
    b.text(795, 8.8, "Low-risk limit @ 100 Hz (8 %)", color=C4, fontsize=8, ha="right")
    b.text(795, 3.75, "No-effect limit @ 100 Hz (3.33 %)", color=C3, fontsize=8, ha="right")
    b.fill_between(Co * 1e6, 8.0, 40, color=C2, alpha=0.09)
    for Cv, col, mk in [(130e-6, C2, "o"), (229e-6, C4, "s"), (549e-6, C3, "^")]:
        b.plot(Cv * 1e6, percent_modulation(ripple_amplitude(Cv)), mk, color=col,
               ms=8, mec="white", mew=1.2, zorder=6)
    b.annotate("130 µF → 14.1 %\n(course design, above limit)", xy=(130, 14.1),
               xytext=(255, 23), fontsize=8.1, color=C2,
               arrowprops=dict(arrowstyle="->", color=C2, lw=1.05))
    b.annotate("229 µF\n(1.8×)", xy=(229, 8.0), xytext=(300, 12.4), fontsize=7.8, color=C4,
               arrowprops=dict(arrowstyle="->", color=C4, lw=0.95))
    b.annotate("549 µF\n(4.2×)", xy=(549, 3.33), xytext=(590, 7.0), fontsize=7.8, color=C3,
               arrowprops=dict(arrowstyle="->", color=C3, lw=0.95))
    b.set_xlabel("Bus capacitance $C_o$ (µF)")
    b.set_ylabel("Percent modulation at 100 Hz (%)")
    b.set_title("Flicker sets a lower bound on $C_o$", fontsize=9.5)
    b.set_ylim(0, 32); b.set_xlim(60, 800)
    fig.suptitle("Health-based flicker limits bound how far the bulk capacitor can be shrunk",
                 fontsize=10.5, y=1.02)
    fig.tight_layout(); save(fig, "fig5_ieee1789_vs_capacitance.png")


# ==========================================================================
# Fig. 6 - Conceptual diagram: eliminate vs monitor
# ==========================================================================
def fig6():
    fig, ax = plt.subplots(figsize=(7.6, 3.9)); ax.axis("off")

    def box(x, y, w, h, txt, fc, ec, fs=8.4, weight="normal"):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.4, alpha=0.92))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                color="#12232e", weight=weight, linespacing=1.45)

    box(0.30, 0.78, 0.40, 0.16,
        "Reliability bottleneck\nElectrolytic capacitor: ~10 000 h\nvs LED lamp: ~100 000 h",
        "#fdece9", C2, 9, "bold")
    ax.annotate("", xy=(0.28, 0.64), xytext=(0.44, 0.78),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=C1))
    ax.annotate("", xy=(0.72, 0.64), xytext=(0.56, 0.78),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=C3))

    box(0.03, 0.30, 0.44, 0.34,
        "STRATEGY A — ELIMINATE\n\n• Reduce required $C$ (ripple cancellation,\n   third-harmonic injection)\n"
        "• Inductive / third-port energy storage\n• Replace with film capacitors\n"
        "• Integrated single-stage topologies (IDBB)", "#eaf1f8", C1, 8.2)
    box(0.53, 0.30, 0.44, 0.34,
        "STRATEGY B — MONITOR\n\n• Track health indicators: $C$, ESR\n• Data-driven condition monitoring\n"
        "• Predictive replacement, not scheduled\n• Accepts the capacitor, removes the surprise",
        "#eaf7ee", C3, 8.2)
    ax.annotate("", xy=(0.53, 0.47), xytext=(0.47, 0.47),
                arrowprops=dict(arrowstyle="<->", lw=1.5, color="#7f8c8d"))
    box(0.20, 0.03, 0.60, 0.17,
        "COMPLEMENTARY, NOT COMPETING\n"
        "Topology reduces the exposure  •  Monitoring removes the remaining uncertainty\n"
        "Flicker limits (IEEE 1789) bound how far Strategy A can go",
        "#f4f6f7", "#7f8c8d", 8.4, "bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); save(fig, "fig6_eliminate_vs_monitor.png")


# ==========================================================================
if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6):
        fn()
        print(f"  {fn.__name__} written")

    print("\nKey numerical results reported in the paper")
    print("-" * 52)
    mod130 = percent_modulation(ripple_amplitude(130e-6))
    print(f"  Course design, Co = 130 uF   -> %Mod = {mod130:5.1f} %  at 100 Hz")
    for limit, label in ((0.080 * 100, "low risk"), (0.0333 * 100, "no observable effect")):
        C = capacitance_for(limit / 100 * Idc)
        print(f"  IEEE 1789 {label:22s} ({limit:5.2f} %) -> Co = {C*1e6:5.0f} uF"
              f"  ({C/130e-6:.1f}x)")
