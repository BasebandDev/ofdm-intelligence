try:
    import sionna.phy
except ImportError as e:
    import os
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

sionna.phy.config.seed = 50
import matplotlib.pyplot as plt
import numpy as np
import torch

from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.utils import sim_ber, ebnodb2no
from sionna.phy.mapping import Mapper, QAMSource, BinarySource
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, \
                            LMMSEInterpolator, LinearDetector, KBestDetector, \
                            EPDetector, MMSEPICDetector, ResourceGridDemapper
from sionna.phy.channel import GenerateOFDMChannel, OFDMChannel, gen_single_sector_topology
from scipy.special import erfc
from scipy.interpolate import interp1d
# ── Simulation parameters ──────────────────────────────────────────────────────
NUM_BITS_PER_SYMBOL = 1
BATCH_SIZE          = 2000
EBN0_DB_MIN         = -3.0
EBN0_DB_MAX         = 5.0
NUM_OFDM_SYMBOLS    = 14
FFT_SIZE            = 12 * 4
SUBCARRIER_SPACING  = 30e3
NUM_RX_ANT          = 1
N_ITERS             = 200

# ── Pilot configurations to compare ───────────────────────────────────────────
PILOT_CONFIGS = {
    "2 pilots":  [2, 11],
    "4 pilots":  [1, 5, 9, 13],
    "6 pilots":  [0, 3, 5, 8, 10, 13],
    "Perfect CSI": [2, 11],   # same grid, but h_hat = h_freq
}

# ── Model ──────────────────────────────────────────────────────────────────────
# ── Model ──────────────────────────────────────────────────────────────────────
class OFDMRayleigh(sionna.phy.Block):
    """
    BPSK over Rayleigh fading + OFDM.

    Args:
        num_bits_per_symbol : bits per symbol (1 for BPSK/2-PAM)
        pilot_indices       : list of OFDM symbol indices carrying pilots
        perfect_csi         : if True, use true channel h_freq instead of h_hat
                              this gives a lower bound (no estimation error)

    Logs:
        mse_log : list of (ebno_db, mse) tuples collected during call()
                  used to compute channel estimation MSE per SNR point
                  without running a separate simulation loop
    """
    def __init__(self, num_bits_per_symbol, pilot_indices, perfect_csi=False):
        super().__init__()

        rg = ResourceGrid(
            num_ofdm_symbols=NUM_OFDM_SYMBOLS,
            fft_size=FFT_SIZE,
            subcarrier_spacing=SUBCARRIER_SPACING,
            num_tx=1,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=pilot_indices
        )
        self.rg           = rg
        self.n_bits       = num_bits_per_symbol
        self.block_length = rg.num_data_symbols * num_bits_per_symbol
        self.perfect_csi  = perfect_csi

        # mse_log stores (ebno_db, mse) per call() — no separate loop needed
        self.mse_log = []

        rx_tx = np.array([[1]])
        sm    = StreamManagement(rx_tx, num_streams_per_tx=1)

        self.rg_mapper     = ResourceGridMapper(rg)
        self.binary_source = sionna.phy.mapping.BinarySource()
        self.constellation = sionna.phy.mapping.Constellation("pam", num_bits_per_symbol)
        self.mapper        = sionna.phy.mapping.Mapper(constellation=self.constellation)
        self.ch_model      = sionna.phy.channel.RayleighBlockFading(
                                num_rx=1, num_rx_ant=NUM_RX_ANT,
                                num_tx=1, num_tx_ant=1)
        self.channel       = OFDMChannel(self.ch_model, rg, return_channel=True)
        self.estimator     = LSChannelEstimator(rg, interpolation_type='lin')
        self.equalizer     = LinearDetector(
                                equalizer="lmmse",
                                output="bit",
                                demapping_method="app",
                                resource_grid=rg,
                                stream_management=sm,
                                constellation_type="pam",
                                num_bits_per_symbol=num_bits_per_symbol)

    def reset_mse_log(self):
        """Clear MSE log — call before each simulate() to start fresh."""
        self.mse_log = []

    def call(self, batch_size, ebno_db):
        no = sionna.phy.utils.ebnodb2no(
            ebno_db, num_bits_per_symbol=self.n_bits, coderate=1.0)
        if isinstance(no, torch.Tensor):
            no = no.detach().clone().to(torch.float32)
        else:
            no = torch.tensor(no, dtype=torch.float32)

        bits = self.binary_source([batch_size, 1, 1, self.block_length])
        x    = self.mapper(bits)
        x_rg = self.rg_mapper(x)

        y_rg, h_freq   = self.channel(x_rg, no)
        h_hat, err_var = self.estimator(y_rg, no)

        # Log channel estimation MSE during this forward pass
        # MSE = E[|H_true - H_estimated|^2] averaged over batch + subcarriers
        # Stored as (ebno_db, mse) so we can group by SNR point later
        # This avoids running a separate simulation loop just for MSE
        mse     = ((h_freq - h_hat).abs() ** 2).mean().item()
        ebno_val = ebno_db.item() if isinstance(ebno_db, torch.Tensor) else float(ebno_db)
        self.mse_log.append((ebno_val, mse))

        # perfect_csi=True: bypass estimated h_hat with true h_freq
        # used as a lower bound — shows what perfect channel knowledge gives
        if self.perfect_csi:
            h_hat   = h_freq
            err_var = torch.zeros_like(h_hat)

        llr = self.equalizer(y_rg, h_hat, err_var, no)
        return bits, llr

def get_mse_per_snr(model):
    """
    Extracts mean channel MSE per SNR point from model.mse_log.
    mse_log is populated during simulate() — no extra run needed.
    Returns a pandas Series indexed by Eb/N0 (dB).
    """
    df = pd.DataFrame(model.mse_log, columns=["ebno_db", "mse"])
    return df.groupby("ebno_db")["mse"].mean()

# ── Instantiate models ─────────────────────────────────────────────────────────
models = {
    "2 pilots":    OFDMRayleigh(NUM_BITS_PER_SYMBOL, [2, 11],              perfect_csi=False),
    "4 pilots":    OFDMRayleigh(NUM_BITS_PER_SYMBOL, [1, 5, 9, 13],        perfect_csi=False),
    "6 pilots":    OFDMRayleigh(NUM_BITS_PER_SYMBOL, [0, 3, 5, 8, 10, 13], perfect_csi=False),
    "Perfect CSI": OFDMRayleigh(NUM_BITS_PER_SYMBOL, [2, 11],              perfect_csi=True),
}

print("Block lengths (data bits per transmission):")
print(f"{'Config':<12} {'Pilot syms':>10} {'Data bits':>10} {'Overhead %':>12}")
print("-" * 46)
base_bits = models["2 pilots"].block_length
configs_info = [
    ("2 pilots",    2,  models["2 pilots"].block_length),
    ("4 pilots",    4,  models["4 pilots"].block_length),
    ("6 pilots",    6,  models["6 pilots"].block_length),
    ("Perfect CSI", 2,  models["Perfect CSI"].block_length),
]
for name, npilots, nbits in configs_info:
    overhead = (base_bits - nbits) / base_bits * 100
    print(f"{name:<12} {npilots:>10} {nbits:>10} {overhead:>11.1f}%")

# ── Simulate all models ────────────────────────────────────────────────────────
# MSE is logged inside call() during simulate() — no separate loop needed
ebno_dbs  = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)
ber_plots = sionna.phy.utils.PlotBER("Pilot density comparison")

styles = {
    "2 pilots":    ("md-", "LS est — 2 pilots"),
    "4 pilots":    ("bs-", "LS est — 4 pilots"),
    "6 pilots":    ("g^-", "LS est — 6 pilots"),
    "Perfect CSI": ("k*-", "Perfect CSI"),
}

for name, model in models.items():
    model.reset_mse_log()   # clear before simulate()
    print(f"\nSimulating: {name} ...")
    ber_plots.simulate(
        model,
        ebno_dbs=ebno_dbs,
        batch_size=BATCH_SIZE,
        num_target_block_errors=100,
        legend=name,
        soft_estimates=True,
        max_mc_iter=N_ITERS,
        show_fig=False
    )

# ── Extract BER results ────────────────────────────────────────────────────────
results = {}
for i, name in enumerate(models.keys()):
    results[name] = {
        "snrs": ber_plots._snrs[i],
        "bers": ber_plots._bers[i],
    }

# ── Extract MSE results — free, already collected during simulate() ────────────
mse_2p = get_mse_per_snr(models["2 pilots"])
mse_4p = get_mse_per_snr(models["4 pilots"])
mse_6p = get_mse_per_snr(models["6 pilots"])

# ── Theoretical curves ─────────────────────────────────────────────────────────
def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2.0))

def rayleigh_ber(ebno_linear):
    return 0.5 * (1 - np.sqrt(ebno_linear / (1 + ebno_linear)))

ebno_theory  = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 200)
ebno_lin     = 10 ** (ebno_theory / 10.0)
ber_awgn     = qfunc(np.sqrt(2 * ebno_lin))
ber_rayleigh = rayleigh_ber(ebno_lin)

# ── Plot 1 — BER comparison ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for name, (style, label) in styles.items():
    ax.semilogy(results[name]["snrs"], results[name]["bers"],
                style, linewidth=2, markersize=5, label=label)

ax.semilogy(ebno_theory, ber_rayleigh, 'r--', linewidth=2,
            label="Rayleigh theoretical (perfect CSI)")
ax.semilogy(ebno_theory, ber_awgn,     'k--', linewidth=1.5,
            label="AWGN theoretical")

ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
ax.set_ylabel("BER", fontsize=12)
ax.set_title("Effect of Pilot Density on BER\nBPSK over Rayleigh + OFDM", fontsize=13)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, which="both")
ax.set_ylim([1e-5, 1])
ax.set_xlim([EBN0_DB_MIN, EBN0_DB_MAX])

plt.tight_layout()
plt.show()

# ── Plot 2 — Channel MSE comparison ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogy(mse_2p.index, mse_2p.values, 'md-', linewidth=2, markersize=5,
            label="Channel MSE — 2 pilots")
ax.semilogy(mse_4p.index, mse_4p.values, 'bs-', linewidth=2, markersize=5,
            label="Channel MSE — 4 pilots")
ax.semilogy(mse_6p.index, mse_6p.values, 'g^-', linewidth=2, markersize=5,
            label="Channel MSE — 6 pilots")

ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
ax.set_ylabel("Channel MSE  E[|H - Ĥ|²]", fontsize=12)
ax.set_title("Channel Estimation MSE vs SNR\nEffect of Pilot Density — Rayleigh Fading", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which="both")
ax.set_xlim([EBN0_DB_MIN, EBN0_DB_MAX])

plt.tight_layout()
plt.show()
