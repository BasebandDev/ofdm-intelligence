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
class OFDMRayleigh(sionna.phy.Block):
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
        self.rg              = rg
        self.n_bits          = num_bits_per_symbol
        self.block_length    = rg.num_data_symbols * num_bits_per_symbol
        self.perfect_csi     = perfect_csi

        rx_tx   = np.array([[1]])
        sm      = StreamManagement(rx_tx, num_streams_per_tx=1)

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

        if self.perfect_csi:
            h_hat   = h_freq
            err_var = torch.zeros_like(h_hat)

        llr = self.equalizer(y_rg, h_hat, err_var, no)
        return bits, llr

# ── Instantiate models ─────────────────────────────────────────────────────────
models = {
    "2 pilots":    OFDMRayleigh(NUM_BITS_PER_SYMBOL, [2, 11],             perfect_csi=False),
    "4 pilots":    OFDMRayleigh(NUM_BITS_PER_SYMBOL, [1, 5, 9, 13],       perfect_csi=False),
    "6 pilots":    OFDMRayleigh(NUM_BITS_PER_SYMBOL, [0, 3, 5, 8, 10, 13],perfect_csi=False),
    "Perfect CSI": OFDMRayleigh(NUM_BITS_PER_SYMBOL, [2, 11],             perfect_csi=True),
}

print("Block lengths (data bits per transmission):")
for name, m in models.items():
    print(f"  {name:12s} → {m.block_length} bits")

# ── Simulate all models ────────────────────────────────────────────────────────
ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)

ber_plots = sionna.phy.utils.PlotBER("Pilot density comparison")

styles = {
    "2 pilots":    ("md-", "LS est — 2 pilots"),
    "4 pilots":    ("bs-", "LS est — 4 pilots"),
    "6 pilots":    ("g^-", "LS est — 6 pilots"),
    "Perfect CSI": ("k*-", "Perfect CSI"),
}

for name, model in models.items():
    print(f"\nSimulating: {name}")
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

# ── Extract results ────────────────────────────────────────────────────────────
results = {}
for i, name in enumerate(models.keys()):
    results[name] = {
        "snrs": ber_plots._snrs[i],
        "bers": ber_plots._bers[i],
    }

# ── Theoretical curves ─────────────────────────────────────────────────────────
def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2.0))

def rayleigh_ber(ebno_linear):
    return 0.5 * (1 - np.sqrt(ebno_linear / (1 + ebno_linear)))

ebno_theory  = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 200)
ebno_lin     = 10 ** (ebno_theory / 10.0)
ber_awgn     = qfunc(np.sqrt(2 * ebno_lin))
ber_rayleigh = rayleigh_ber(ebno_lin)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Monte Carlo curves
for name, (style, label) in styles.items():
    ax.semilogy(results[name]["snrs"], results[name]["bers"],
                style, linewidth=2, markersize=5, label=label)

# Theoretical reference curves
ax.semilogy(ebno_theory, ber_rayleigh, 'r--',
            linewidth=2, label="Rayleigh theoretical (perfect CSI)")
ax.semilogy(ebno_theory, ber_awgn, 'k--',
            linewidth=1.5, label="AWGN theoretical")

ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
ax.set_ylabel("BER", fontsize=12)
ax.set_title("Effect of Pilot Density on Channel Estimation\nBPSK over Rayleigh + OFDM", fontsize=13)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, which="both")
ax.set_ylim([1e-5, 1])
ax.set_xlim([EBN0_DB_MIN, EBN0_DB_MAX])

plt.tight_layout()
plt.show()

# ── Print data bits per transmission for each config ──────────────────────────
print("\nPilot overhead summary:")
print(f"{'Config':<12} {'Pilot syms':>10} {'Data bits':>10} {'Overhead %':>12}")
print("-" * 46)
configs = [
    ("2 pilots",  2,  models["2 pilots"].block_length),
    ("4 pilots",  4,  models["4 pilots"].block_length),
    ("6 pilots",  6,  models["6 pilots"].block_length),
]
max_bits = configs[0][2]
for name, npilots, nbits in configs:
    overhead = (max_bits - nbits) / max_bits * 100
    print(f"{name:<12} {npilots:>10} {nbits:>10} {overhead:>11.1f}%")

ber_2_pilots = results['2 pilots']['bers']
ber_4_pilots = results['4 pilots']['bers']
ber_6_pilots = results['6 pilots']['bers']
ber_perfect_csi = results['Perfect CSI']['bers']
ebno_dbs_common = results['2 pilots']['snrs'] # Eb/No values are the same for all

mse_2_pilots = (ber_2_pilots - ber_perfect_csi)**2
mse_4_pilots = (ber_4_pilots - ber_perfect_csi)**2
mse_6_pilots = (ber_6_pilots - ber_perfect_csi)**2

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(ebno_dbs_common, mse_2_pilots, 'md-', label='MSE (2 pilots vs Perfect CSI)')
ax.semilogy(ebno_dbs_common, mse_4_pilots, 'bs-', label='MSE (4 pilots vs Perfect CSI)')
ax.semilogy(ebno_dbs_common, mse_6_pilots, 'g^-', label='MSE (6 pilots vs Perfect CSI)')

ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
ax.set_ylabel("Mean Squared Error (BER)", fontsize=12)
ax.set_title("Mean Squared Error between Estimated and Perfect CSI BER", fontsize=13)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, which="both")
plt.tight_layout()
plt.show()
