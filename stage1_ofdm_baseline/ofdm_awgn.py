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
                            EPDetector, MMSEPICDetector
from sionna.phy.channel import GenerateOFDMChannel, OFDMChannel, gen_single_sector_topology
from scipy.special import erfc

# Model
class UncodedOFDMAWGN(sionna.phy.Block):
    def __init__(self, num_bits_per_symbol):
        super().__init__()
        rg = ResourceGrid(
            num_ofdm_symbols=NUM_OFDM_SYMBOLS,
            fft_size=FFT_SIZE,
            subcarrier_spacing=SUBCARRIER_SPACING,
            num_tx=1,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11]
        )
        self.rg                  = rg
        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length        = rg.num_data_symbols * num_bits_per_symbol

        rx_tx_association = np.array([[1]])
        stream_management = StreamManagement(rx_tx_association, num_streams_per_tx=1)

        self.rg_mapper       = ResourceGridMapper(rg)
        self.binary_source   = sionna.phy.mapping.BinarySource()
        self.constellation   = sionna.phy.mapping.Constellation("pam", num_bits_per_symbol)
        self.mapper          = sionna.phy.mapping.Mapper(constellation=self.constellation)
        self.channel_model   = sionna.phy.channel.AWGN()
        self.channel_estimator = LSChannelEstimator(rg, interpolation_type='lin')
        self.equalizer = LinearDetector(
            equalizer="lmmse",
            output="bit",
            demapping_method="app",
            resource_grid=rg,
            stream_management=stream_management,
            constellation_type="pam",
            num_bits_per_symbol=num_bits_per_symbol
        )
    def call(self, batch_size, ebno_db):
      no = sionna.phy.utils.ebnodb2no(
        ebno_db,
        num_bits_per_symbol=self.num_bits_per_symbol,
        coderate=1.0
    )
      bits = self.binary_source([batch_size, 1, 1, self.block_length])
      x    = self.mapper(bits)
      x_rg = self.rg_mapper(x)
      y_rg = self.channel_model(x_rg, no)

      # 7D shape: [batch, num_rx, num_tx, num_streams, num_time_steps, num_ofdm_symbols, fft_size]
      h_hat = torch.ones(
        [batch_size, 1, NUM_RX_ANT, 1, 1, NUM_OFDM_SYMBOLS, FFT_SIZE],
        dtype=torch.complex64,
        device=x_rg.device
      ) #AWGN
      err_var = torch.zeros_like(h_hat)

      llr = self.equalizer(y_rg, h_hat, err_var, no)
      print("✓ equalizer", llr.shape)

      return bits, llr


# Simulation parameters
NUM_BITS_PER_SYMBOL = 1
BATCH_SIZE          = 1000
EBN0_DB_MIN         = -3.0
EBN0_DB_MAX         = 5.0
NUM_OFDM_SYMBOLS    = 14
FFT_SIZE            = 12*4
SUBCARRIER_SPACING  = 30e3
CARRIER_FREQUENCY   = 3.5e9
NUM_RX_ANT          = 1
N_ITERS = 100

model = UncodedOFDMAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL)
# ── Monte Carlo Sim via Sionna
ber_plots = sionna.phy.utils.PlotBER("2-PAM / BPSK over AWGN with OFDM")
ber_plots.simulate(
    model,
    ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
    batch_size=BATCH_SIZE,
    num_target_block_errors=100,
    legend="2-PAM Monte Carlo",
    soft_estimates=True,
    max_mc_iter=100,
    show_fig=False          # suppressing Sionna's own figure
)

#print(dir(ber_plots))  This gave the attributes that PlotBER stores, using the _snrs and _bers from it.
mc_snrs = ber_plots._snrs[0]   # Eb/N0 values for first (only) curve
mc_bers = ber_plots._bers[0]   # BER values for first (only) curve

# Theoretical BPSK BER
def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2.0))

ebno_dbs_theory = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 100)
ebno_linear     = 10 ** (ebno_dbs_theory / 10.0)
ber_theory      = qfunc(np.sqrt(2 * ebno_linear))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogy(mc_snrs, mc_bers, 'bo-',
            linewidth=2, markersize=5, label="2-PAM Monte Carlo")
ax.semilogy(ebno_dbs_theory, ber_theory, 'r--',
            linewidth=2, label="BPSK Theoretical")

ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
ax.set_ylabel("BER", fontsize=12)
ax.set_title("2-PAM / BPSK over AWGN + OFDM", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, which="both")
ax.set_ylim([1e-5, 1])
ax.set_xlim([EBN0_DB_MIN, EBN0_DB_MAX])

plt.tight_layout()
plt.show()

