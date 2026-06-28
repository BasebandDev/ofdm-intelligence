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
# ── Model ────────────────────────────────────────────────────────────────
class BPSKModel(sionna.phy.Block):
    """
    Unified BPSK model for AWGN with and without OFDM.
    flag_ofdm=False → plain BPSK over AWGN (no resource grid)
    flag_ofdm=True  → BPSK + OFDM over AWGN (with resource grid + equalizer)
    """
    def __init__(self, num_bits_per_symbol, flag_ofdm=False):
        super().__init__()
        self.n_bits   = num_bits_per_symbol
        self.flag_ofdm = flag_ofdm

        # Shared blocks — used by both paths
        self.constellation = sionna.phy.mapping.Constellation("pam", num_bits_per_symbol)
        self.mapper        = sionna.phy.mapping.Mapper(constellation=self.constellation)
        self.binary_source = sionna.phy.mapping.BinarySource()
        self.awgn          = sionna.phy.channel.AWGN()

        if flag_ofdm:
            rg = ResourceGrid(
                num_ofdm_symbols=NUM_OFDM_SYMBOLS,
                fft_size=FFT_SIZE,
                subcarrier_spacing=SUBCARRIER_SPACING,
                num_tx=1,
                pilot_pattern="kronecker",
                pilot_ofdm_symbol_indices=[2, 11]
            )
            self.rg           = rg
            self.block_length = rg.num_data_symbols * num_bits_per_symbol
            self.rg_mapper    = ResourceGridMapper(rg)
            sm = StreamManagement(np.array([[1]]), num_streams_per_tx=1)
            self.equalizer    = LinearDetector(
                equalizer="lmmse",
                output="bit",
                demapping_method="app",
                resource_grid=rg,
                stream_management=sm,
                constellation_type="pam",
                num_bits_per_symbol=num_bits_per_symbol
            )
        else:
            # block_length set externally after OFDM model is instantiated
            # so both models use same number of bits — fair comparison
            self.block_length = None
            self.demapper     = sionna.phy.mapping.Demapper(
                                    "app", constellation=self.constellation)

    def call(self, batch_size, ebno_db):
        no = sionna.phy.utils.ebnodb2no(
            ebno_db, num_bits_per_symbol=self.n_bits, coderate=1.0)
        if isinstance(no, torch.Tensor):
            no = no.detach().clone().to(torch.float32)
        else:
            no = torch.tensor(no, dtype=torch.float32)

        if self.flag_ofdm:
            # BPSK + OFDM path
            bits = self.binary_source([batch_size, 1, 1, self.block_length])
            x    = self.mapper(bits)
            x_rg = self.rg_mapper(x)
            y_rg = self.awgn(x_rg, no)

            # AWGN → flat channel H=1, perfect knowledge, no estimation needed
            h_hat   = torch.ones(
                [batch_size, 1, NUM_RX_ANT, 1, 1, NUM_OFDM_SYMBOLS, FFT_SIZE],
                dtype=torch.complex64, device=x_rg.device)
            err_var = torch.zeros_like(h_hat)

            llr = self.equalizer(y_rg, h_hat, err_var, no)

        else:
            # Plain BPSK path
            bits = self.binary_source([batch_size, self.block_length])
            x    = self.mapper(bits)
            y    = self.awgn(x, no)
            llr  = self.demapper(y, no)

        return bits, llr

# ── Instantiate ────────────────────────────────────────────────────────────────
model_ofdm = BPSKModel(NUM_BITS_PER_SYMBOL, flag_ofdm=True)

# Set plain BPSK block_length to match OFDM — fair comparison
model_bpsk = BPSKModel(NUM_BITS_PER_SYMBOL, flag_ofdm=False)
model_bpsk.block_length = model_ofdm.block_length

print(f"Block length: {model_ofdm.block_length} bits")

# ── Simulate ───────────────────────────────────────────────────────────────────
ebno_dbs  = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)
ber_plots = sionna.phy.utils.PlotBER("BPSK vs BPSK+OFDM over AWGN")

for model, legend in [(model_bpsk, "BPSK AWGN"), (model_ofdm, "BPSK+OFDM AWGN")]:
    ber_plots.simulate(
        model,
        ebno_dbs=ebno_dbs,
        batch_size=BATCH_SIZE,
        num_target_block_errors=100,
        legend=legend,
        soft_estimates=True,
        max_mc_iter=N_ITERS,
        show_fig=False
    )

# ── Extract results ────────────────────────────────────────────────────────────
snrs_bpsk, bers_bpsk = ber_plots._snrs[0], ber_plots._bers[0]
snrs_ofdm, bers_ofdm = ber_plots._snrs[1], ber_plots._bers[1]

# ── Theoretical curve ──────────────────────────────────────────────────────────
def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2.0))

ebno_theory = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 200)
ber_theory  = qfunc(np.sqrt(2 * 10 ** (ebno_theory / 10.0)))

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogy(snrs_bpsk, bers_bpsk, 'bo-', linewidth=2, markersize=5,
            label="BPSK AWGN — Monte Carlo")
ax.semilogy(snrs_ofdm, bers_ofdm, 'gs-', linewidth=2, markersize=5,
            label="BPSK+OFDM AWGN — Monte Carlo")
ax.semilogy(ebno_theory, ber_theory, 'r--', linewidth=2,
            label="BPSK Theoretical")

ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
ax.set_ylabel("BER", fontsize=12)
ax.set_title("BPSK vs BPSK+OFDM over AWGN", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, which="both")
ax.set_ylim([1e-5, 1])
ax.set_xlim([EBN0_DB_MIN, EBN0_DB_MAX])

plt.tight_layout()
plt.show()



