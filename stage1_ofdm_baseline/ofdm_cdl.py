# ── Simulation parameters ──────────────────────────────────────────────────────
CARRIER_FREQUENCY = 3.5e9
NUM_BITS_PER_SYMBOL = 1
BATCH_SIZE          = 2000
EBN0_DB_MIN         = -3.0
EBN0_DB_MAX         = 5.0
NUM_OFDM_SYMBOLS    = 14
FFT_SIZE            = 12 * 4
SUBCARRIER_SPACING  = 30e3
NUM_RX_ANT          = 1
N_ITERS             = 200


class OFDMCDLFading(sionna.phy.Block):
    """
    BPSK over CDL-A fading + OFDM.
    Mirrors OFDMRayleigh but uses CDL-A channel model.
    No closed-form BER — Monte Carlo only.

    Args:
        num_bits_per_symbol : 1 for BPSK
        pilot_indices       : OFDM symbol indices for pilots
        perfect_csi         : bypass estimator with true channel
        delay_spread        : CDL delay spread in seconds (default 100ns)
    """
    def __init__(self, num_bits_per_symbol, pilot_indices,
                 perfect_csi=False, delay_spread=100e-9):
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
        self.mse_log      = []

        rx_tx = np.array([[1]])
        sm    = StreamManagement(rx_tx, num_streams_per_tx=1)

        self.rg_mapper     = ResourceGridMapper(rg)
        self.binary_source = sionna.phy.mapping.BinarySource()
        self.constellation = sionna.phy.mapping.Constellation("pam", num_bits_per_symbol)
        self.mapper        = sionna.phy.mapping.Mapper(constellation=self.constellation)

        # SISO antenna arrays — minimal config for single antenna each end
        ut_array = AntennaArray(
            num_rows=1, num_cols=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=CARRIER_FREQUENCY
        )
        bs_array = AntennaArray(
            num_rows=1, num_cols=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=CARRIER_FREQUENCY
        )

        # CDL-A channel model
        # delay_spread controls frequency selectivity — larger = more selective
        # min/max speed = 0 → static channel, no Doppler
        self.ch_model = CDL(
            model="A",
            delay_spread=delay_spread,
            carrier_frequency=CARRIER_FREQUENCY,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            min_speed=0.0,
            max_speed=0.0
        )
        self.channel   = OFDMChannel(self.ch_model, rg, return_channel=True)
        self.estimator = LSChannelEstimator(rg, interpolation_type='lin')
        self.equalizer = LinearDetector(
                            equalizer="lmmse",
                            output="bit",
                            demapping_method="app",
                            resource_grid=rg,
                            stream_management=sm,
                            constellation_type="pam",
                            num_bits_per_symbol=num_bits_per_symbol)

    def reset_mse_log(self):
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

        # Log channel MSE — free side effect of forward pass
        mse      = ((h_freq - h_hat).abs() ** 2).mean().item()
        ebno_val = ebno_db.item() if isinstance(ebno_db, torch.Tensor) else float(ebno_db)
        self.mse_log.append((ebno_val, mse))

        if self.perfect_csi:
            h_hat   = h_freq
            err_var = torch.zeros_like(h_hat)

        llr = self.equalizer(y_rg, h_hat, err_var, no)
        return bits, llr

models_cdl = {
    "CDL-A LS est":    OFDMCDLFading(NUM_BITS_PER_SYMBOL, [2, 11], perfect_csi=False),
    "CDL-A perfect":   OFDMCDLFading(NUM_BITS_PER_SYMBOL, [2, 11], perfect_csi=True),
}

# ── Simulate ───────────────────────────────────────────────────────────────────
ebno_dbs   = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)
ber_plots  = sionna.phy.utils.PlotBER("CDL-A fading")

for name, model in models_cdl.items():
    model.reset_mse_log()
    print(f"Simulating: {name} ...")
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
snrs_ls,      bers_ls      = ber_plots._snrs[0], ber_plots._bers[0]
snrs_perfect, bers_perfect = ber_plots._snrs[1], ber_plots._bers[1]

# ── Theoretical AWGN reference ─────────────────────────────────────────────────
def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2.0))

ebno_theory = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 200)
ber_awgn    = qfunc(np.sqrt(2 * 10 ** (ebno_theory / 10.0)))

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogy(snrs_ls,      bers_ls,      'md-', linewidth=2, markersize=5,
            label="CDL-A — LS est + LMMSE EQ")
ax.semilogy(snrs_perfect, bers_perfect, 'k^-', linewidth=2, markersize=5,
            label="CDL-A — perfect CSI + LMMSE EQ")
ax.semilogy(ebno_theory,  ber_awgn,     'r--', linewidth=2,
            label="AWGN theoretical (reference)")

ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
ax.set_ylabel("BER", fontsize=12)
ax.set_title("BPSK over CDL-A + OFDM\nLS Estimated vs Perfect CSI", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, which="both")
ax.set_ylim([1e-5, 1])
ax.set_xlim([EBN0_DB_MIN, EBN0_DB_MAX])

plt.tight_layout()
plt.show()
