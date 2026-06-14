- Mapper - at the transmiiter (acts a modulator) - configured using a constellation. It generates complex valued input symbols for the input which come from a binary source. The mapping follows Gray coding by default - adjacent symbols differ by only 1 bit this minimizes BER when symbol errors occur.
- Demapper - at the receiver (acts as a demodulator and decoder) - configured using a constellation. It generates LLRs (soft desision) for symbols at the receiver.
- S/P converter output is one OFDM symbol — a column vector of size FFT_size. Each element maps to one subcarrier. IFFT converts this frequency-domain vector to a time-domain signal for transmission.
- ResourceGrid - Gives the information on which OFDM symbol and sub carrier index we have data or pilots transmission. The fundamental unit of the grid is the RE, representing one subcarrier in one OFDM symbol
- GenerateOFDMChannel - Gives the frequency domain channel response for each of the OFDM symbol in each subcarrier, so it uses ResourceGrid's output as one of the parameter, other being the channel model which has the channel impulse response parameters. So we get the channel impulse response for entire time-freq grid.
- sample channel using GenerateOFDMChannel batch size number of times, using which we estimate the time domain (fftsize x fft_size), frequency domain (n_ofdm_sym x n_ofdm_sym) and spatial covariance (nrx x nrx) matrices.
- StreamManagement - defines the relationship between transmitters, receivers and data streams. SISO case - 1 Rx, 1 Tx and 1 stream
- Channel estimation at pilot REs is done using least squares, and the interpolation for data caryring REs using nearest negbours, Linear and LMMSE are supported.
- LMMSE needs channel statistics and nearest neigbours is the easiest but very error prone.



