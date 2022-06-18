# tested with python3.6, will probably work with version 3.6+
import numpy as np
import typing
import nptyping
import scipy
from scipy import stats
from scipy import misc
import mne

# [ ] create tag: support python 3.6 and above for 2020-2021 work

#%% [markdown]
# The approach used here is the "Filter-Hilbert" method
# This is summarized in Chapter 14 of Analyzing Neural Time Series Data (2014)
# by Michael X. Cohen
# Note: I will try to use z(t) for signals that are complex valued
# and x(t) for the original real signals
#
# This an alternative to the approach of convolving the signal x(t)
# with a complex Wavelet like Morlet $\Psi = A e^{-t^2/2s^2}e^{i 2\pi f t}$
# $A = (\sqrt{s \sqrt{\pi} })^{-1}$
# instantaneous phase at frequency $f$: phase(t)
#      $$ z(t) = a(t) e^{i\phi(t)} = \Psi_{f}(t) * x(t)$$
# phase(t) = angle(z(t))
# there is a recurring theme that we want act on the euler_phase_difference
#   phase_difference = get_phase_difference(epochs, picks)
#   euler_phase_difference = eulerize_phase(phase_difference)
# quote from Mike Cohen Analyzing Neural Time Series
# ISPC is similar to ITPC presented in chapter 19. Recall from equation 19.1
#
# mne provides a epoch.apply_hibert() function to create a analytic signal
# it uses the _my_hilbert function which uses scipy.signal.hilbert

# example of using scipy.signal.hilbert
# fs = 4.00; samples = int(fs*duration);
# t = np.arange(samples)/fs
# signal = chirp(t, 20.0, t[-1] 100.0) # chirp signal
# signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) # ampliutde modulate
#
# analytic_signal = hilbert(signal) # works on last dimension (axis=-1) by default
# ampliutde_envelope = np.abs(analytic_signal)
# instantaneous_phase = np.unwrap(np.angle(analytic_signal)
# instantaneous_frequency = (np.diff(instantaneous_phase)/(2.0*np.pi) * fs)
#
# Question why are you not using mne.spectral_connectivity
# from what I can see there these are all measures with "E" expectation across trials
# not across time intervals
#
# for two electrodes x, y, signal bandpass previously filtered around $f$
# $$ISPC_f = \| n^{-1} \sum_{t=1}^{n} \exp(i (\phi_{xt} - \phi_{yt}) )  \|$$
#
# Note the "euler_phase_difference" acts like a unit vector pointing
# in direction of the phase difference so it makes sense to sum them
# together to get the average phase difference
# %%
# naming convention many variables are numpy ndarrays with multiple dimensions
# A_exyt the _exyt indicates that A is an array with the first dimension e indexing the epoch number
#            xy indicates that two channel dimensions come next for example x=0 and y=4 indiates that
# A[4, 0, 4, 50] indates e=4, x=0, y=4, t=50  or at this refers to the index 4 epoch (actually the 5th epoch)
#                channel 0, channel 5 and timepoint 50
# remember that python, like C, starts counting from 0

def fc_over_time_on_epochs(epochs_pre: mne.Epochs, picks=None, method="wpli"):
    """
    in general this presumes that the epochs have already been bandpass-filtered
    appropriately

    Currently simplified to only support wPLI as a connectivity measure

    returns @cons_exy : "connectivity index by epcoh e, between channel x and y" 
       cons_exy.shape = (n_epochs, n_channel, n_channel)
    """
    epochs = epochs_pre.copy()

    epochs.load_data()
    n_epochs = len(epochs)
    n_channels = len(epochs.info["chs"])

    if not (picks == None):
        n_channels == len(picks)

    cons = np.zeros((n_epochs, n_channels, n_channels))

    phase_based = 0
    phase_options = ["plv", "wpli", "pli", "db_pli", "db_wpli"]
    amplitude_options = ["ortho_power"]
    if method in phase_options:
        phase_difference = get_phase_difference(epochs, picks)
        dataZ_ext = get_analytic_signal(epochs, picks)
        csd_exyt = cross_spectral_density_from_analytic_signal(dataZ_ext)  # complex128

        if method == "wpli":
            cons_exy = wpli_time_on_csd_exyt(csd_exyt)
    else:
        # print("Connectivity metric not supported")
        raise Exception("Connectivity metric not supported")

    return cons_exy



def wpli_time_on_csd_exyt(csd_exyt: np.ndarray):
    """

    Weighted phase-lag index is an extension of the phase-lag index in which angle 
    differences are weighted according to their distance from the real axis.
 
              | E_t[|Im(Sxyt)| sign(Im(Sxyt)] |
    WPLI_xy = -------------------------------
                       E_t[|Im(Sxyt)|]


    input @csd_exyt := cross-spectral density important part is that sample times are the
                    last axis
    returns wpli_exy : typical will return array with shape (n_epochs, n_chan, n_chan)
        though it could work for any csd which has n_times as the last on input                    


    @csd_exyt : cross-spectral density $S_{exyt}$ between x and y at time point t indexed by epoch $e$
    shape: (n_epochs, n_channels, n_channels, n_times), dtype=np.complex128

    
    Refernces: 
    Vinck and colleagues, NeuroImage (2011), Eqn 8
    Mike Cohen, Analyzing Neural Time Series Data, 2014
    Chapter 26: "Phase-Based Connectivity" eqn 26.7 (with correction form Vinck 2011)
    
    """
    icsd_exyt = np.imag(csd_exyt)  # imaginary csd :icsd or $Im(S_{exyt})$
    numerator = np.abs(np.mean(np.abs(icsd_exyt) * np.sign(icsd_exyt), axis=-1))
                                  
    denominator = np.mean(np.abs(icsd_exyt), axis=-1)
    # handle zeros in denominator
    denom_zeros = np.where(denominator == 0.0)  # what about v. near zero?
    denominator[denom_zeros] = 1.0
    wpli_exy = numerator / denominator

    return wpli_exy

def get_phase(analytic_signal):
    """for a given analtyic_signal, usually with last dimention as time point
    following numpy ndaarray convention

    question??? I am not sure if always does this restrict phase to range [0,2pi) ?
    """
    phase_data = np.angle(analytic_signal)
    return phase_data


def get_phase_difference(epochs_pre, picks):
    """
    returns @phase_dif_exyt
    phase_difference shape (n_epochs, n_channels, n_channels, n_times)
    which is the phase difference bewteen (i,j,k,t)
    channel j and k in epoch i at time sample t

    """
    epochs = epochs_pre.copy()
    n_channels = len(picks)

    epoch_data = get_analytic_signal(epochs, picks)

    n_epochs = epoch_data.shape[0]
    n_times = epoch_data.shape[2]

    phase_vals = get_phase(epoch_data)

    phase_dif_exyt = np.zeros((n_epochs, n_channels, n_channels, n_times))
    for idx in range(0, n_epochs):
        epoch_phase = phase_vals[idx, :, :]
        for idx2 in range(0, n_channels):
            phase_1 = epoch_phase[idx2, :]
            for idx3 in range(idx2, n_channels):
                phase_2 = epoch_phase[idx3, :]
                cur_phase_dif = phase_1 - phase_2
                phase_dif_exyt[idx, idx2, idx3, :] = cur_phase_dif
                phase_dif_exyt[idx, idx3, idx2, :] = -cur_phase_dif

    return phase_dif_exyt

def get_analytic_signal(epochs_pre: mne.Epochs, picks):
    """
    Paraemters
    ----------
    @epochs_pre

    @picks

    takes the mne epochs, limited to channels in picks, uses the apply_hilbert() 
    function with envelop=False to get the analytic
    signal: $$x(t) \mapsto  x_a = F^{-1}(F(x) 2 H) = x + i y $$ 
            where $F$ is the fourier transform operator and 
    U is the Heaviside step function (H(t) = 0 for t<0), 

    y(t) = H(x(t)) is the hilbert transform of x(t)
    $$ \frac{1}{\pi t} convolved x(t)  = \frac{1}{\pi} \int_{-\inf}{inf} x(\tau)/(t-\tar) d\tau $$
    using the principle value defintion of the integral

    note: H^2 = -1 

    This is complex-valued with no negative frequency components. The real and 
    imaginary parts are related to each other by the hilbert transform. You can get back 
    to real by just discarding the imagingary part ($y$ above.

    for real valued signal $s(t)$ with fourier pair s(t) <-> S(f) 
    then $S(-f) = S(f)^{*}$

    returns
    -------
    epoch_data : a new 3D array of epochs data(n_epochs, n_channels, n_times)
    see (https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.apply_hilbert)
    """
    epochs = epochs_pre.copy()
    epochs.apply_hilbert(
        picks=picks, envelope=False, n_jobs=1, n_fft="auto", verbose=None
    )
    epoch_data = epochs.get_data(picks=picks)
    return epoch_data


def cross_spectral_density_from_analytic_signal(analytic_signal_ext):
    """
    input @analytic_signal_ext := shape (n_epoch, n_channel, n_times)
    
    returns := csd_exyt # $S_exyt$  the cross-spectral density between x and y at time point t indexed by epoch $e$

    .shape = (n_epochs, n_channels, n_channels, n_times)
    """

    Z_ext = analytic_signal_ext  # complex number
    (n_epochs, n_channels, n_times) = Z_ext.shape

    csd_exyt = np.zeros(
        (n_epochs, n_channels, n_channels, n_times), dtype=np.complex128
    )
    for ee in range(0, n_epochs):
        for xx in range(0, n_channels):
            for yy in range(xx, n_channels):
                csd_exyt[ee, xx, yy, :] = Z_ext[ee, xx, :] * np.conjugate(
                    Z_ext[ee, yy, :]
                )
                csd_exyt[ee, yy, xx, :] = np.conjugate(
                    csd_exyt[ee, xx, yy, :]
                )  # fill in S_ij = conj(S_ji)

    return csd_exyt

# from mne.filter
def _my_hilbert(x, n_fft=None, envelope=False):
    """Compute Hilbert transform of signals w/ zero padding.

    Parameters
    ----------
    x : array, shape (n_times)
        The signal to convert
    n_fft : int
        Size of the FFT to perform, must be at least ``len(x)``.
        The signal will be cut back to original length.
    envelope : bool
        Whether to compute amplitude of the hilbert transform in order
        to return the signal envelope.

    Returns
    -------
    out : array, shape (n_times)
        The hilbert transform of the signal, or the envelope.
    """
    from scipy.signal import hilbert

    n_x = x.shape[-1]
    out = hilbert(x, N=n_fft, axis=-1)[..., :n_x]
    if envelope:
        out = np.abs(out)
    return out
