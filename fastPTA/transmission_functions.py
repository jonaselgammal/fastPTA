# Global
import jax
import jax.numpy as jnp

# Local
import fastPTA.utils as ut


@jax.jit
def transmission_function_approx(frequencies, T_obs):
    """
    Compute the transmission function (see Eq. 3 of 2404.02864), which
    represents the attenuation of signals, for some frequencies given the
    observation time.

    Parameters:
    -----------
    frequencies : Array
        Array of frequencies (in Hz).
    T_obs : float
        Observation time (in seconds).

    Returns:
    --------
    transmission : Array
        Array of transmission values computed for the given frequencies and
        observation time.

    """

    return 1 / (1 + 1 / (frequencies * T_obs) ** 6)

@jax.jit
def transmission_function_quadratic(frequencies, T_obs):
    """
    Compute the transmission function (see App. C of 2507.18593), which
    gives the attenuation of signals, for a quadratic spindown timing model.

    Parameters:
    -----------
    frequencies : Array
        Array of frequencies (in Hz).
    T_obs : float
        Observation time (in seconds).

    Returns:
    --------
    transmission : Array
        Array of transmission values computed for the given frequencies and
        observation time.

    """
    f = frequencies * T_obs
    return (1. - (jnp.sin(jnp.pi*f)/(jnp.pi*f))**2
            - (jnp.sqrt(3)/(2*jnp.pi) * (-2 * jnp.sin(jnp.pi*f)/(jnp.pi*f**2) + 2*jnp.cos(jnp.pi*f)/f))**2
            - (jnp.sqrt(5)/(-4*jnp.pi**2) * (12*jnp.sin(jnp.pi*f) / (jnp.pi*f**3) -12*jnp.cos(jnp.pi*f) / f**2 - 4*jnp.pi*jnp.sin(jnp.pi*f)/f))**2)

@jax.jit
def transmission_function_quadratic_1yr_peak(frequencies, T_obs):
    """
    Compute the transmission function (see App. C of 2507.18593), which
    gives the attenuation of signals, for a quadratic spindown timing model,
    reproducing the one year peak due to the Earth orbit.

    Parameters:
    -----------
    frequencies : Array
        Array of frequencies (in Hz).
    T_obs : float
        Observation time (in seconds).

    Returns:
    --------
    transmission : Array
        Array of transmission values computed for the given frequencies and
        observation time.

    """
    f = frequencies * T_obs
    return (1. - (jnp.sin(jnp.pi*f)/(jnp.pi*f))**2
            - (jnp.sqrt(3)/(2*jnp.pi) * (-2 * jnp.sin(jnp.pi*f)/(jnp.pi*f**2) + 2*jnp.cos(jnp.pi*f)/f))**2
            - (jnp.sqrt(5)/(-4*jnp.pi**2) * (12*jnp.sin(jnp.pi*f) / (jnp.pi*f**3) -12*jnp.cos(jnp.pi*f) / f**2 - 4*jnp.pi*jnp.sin(jnp.pi*f)/f))**2
            - (0.99 * jnp.sinc((frequencies - ut.f_yr)*T_obs))**2)