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
    x = frequencies * T_obs
    pix = jnp.pi * x

    first_term = (jnp.sin(pix) / (pix)) ** 2

    second_term = (
        jnp.sqrt(3)
        / (2 * jnp.pi)
        * (-2 * jnp.sin(pix) / (pix**2) + 2 * jnp.cos(pix) / x)
    ) ** 2

    third_term = (
        jnp.sqrt(5)
        / (-4 * jnp.pi**2)
        * (
            12 * jnp.sin(pix) / (pix**3)
            - 12 * jnp.cos(pix) / x**2
            - 4 * jnp.pi * jnp.sin(pix) / x
        )
    ) ** 2

    return 1.0 - first_term - second_term - third_term


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

    transmission = transmission_function_quadratic(frequencies, T_obs)

    year_peak = (0.99 * jnp.sinc((frequencies - ut.f_yr) * T_obs)) ** 2

    return transmission - year_peak


@jax.jit
def get_tf(f, t, Mmat):

    N, m = jnp.shape(Mmat)
    U, _, _ = jnp.linalg.svd(Mmat)
    G = U[:, m:]

    exp = jnp.exp(jnp.einsum("f,t->ft", f, 2.0j * jnp.pi * t))

    G_exp = jnp.einsum("ta,ft->fta", G, exp)

    GG = jnp.real(jnp.einsum("fta,fwa->f", G_exp, jnp.conj(G_exp))) / N

    return GG
