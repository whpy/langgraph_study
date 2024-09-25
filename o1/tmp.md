```python
from functools import partial
from jax import jit
import jax.numpy as jnp
from src.models import BGKSim

class ReflectionEquationSim(BGKSim):
    """
    Reflection Equation simulation class.

    This class implements the modified equilibrium function according to the given reflection equation.

    The modified equilibrium function is:

    f_eq_alpha = omega_alpha * rho * { 1 + e_alpha · u + 0.5 * [ (e_alpha · u)^2 - 3 * u · u + (T - 1)(e_alpha · e_alpha - D) ] }

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.T = kwargs.get("T", 1.0)  # Temperature parameter, default 1.0
        self.D = kwargs.get("D", self.dim)  # Diffusive parameter, typically equals the dimension

    @partial(jit, static_argnums=(0, 3), inline=True)
    def equilibrium(self, rho, u, cast_output=True):
        """
        Compute the modified equilibrium distribution function according to the reflection equation.

        Parameters
        ----------
        rho : jax.numpy.ndarray
            The macroscopic density.
        u : jax.numpy.ndarray
            The macroscopic velocity.
        cast_output : bool, optional
            Whether to cast outputs to the output precision. Default is True.

        Returns
        -------
        feq : jax.numpy.ndarray
            The equilibrium distribution function.
        """
        # Cast to compute precision if necessary
        if cast_output:
            rho, u = self.precisionPolicy.cast_to_compute((rho, u))

        c = jnp.array(self.c, dtype=self.precisionPolicy.compute_dtype)  # Shape (dim, q)
        c = c.T  # Now shape (q, dim)

        # Compute e_alpha · u
        # u shape: (..., dim)
        # c shape: (q, dim)
        # Need to compute dot product over dim axis
        cu = jnp.einsum('...i, qi -> ...q', u, c)  # Shape (..., q)

        # (e_alpha · u)^2
        cu2 = cu ** 2  # Shape (..., q)

        # Compute u · u
        u_sq = jnp.sum(u ** 2, axis=-1, keepdims=True)  # Shape (..., 1)

        # Compute e_alpha · e_alpha
        e_sq = jnp.sum(c ** 2, axis=-1)  # Shape (q,)

        # Compute (e_alpha · e_alpha - D)
        e_sq_minus_D = e_sq - self.D  # Shape (q,)

        # Compute the equilibrium distribution
        term1 = cu  # Shape (..., q)
        term2 = 0.5 * (cu2 - 3.0 * u_sq + (self.T - 1.0) * e_sq_minus_D)  # Shape (..., q)

        feq = rho * self.w * (1.0 + term1 + term2)  # Shape (..., q)

        if cast_output:
            return self.precisionPolicy.cast_to_output(feq)
        else:
            return feq
```
