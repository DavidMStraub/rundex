import jax

# Enable 64-bit floats for all tests.
# Must happen before any jax.numpy operations are traced.
jax.config.update("jax_enable_x64", True)
