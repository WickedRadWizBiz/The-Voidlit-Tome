import jaxpath  # noqa
import jax, jax.numpy as jnp
import tome_jax as T
st, static = T.build(0); win = T.make_window(); drive = jnp.zeros(T.N)
a0 = jax.tree_util.tree_map(lambda x: (x.shape, x.dtype, getattr(x, 'weak_type', None)), st)
st2, sp = win(static, st, drive)
a1 = jax.tree_util.tree_map(lambda x: (x.shape, x.dtype, getattr(x, 'weak_type', None)), st2)
flat0, _ = jax.tree_util.tree_flatten_with_path(a0, is_leaf=lambda x: isinstance(x, tuple))
flat1, _ = jax.tree_util.tree_flatten_with_path(a1, is_leaf=lambda x: isinstance(x, tuple))
for (p0, v0), (p1, v1) in zip(flat0, flat1):
    if v0 != v1: print(jax.tree_util.keystr(p0), v0, '->', v1)
