# The Piecewise Parabolic Method for Hydrodynamics


The PPM interface construction is more involved for hydrodynamics than advection
because there are 3 characteristic waves.  The basic overview of the method is:

* Convert the conserved state, ${\bf U}$ to primitive variables, ${\bf q}$

* Reconstruct each of the primitive variables as a parabola.

  We first find the values of the parabola on the left and right edges of the zone,
  ${\bf q}_{-,i}$ and ${\bf q}_{+,i}$.  This is done by defining a conservative
  cubic interpolant that passes through the 2 zones on each side of the interface.
  The unlimited version of this would be:

  $${\bf q}_{i+1/2} = \frac{7}{12} ({\bf q}_i + {\bf q}_{i+1}) - \frac{1}{12} ({\bf q}_{i-1} + {\bf q}_{i+2})$$

  we then use this value to set:

  $${\bf q}_{+,i} = {\bf q}_{-,i+1} = {\bf q}_{i+1/2}$$

  Working zone-by-zone, the values ${\bf q}_{-,i}$ and ${\bf q}_{+,i}$
  are then limited, and we define the parabolic reconstruction in zone
  $i$ as:

  $${\bf q}_i(x) = {\bf q}_{-,i} + \xi (\Delta {\bf q}_i + {\bf q}_{6,i} (1 - \xi))$$

  where

  $$\Delta {\bf q}_i = {\bf q}_{+,i} - {\bf q}_{-,i}$$

  $${\bf q}_{6,i} = 6 \left({\bf q}_i - \frac{1}{2} ({\bf q}_{-,i} + {\bf q}_{+,i})\right )$$

  and

  $$\xi = \frac{x - x_{i-1/2}}{\Delta x}$$

* Integrate under the parabola for the distance $\lambda^{(\nu)}_i \Delta t$ for each of the
  characteristic waves $\nu$: $u-c$, $u$, and $u+c$.  We define the dimensionless wave speed, $\sigma_i^{(\nu)}$:

  $$\sigma_i^{(\nu)} = \frac{\lambda^{(\nu)} \Delta t}{\Delta x}$$

  From the right edge, we have:

  \begin{align*}
  \mathcal{I}_+^{(\nu)}({\bf q}_i) &=
      \frac{1}{\sigma_i^{(\nu)} \Delta x} \int_{x_{i+1/2} - \sigma_i^{(\nu)}\Delta x}^{x_{i+1/2}} {\bf q}(x) \, dx \\
      &= {\bf q}_{+,i} - \frac{\sigma_i^{(\nu)}}{2} \left [ \Delta {\bf q}_i - {\bf q}_{6,i} \left (1 - \frac{2}{3} \sigma_i^{(\nu)}\right )\right ]
  \end{align*}

  and from the left edge, we have:

  \begin{align*}
  \mathcal{I}_-^{(\nu)}({\bf q}_i) &=
      \frac{1}{\sigma_i^{(\nu)} \Delta x} \int_{x_{i-1/2}}^{x_{i-1/2} + \sigma_i^{(\nu)}\Delta x} {\bf q}(x) \, dx \\
      &= {\bf q}_{-,i} + \frac{\sigma_i^{(\nu)}}{2} \left [ \Delta {\bf q}_i + {\bf q}_{6,i} \left (1 - \frac{2}{3} \sigma_i^{(\nu)}\right )\right ]
  \end{align*}

* Define a reference state.  We are going to project the amount of
  ${\bf q}$ carried by each wave into the characteristic variables and
  then sum up all of the jumps that move toward each interface.  To
  minimize the effects of this characteristic projection, we will
  subtract off a reference state, $\tilde{\bf q}$:

  $$\tilde{\bf q}_{+,i} = \mathcal{I}_+^{(+)}({\bf q}_i)$$

  $$\tilde{\bf q}_{-,i} = \mathcal{I}_-^{(-)}({\bf q}_i)$$

  In each case, we are picking the fastest wave moving toward the interface.

* Define the left and right states on the interfaces seen by zones $i$ by
  adding up all of the jumps that reach that interface:

  $${\bf q}_{i+1/2,L}^{n+1/2} = \tilde{{\bf q}}_+ -
   \sum_{\nu;\lambda^{(\nu)}\ge 0} {\bf l}_i^{(\nu)} \cdot \left (
        \tilde{{\bf q}}_+ - \mathcal{I}_+^{(\nu)}({\bf q}_i)
       \right ) {\bf r}_i^{(\nu)}$$

  $${\bf q}_{i-1/2,R}^{n+1/2} = \tilde{{\bf q}}_- -
   \sum_{\nu;\lambda^{(\nu)}\le 0} {\bf l}_i^{(\nu)} \cdot \left (
        \tilde{{\bf q}}_+ - \mathcal{I}_-^{(\nu)}({\bf q}_i)
       \right ) {\bf r}_i^{(\nu)}$$

  Notice that zone $i$ gives the left state on interface $i+1/2$ and the
  right state on zone $i-1/2$.

We then solve the Riemann problem using these state.
