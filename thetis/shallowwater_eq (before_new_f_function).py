r"""
Depth averaged shallow water equations

---------
Equations
---------

The state variables are water elevation, :math:`\eta`, and depth averaged
velocity :math:`\bar{\textbf{u}}`.

Denoting the total water depth by :math:`H=\eta + h`, the non-conservative form of
the free surface equation is

.. math::
   \frac{\partial \eta}{\partial t} + \nabla \cdot (H \bar{\textbf{u}}) = 0
   :label: swe_freesurf

The non-conservative momentum equation reads

.. math::
   \frac{\partial \bar{\textbf{u}}}{\partial t} +
   \bar{\textbf{u}} \cdot \nabla\bar{\textbf{u}} +
   f\textbf{e}_z\wedge \bar{\textbf{u}} +
   g \nabla \eta +
   g \frac{1}{H}\int_{-h}^\eta \nabla r dz
   = \nabla \cdot \big( \nu_h ( \nabla \bar{\textbf{u}} + (\nabla \bar{\textbf{u}})^T )\big) +
   \frac{\nu_h \nabla(H)}{H} \cdot ( \nabla \bar{\textbf{u}} + (\nabla \bar{\textbf{u}})^T ),
   :label: swe_momentum

where :math:`g` is the gravitational acceleration, :math:`f` is the Coriolis
frequency, :math:`\wedge` is the cross product,
:math:`\textbf{e}_z` is a vertical unit vector, and :math:`\nu_h`
is viscosity. Water density is given by :math:`\rho = \rho'(T, S, p) + \rho_0`,
where :math:`\rho_0` is a constant reference density.

Above :math:`r` denotes the baroclinic head

.. math::

  r = \frac{1}{\rho_0} \int_{z}^\eta  \rho' d\zeta.

In the case of purely barotropic problems the :math:`r` and the internal pressure
gradient are omitted.

If the option :attr:`.ModelOptions.nonlin` is ``False``, we solve the linear shallow water
equations (i.e. wave equation):

.. math::
   \frac{\partial \eta}{\partial t} + \nabla \cdot (h \bar{\textbf{u}}) = 0
   :label: swe_freesurf_linear

.. math::
   \frac{\partial \bar{\textbf{u}}}{\partial t} +
   f\textbf{e}_z\wedge \bar{\textbf{u}} +
   g \nabla \eta
   = \nabla \cdot \big( \nu_h ( \nabla \bar{\textbf{u}} + (\nabla \bar{\textbf{u}})^T )\big) +
   \frac{\nu_h \nabla(H)}{H} \cdot ( \nabla \bar{\textbf{u}} + (\nabla \bar{\textbf{u}})^T ).
   :label: swe_momentum_linear

In case of a 3D problem with mode splitting, we use a simplified 2D
system that contains nothing but the rotational external gravity waves:

.. math::
    \frac{\partial \eta}{\partial t} + \nabla \cdot (H \bar{\textbf{u}}) = 0
    :label: swe_freesurf_modesplit

.. math::
    \frac{\partial \bar{\textbf{u}}}{\partial t} +
    f\textbf{e}_z\wedge \bar{\textbf{u}} +
    g \nabla \eta
    = \textbf{G},
    :label: swe_momentum_modesplit

where :math:`\textbf{G}` is a source term used to couple the 2D and 3D momentum
equations.

-------------------
Boundary Conditions
-------------------

All boundary conditions are imposed weakly by providing external values for
:math:`\eta` and :math:`\bar{\textbf{u}}`.

Boundary conditions are set with a dictionary that defines all prescribed
variables at each open boundary.
For example, to assign elevation and volume flux on boundary ``1`` we set

.. code-block:: python

    swe_bnd_funcs = {}
    swe_bnd_funcs[1] = {'elev':myfunc1, 'flux':myfunc2}

where ``myfunc1`` and ``myfunc2`` are :class:`Constant` or :class:`Function`
objects.

The user can provide :math:`\eta` and/or :math:`\bar{\textbf{u}}` values.
Supported combinations are:

- *unspecified* : impermeable (land) boundary, implies symmetric :math:`\eta` condition and zero normal velocity
- ``'elev'``: elevation only, symmetric velocity (usually unstable)
- ``'uv'``: 2d velocity vector :math:`\bar{\textbf{u}}=(u, v)` (in mesh coordinates), symmetric elevation
- ``'un'``: normal velocity (scalar, positive out of domain), symmetric elevation
- ``'flux'``: normal volume flux (scalar, positive out of domain), symmetric elevation
- ``'elev'`` and ``'uv'``: water elevation and 2d velocity vector
- ``'elev'`` and ``'un'``: water elevation and normal velocity
- ``'elev'`` and ``'flux'``: water elevation and normal flux

The boundary conditions are assigned to the :class:`.FlowSolver2d` or
:class:`.FlowSolver` objects:

.. code-block:: python

    solver_obj = solver2d.FlowSolver2d(...)
    ...
    solver_obj.bnd_functions['shallow_water'] = swe_bnd_funcs

Internally the boundary conditions passed to the :meth:`.Term.residual` method
of each term:

.. code-block:: python

    adv_term = shallowwater_eq.HorizontalAdvectionTerm(...)
    adv_form = adv_term.residual(..., bnd_conditions=swe_bnd_funcs)

"""
from __future__ import absolute_import
from .utility import *
from .equation import Term, Equation

__all__ = [
    'BaseShallowWaterEquation',
    'ShallowWaterEquations',
    'ModeSplit2DEquations',
    'ShallowWaterMomentumEquation',
    'FreeSurfaceEquation',
    'ShallowWaterTerm',
    'ShallowWaterMomentumTerm',
    'ShallowWaterContinuityTerm',
    'HUDivTerm',
    'ContinuitySourceTerm',
    'HorizontalAdvectionTerm',
    'HorizontalViscosityTerm',
    'ExternalPressureGradientTerm',
    'CoriolisTerm',
    'LinearDragTerm',
    'QuadraticDragTerm',
    'BottomDrag3DTerm',
    'MomentumSourceTerm',
    'WindStressTerm',
]

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']


class ShallowWaterTerm(Term):
    """
    Generic term in the shallow water equations that provides commonly used
    members and mapping for boundary functions.
    """
    def __init__(self, space,
                 bathymetry=None,
                 nonlin=True,
                 wd_alpha=None,
                 wd_mindep=None):
        super(ShallowWaterTerm, self).__init__(space)

        self.bathymetry = bathymetry
        self.nonlin = nonlin
        self.wd_alpha = wd_alpha
        self.wd_mindep = wd_mindep

        # mesh dependent variables
        self.cellsize = CellSize(self.mesh)

        # define measures with a reasonable quadrature degree
        p = self.function_space.ufl_element().degree()
        self.quad_degree = 2*p + 1
        self.dx = dx(degree=self.quad_degree,
                     domain=self.function_space.ufl_domain())
        self.dS = dS(degree=self.quad_degree,
                     domain=self.function_space.ufl_domain())

    def get_bnd_functions(self, eta_in, uv_in, bnd_id, bnd_conditions):
        """
        Returns external values of elev and uv for all supported
        boundary conditions.

        Volume flux (flux) and normal velocity (un) are defined positive out of
        the domain.
        """
        bath = self.bathymetry
        bnd_len = self.boundary_len[bnd_id]
        funcs = bnd_conditions.get(bnd_id)
        if 'elev' in funcs and 'uv' in funcs:
            eta_ext = funcs['elev']
            uv_ext = funcs['uv']
        elif 'elev' in funcs and 'un' in funcs:
            eta_ext = funcs['elev']
            uv_ext = funcs['un']*self.normal
        elif 'elev' in funcs and 'flux' in funcs:
            eta_ext = funcs['elev']
            h_ext = eta_ext + bath
            area = h_ext*bnd_len  # NOTE using external data only
            uv_ext = funcs['flux']/area*self.normal
        elif 'elev' in funcs:
            eta_ext = funcs['elev']
            uv_ext = uv_in  # assume symmetry
        elif 'uv' in funcs:
            eta_ext = eta_in  # assume symmetry
            uv_ext = funcs['uv']
        elif 'un' in funcs:
            eta_ext = eta_in  # assume symmetry
            uv_ext = funcs['un']*self.normal
        elif 'flux' in funcs:
            eta_ext = eta_in  # assume symmetry
            h_ext = eta_ext + bath
            area = h_ext*bnd_len  # NOTE using internal elevation
            uv_ext = funcs['flux']/area*self.normal
        else:
            raise Exception('Unsupported bnd type: {:}'.format(funcs.keys()))
        return eta_ext, uv_ext

#    def wd_bathymetry_displacement(self, solution):
        """
        Returns wetting and drying bathymetry discplacement as described in:
        Karna et al.,  2011.
        """
#        if self.wd_alpha is None:
#            return self.update_depth_wd(self, solution) - (self.bathymetry + solution[2]) # Wei
#        else:
#            H = self.bathymetry + solution[2]
#            return 0.5 * (sqrt(H ** 2 + self.wd_alpha ** 2) - H)

    def get_depth_ext(self, eta): # Wei, 'get_total_depth' before
        """
        Returns total water column depth for external input
        """
        if self.nonlin:
            if self.wd_alpha is None:
                total_h = self.bathymetry + eta
            else:
                H = self.bathymetry + eta
                total_h = H + 0.5 * (sqrt(H ** 2 + self.wd_alpha ** 2) - H)
        else:
            total_h = self.bathymetry
        return total_h

#    def update_depth_wd(self, solution): # Wei, added
        """
        Returns new total water column depth because of wetting-drying judges
        """
#        if self.nonlin:
        
#            if self.wd_alpha is None:
#                H = self.bathymetry.dat.data + solution.dat.data[1]        
#                H[H < self.wd_mindep] = self.wd_mindep
#                total_h = self.bathymetry
#                tmp = total_h.dat.data
#                assert H.shape[0] == tmp.shape[0]
#                for i, h_wd in enumerate(H):
#                    tmp[i] = h_wd
#            else:
#                total_h = self.bathymetry + solution[2] + self.wd_bathymetry_displacement(solution)

#        else:
#            total_h = self.bathymetry

#        return total_h


    def update_uv_wd(self, uv): # Wei, added
        """
        Updates if dying, flux should be set zero AND redistributed
        """        


class ShallowWaterMomentumTerm(ShallowWaterTerm):
    """
    Generic term in the shallow water momentum equation that provides commonly used
    members and mapping for boundary functions.
    """
    def __init__(self, u_test, u_space, eta_space,
                 bathymetry=None,
                 nonlin=True,
                 wd_alpha=None,
                 wd_mindep=None,
                 include_grad_div_viscosity_term=False,
                 include_grad_depth_viscosity_term=True):
        super(ShallowWaterMomentumTerm, self).__init__(u_space, bathymetry, nonlin, wd_alpha, wd_mindep)

        self.include_grad_div_viscosity_term = include_grad_div_viscosity_term
        self.include_grad_depth_viscosity_term = include_grad_depth_viscosity_term

        self.u_test = u_test
        self.u_space = u_space
        self.eta_space = eta_space

        self.u_continuity = element_continuity(self.u_space.ufl_element()).horizontal
        self.eta_is_dg = element_continuity(self.eta_space.ufl_element()).horizontal == 'dg'


class ShallowWaterContinuityTerm(ShallowWaterTerm):
    """
    Generic term in the depth-integrated continuity equation that provides commonly used
    members and mapping for boundary functions.
    """
    def __init__(self, eta_test, eta_space, u_space,
                 bathymetry=None,
                 nonlin=True,
                 wd_alpha=None,
                 wd_mindep=None):
        super(ShallowWaterContinuityTerm, self).__init__(eta_space, bathymetry, nonlin, wd_alpha, wd_mindep)

        self.eta_test = eta_test
        self.eta_space = eta_space
        self.u_space = u_space

        self.u_continuity = element_continuity(self.u_space.ufl_element()).horizontal
        self.eta_is_dg = element_continuity(self.eta_space.ufl_element()).horizontal == 'dg'


class ExternalPressureGradientTerm(ShallowWaterMomentumTerm):
    """
    External pressure gradient term, :math:`g \nabla \eta`

    The weak form reads

    .. math::
        \int_\Omega g \nabla \eta \cdot \boldsymbol{\psi} dx
        = \int_\Gamma g \eta^* \text{jump}(\boldsymbol{\psi} \cdot \textbf{n}) dS
        - \int_\Omega g \eta \nabla \cdot \boldsymbol{\psi} dx

    where the right hand side has been integrated by parts; :math:`\textbf{n}`
    denotes the unit normal of the element interfaces, :math:`n^*` is value at
    the interface obtained from an approximate Riemann solver.

    If :math:`\eta` belongs to a discontinuous function space, the form on the
    right hand side is used.
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None): # Wei

        head = eta

        grad_eta_by_parts = self.eta_is_dg # true means discontinuous Lagrange, Wei 28/03/2017

        if grad_eta_by_parts:
            f = -g_grav*head*nabla_div(self.u_test)*self.dx
            if uv is not None:
                head_star = avg(head) + 0.5*sqrt(avg(total_h)/g_grav)*jump(uv, self.normal)
            else:
                head_star = avg(head)
            f += g_grav*head_star*jump(self.u_test, self.normal)*self.dS
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(head, uv, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    un_jump = inner(uv - uv_ext, self.normal)
                    eta_rie = 0.5*(head + eta_ext) + sqrt(total_h/g_grav)*un_jump
                    f += g_grav*eta_rie*dot(self.u_test, self.normal)*ds_bnd
                if funcs is None or 'symm' in funcs:
                    # assume land boundary
                    # impermeability implies external un=0
                    un_jump = inner(uv, self.normal)
                    head_rie = head + sqrt(total_h/g_grav)*un_jump
                    f += g_grav*head_rie*dot(self.u_test, self.normal)*ds_bnd
        else:
            f = g_grav*inner(grad(head), self.u_test) * self.dx
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(head, uv, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    un_jump = inner(uv - uv_ext, self.normal)
                    eta_rie = 0.5*(head + eta_ext) + sqrt(total_h/g_grav)*un_jump
                    f += g_grav*(eta_rie-head)*dot(self.u_test, self.normal)*ds_bnd
        return -f


class HUDivTerm(ShallowWaterContinuityTerm):
    r"""
    Divergence term, :math:`\nabla \cdot (H \bar{\textbf{u}})`

    The weak form reads

    .. math::
        \int_\Omega \nabla \cdot (H \bar{\textbf{u}}) \phi dx
        = \int_\Gamma (H^* \bar{\textbf{u}}^*) \cdot \text{jump}(\phi \textbf{n}) dS
        - \int_\Omega H (\bar{\textbf{u}}\cdot\nabla \phi) dx

    where the right hand side has been integrated by parts; :math:`\textbf{n}`
    denotes the unit normal of the element interfaces, and :math:`\text{jump}`
    and :math:`\text{avg}` denote the jump and average operators across the
    interface. :math:`H^*, \bar{\textbf{u}}^*` are values at the interface
    obtained from an approximate Riemann solver.

    If :math:`\bar{\textbf{u}}` belongs to a discontinuous function space,
    the form on the right hand side is used.
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None): # Wei

        hu_by_parts = self.u_continuity in ['dg', 'hdiv']

        if hu_by_parts:
            f = -inner(grad(self.eta_test), total_h*uv)*self.dx
            if self.eta_is_dg:
                h = avg(total_h)
                uv_rie = avg(uv) + sqrt(g_grav/h)*jump(eta, self.normal)
                hu_star = h*uv_rie
                f += inner(jump(self.eta_test, self.normal), hu_star)*self.dS
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(eta, uv, bnd_marker, bnd_conditions)
                    eta_ext_old, uv_ext_old = self.get_bnd_functions(eta_old, uv_old, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    total_h_ext = self.get_depth_ext(eta_ext_old) # Wei
                    h_av = 0.5*(total_h + total_h_ext)
                    eta_jump = eta - eta_ext
                    un_rie = 0.5*inner(uv + uv_ext, self.normal) + sqrt(g_grav/h_av)*eta_jump
                    un_jump = inner(uv_old - uv_ext_old, self.normal)
                    eta_rie = 0.5*(eta_old + eta_ext_old) + sqrt(h_av/g_grav)*un_jump
                    h_rie = self.bathymetry + eta_rie
                    f += h_rie*un_rie*self.eta_test*ds_bnd
        else:
            f = div(total_h*uv)*self.eta_test*self.dx
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is None or 'un' in funcs:
                    f += -total_h*dot(uv, self.normal)*self.eta_test*ds_bnd
        return -f


class HorizontalAdvectionTerm(ShallowWaterMomentumTerm):
    r"""
    Advection of momentum term, :math:`\bar{\textbf{u}} \cdot \nabla\bar{\textbf{u}}`

    The weak form is

    .. math::
        \int_\Omega \bar{\textbf{u}} \cdot \nabla\bar{\textbf{u}} \cdot \boldsymbol{\psi} dx
        = - \int_\Omega \nabla_h \cdot (\bar{\textbf{u}} \boldsymbol{\psi}) \cdot \bar{\textbf{u}} dx
        + \int_\Gamma \text{avg}(\bar{\textbf{u}}) \cdot \text{jump}(\boldsymbol{\psi}
        (\bar{\textbf{u}}\cdot\textbf{n})) dS

    where the right hand side has been integrated by parts;
    :math:`\textbf{n}` is the unit normal of
    the element interfaces, and :math:`\text{jump}` and :math:`\text{avg}` denote the
    jump and average operators across the interface.
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None): # Wei
        uv_lax_friedrichs = fields_old.get('uv_lax_friedrichs')

        if not self.nonlin:
            return 0

        horiz_advection_by_parts = True

        if horiz_advection_by_parts:
            # f = -inner(nabla_div(outer(uv, self.u_test)), uv)
            f = -(Dx(uv_old[0]*self.u_test[0], 0)*uv[0] +
                  Dx(uv_old[0]*self.u_test[1], 0)*uv[1] +
                  Dx(uv_old[1]*self.u_test[0], 1)*uv[0] +
                  Dx(uv_old[1]*self.u_test[1], 1)*uv[1])*self.dx
            if self.u_continuity in ['dg', 'hdiv']:
                un_av = dot(avg(uv_old), self.normal('-'))
                # NOTE solver can stagnate
                # s = 0.5*(sign(un_av) + 1.0)
                # NOTE smooth sign change between [-0.02, 0.02], slow
                # s = 0.5*tanh(100.0*un_av) + 0.5
                # uv_up = uv('-')*s + uv('+')*(1-s)
                # NOTE mean flux
                uv_up = avg(uv)
                f += (uv_up[0]*jump(self.u_test[0], uv_old[0]*self.normal[0]) +
                      uv_up[1]*jump(self.u_test[1], uv_old[0]*self.normal[0]) +
                      uv_up[0]*jump(self.u_test[0], uv_old[1]*self.normal[1]) +
                      uv_up[1]*jump(self.u_test[1], uv_old[1]*self.normal[1]))*self.dS
                # Lax-Friedrichs stabilization
                if uv_lax_friedrichs is not None:
                    gamma = 0.5*abs(un_av)*uv_lax_friedrichs
                    f += gamma*dot(jump(self.u_test), jump(uv))*self.dS
                    for bnd_marker in self.boundary_markers:
                        funcs = bnd_conditions.get(bnd_marker)
                        ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                        if funcs is None:
                            # impose impermeability with mirror velocity
                            n = self.normal
                            uv_ext = uv - 2*dot(uv, n)*n
                            gamma = 0.5*abs(dot(uv_old, n))*uv_lax_friedrichs
                            f += gamma*dot(self.u_test, uv-uv_ext)*ds_bnd
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(eta, uv, bnd_marker, bnd_conditions)
                    eta_ext_old, uv_ext_old = self.get_bnd_functions(eta_old, uv_old, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    eta_jump = eta_old - eta_ext_old
                    un_rie = 0.5*inner(uv_old + uv_ext_old, self.normal) + sqrt(g_grav/total_h)*eta_jump
                    uv_av = 0.5*(uv_ext + uv)
                    f += (uv_av[0]*self.u_test[0]*un_rie +
                          uv_av[1]*self.u_test[1]*un_rie)*ds_bnd
        return -f


class HorizontalViscosityTerm(ShallowWaterMomentumTerm):
    r"""
    Viscosity of momentum term

    If option :attr:`.ModelOptions.include_grad_div_viscosity_term` is ``True``, we
    use the symmetric viscous stress :math:`\boldsymbol{\tau}_\nu = \nu_h ( \nabla \bar{\textbf{u}} + (\nabla \bar{\textbf{u}})^T )`.
    Using the symmetric interior penalty method the weak form then reads

    .. math::
        \int_\Omega -\nabla \cdot \boldsymbol{\tau}_\nu \cdot \boldsymbol{\psi} dx
        =& \int_\Omega (\nabla \boldsymbol{\psi}) : \boldsymbol{\tau}_\nu dx \\
        &- \int_\Gamma \text{jump}(\boldsymbol{\psi} \textbf{n}) \cdot \text{avg}(\boldsymbol{\tau}_\nu) dS
        - \int_\Gamma \text{avg}(\nu_h)\big(\text{jump}(\bar{\textbf{u}} \textbf{n}) + \text{jump}(\bar{\textbf{u}} \textbf{n})^T\big) \cdot \text{avg}(\nabla \boldsymbol{\psi}) dS \\
        &+ \int_\Gamma \sigma \text{avg}(\nu_h) \big(\text{jump}(\bar{\textbf{u}} \textbf{n}) + \text{jump}(\bar{\textbf{u}} \textbf{n})^T\big) \cdot \text{jump}(\boldsymbol{\psi} \textbf{n}) dS

    where :math:`\sigma` is a penalty parameter,
    see Epshteyn and Riviere (2007).

    If option :attr:`.ModelOptions.include_grad_div_viscosity_term` is ``False``,
    we use viscous stress :math:`\boldsymbol{\tau}_\nu = \nu_h \nabla \bar{\textbf{u}}`.
    In this case the weak form is

    .. math::
        \int_\Omega -\nabla \cdot \boldsymbol{\tau}_\nu \cdot \boldsymbol{\psi} dx
        =& \int_\Omega (\nabla \boldsymbol{\psi}) : \boldsymbol{\tau}_\nu dx \\
        &- \int_\Gamma \text{jump}(\boldsymbol{\psi} \textbf{n}) \cdot \text{avg}(\boldsymbol{\tau}_\nu) dS
        - \int_\Gamma \text{avg}(\nu_h)\text{jump}(\bar{\textbf{u}} \textbf{n}) \cdot \text{avg}(\nabla \boldsymbol{\psi}) dS \\
        &+ \int_\Gamma \sigma \text{avg}(\nu_h) \text{jump}(\bar{\textbf{u}} \textbf{n}) \cdot \text{jump}(\boldsymbol{\psi} \textbf{n}) dS

    If option :attr:`.ModelOptions.include_grad_depth_viscosity_term` is ``True``, we also include
    the term

    .. math::
        \boldsymbol{\tau}_{\nabla H} = - \frac{\nu_h \nabla(H)}{H} \cdot ( \nabla \bar{\textbf{u}} + (\nabla \bar{\textbf{u}})^T )

    as a source term.

    Epshteyn and Riviere (2007). Estimation of penalty parameters for symmetric
    interior penalty Galerkin methods. Journal of Computational and Applied
    Mathematics, 206(2):843-872. http://dx.doi.org/10.1016/j.cam.2006.08.029

    .. note ::
        Note the minus sign due to :class:`.equation.Term` sign convention
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None): # Wei

        nu = fields_old.get('viscosity_h')
        if nu is None:
            return 0

        n = self.normal
        h = self.cellsize

        if self.include_grad_div_viscosity_term:
            stress = nu*2.*sym(grad(uv))
            stress_jump = avg(nu)*2.*sym(tensor_jump(uv, n))
        else:
            stress = nu*grad(uv)
            stress_jump = avg(nu)*tensor_jump(uv, n)

        f = inner(grad(self.u_test), stress)*self.dx

        if self.u_continuity in ['dg', 'hdiv']:
            # from Epshteyn et al. 2007 (http://dx.doi.org/10.1016/j.cam.2006.08.029)
            # the scheme is stable for alpha > 3*X*p*(p+1)*cot(theta), where X is the
            # maximum ratio of viscosity within a triangle, p the degree, and theta
            # with X=2, theta=6: cot(theta)~10, 3*X*cot(theta)~60
            p = self.u_space.ufl_element().degree()
            alpha = 5.*p*(p+1)
            if p == 0:
                alpha = 1.5
            f += (
                + alpha/avg(h)*inner(tensor_jump(self.u_test, n), stress_jump)*self.dS
                - inner(avg(grad(self.u_test)), stress_jump)*self.dS
                - inner(tensor_jump(self.u_test, n), avg(stress))*self.dS
            )

            # Dirichlet bcs only for DG
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    if 'un' in funcs:
                        delta_uv = (dot(uv, n) - funcs['un'])*n
                    else:
                        eta_ext, uv_ext = self.get_bnd_functions(eta, uv, bnd_marker, bnd_conditions)
                        if uv_ext is uv:
                            continue
                        delta_uv = uv - uv_ext

                    if self.include_grad_div_viscosity_term:
                        stress_jump = nu*2.*sym(outer(delta_uv, n))
                    else:
                        stress_jump = nu*outer(delta_uv, n)

                    f += (
                        alpha/h*inner(outer(self.u_test, n), stress_jump)*ds_bnd
                        - inner(grad(self.u_test), stress_jump)*ds_bnd
                        - inner(outer(self.u_test, n), stress)*ds_bnd
                    )

        if self.include_grad_depth_viscosity_term:
            f += -dot(self.u_test, dot(grad(total_h)/total_h, stress))*self.dx

        return -f


class CoriolisTerm(ShallowWaterMomentumTerm):
    r"""
    Coriolis term, :math:`f\textbf{e}_z\wedge \bar{\textbf{u}}`
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None): # Wei
        coriolis = fields_old.get('coriolis')
        f = 0
        if coriolis is not None:
            f += coriolis*(-uv[1]*self.u_test[0] + uv[0]*self.u_test[1])*self.dx
        return -f


class WindStressTerm(ShallowWaterMomentumTerm):
    r"""
    Wind stress term, :math:`-\tau_w/(H \rho_0)`

    Here :math:`\tau_w` is a user-defined wind stress :class:`Function`.
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None): # Wei
        wind_stress = fields_old.get('wind_stress')
        f = 0
        if wind_stress is not None:
            f += -dot(wind_stress, self.u_test)/total_h/rho_0*self.dx
        return -f


class QuadraticDragTerm(ShallowWaterMomentumTerm):
    r"""
    Quadratic Manning bottom friction term
    :math:`C_D \| \bar{\textbf{u}} \| \bar{\textbf{u}}`

    where the drag term is computed with the Manning formula

    .. math::
        C_D = g \frac{\mu^2}{H^{1/3}}

    if the Manning coefficient :math:`\mu` is defined (see field :attr:`mu_manning`).
    Otherwise :math:`C_D` is taken as a constant (see field :attr:`quadratic_drag`).
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None): # Wei
        mu_manning = fields_old.get('mu_manning')
        C_D = fields_old.get('quadratic_drag')
        f = 0
        if mu_manning is not None:
            if C_D is not None:
                raise Exception('Cannot set both dimensionless and Manning drag parameter')
            C_D = g_grav * mu_manning**2 / total_h**(1./3.)

        if C_D is not None:
            f += C_D * sqrt(dot(uv_old, uv_old)) * inner(self.u_test, uv) / total_h * self.dx
        return -f


class LinearDragTerm(ShallowWaterMomentumTerm):
    r"""
    Linear friction term, :math:`C \bar{\textbf{u}}`

    Here :math:`C` is a user-defined drag coefficient.
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None): # Wei
        linear_drag = fields_old.get('linear_drag')
        f = 0
        if linear_drag is not None:
            bottom_fri = linear_drag*inner(self.u_test, uv)*self.dx
            f += bottom_fri
        return -f


class BottomDrag3DTerm(ShallowWaterMomentumTerm):
    r"""
    Bottom drag term consistent with the 3D mode,
    :math:`C_D \| \textbf{u}_b \| \textbf{u}_b`

    Here :math:`\textbf{u}_b` is the bottom velocity used in the 3D mode, and
    :math:`C_D` the corresponding bottom drag.
    These fields are computed in the 3D model.
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None): # Wei
        bottom_drag = fields_old.get('bottom_drag')
        uv_bottom = fields_old.get('uv_bottom')
        f = 0
        if bottom_drag is not None and uv_bottom is not None:
            uvb_mag = sqrt(uv_bottom[0]**2 + uv_bottom[1]**2)
            stress = bottom_drag*uvb_mag*uv_bottom/total_h
            bot_friction = dot(stress, self.u_test)*self.dx
            f += bot_friction
        return -f


class MomentumSourceTerm(ShallowWaterMomentumTerm):
    r"""
    Generic source term in the shallow water momentum equation

    The weak form reads

    .. math::
        F_s = \int_\Omega \boldsymbol{\tau} \cdot \boldsymbol{\psi} dx

    where :math:`\boldsymbol{\tau}` is a user defined vector valued :class:`Function`.

    .. note ::
        Due to the sign convention of :class:`.equation.Term`, this term is assembled to the left hand side of the equation
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None): # Wei
        f = 0
        uv_source = fields_old.get('uv_source')

        if uv_source is not None:
            f += -inner(uv_source, self.u_test)*self.dx
        return -f


class ContinuitySourceTerm(ShallowWaterContinuityTerm):
    r"""
    Generic source term in the depth-averaged continuity equation

    The weak form reads

    .. math::
        F_s = \int_\Omega S \phi dx

    where :math:`S` is a user defined scalar :class:`Function`.

    .. note ::
        Due to the sign convention of :class:`.equation.Term`, this term is assembled to the left hand side of the equation
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None): # Wei
        f = 0
        elev_source = fields_old.get('elev_source')

        if elev_source is not None:
            f += -inner(elev_source, self.eta_test)*self.dx
        return -f


class BathymetryDisplacementMassTerm(ShallowWaterContinuityTerm):
    r"""
    Bathmetry mass displacement term, :math:`\partial \eta / \partial t + \partial \tilde{h} / \partial t`

    The weak form reads

    .. math::
        \int_\Omega ( \partial \eta / \partial t + \partial \tilde{h} / \partial t ) \phi dx
         = \int_\Omega (\partial \tilde{H} / \partial t) \phi dx
    """
    def residual(self, solution):
        if isinstance(solution, list):
            uv, eta = solution
        else:
            uv, eta = split(solution)
        f = 0
        if self.wd_alpha is not None:
            f += inner(self.wd_bathymetry_displacement(solution), self.eta_test)*self.dx
        return f


class BaseShallowWaterEquation(Equation):
    """
    Abstract base class for ShallowWaterEquations, ShallowWaterMomentumEquation
    and FreeSurfaceEquation.

    Provides common functionality to compute time steps and add either momentum
    or continuity terms.
    """
    def __init__(self, function_space,
                 bathymetry,
                 nonlin=True,
                 wd_alpha=None,
                 wd_mindep=None):
        super(BaseShallowWaterEquation, self).__init__(function_space)
        self.bathymetry = bathymetry
        self.nonlin = nonlin
        self.wd_alpha = wd_alpha
        self.wd_mindep = wd_mindep

    def add_momentum_terms(self, *args):
        self.add_term(ExternalPressureGradientTerm(*args), 'implicit')
        self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(HorizontalViscosityTerm(*args), 'explicit')
        self.add_term(CoriolisTerm(*args), 'explicit')
        self.add_term(WindStressTerm(*args), 'source')
        self.add_term(QuadraticDragTerm(*args), 'explicit')
        self.add_term(LinearDragTerm(*args), 'explicit')
        self.add_term(BottomDrag3DTerm(*args), 'source')
        self.add_term(MomentumSourceTerm(*args), 'source')

    def add_continuity_terms(self, *args):
        self.add_term(HUDivTerm(*args), 'implicit')
        self.add_term(ContinuitySourceTerm(*args), 'source')

    def update_depth_wd(self, solution): # Wei, added
        """
        Returns new total water column depth because of wetting-drying judges
        """
        if self.nonlin:
        
            if self.wd_alpha is None:
                #if self.wd_mindep is not None:
                    # beflow judgement about wd_mindep has been moved to time loop in splver2d
                    #H = self.bathymetry.dat.data + solution.dat.data[1]        
                    #H[H < self.wd_mindep] = self.wd_mindep
                    #total_h = Function(self.bathymetry.function_space())
                    #tmp = total_h.dat.data
                    #assert H.shape[0] == tmp.shape[0]
                    #for i, h_wd in enumerate(H):
                    #    tmp[i] = h_wd
                #else:
                    #total_h = self.bathymetry + solution[2]
                total_h = self.bathymetry + solution[2]
            else:
                total_h = self.bathymetry + solution[2] + self.water_height_displacement(solution)

        else:
            total_h = self.bathymetry

        return total_h

    def water_height_displacement(self, solution):
        """
        Returns wetting and drying water height discplacement as described in:
        Karna et al.,  2011.
        """
        if self.wd_alpha is None:
            return (self.update_depth_wd(solution) - (self.bathymetry + solution[2])) # Wei
        else:
            H = self.bathymetry + solution[2]
            return 0.5 * (sqrt(H ** 2 + self.wd_alpha ** 2) - H)

    def ElevationDisplacementMassTerm(self, solution):
        """
        Elevation mass displacement term, :math:`\partial \eta / \partial t + \partial \tilde{h} / \partial t`

        The weak form reads

        .. math::
            \int_\Omega ( \partial \eta / \partial t + \partial \tilde{h} / \partial t ) \phi dx
             = \int_\Omega (\partial \tilde{H} / \partial t) \phi dx
        """
        f = 0
        if self.wd_alpha is not None:
            p = self.function_space.ufl_element().degree()
            self.quad_degree = 2*p + 1
            self.dx = dx(degree=self.quad_degree,
                         domain=self.function_space.ufl_domain())

            u_test, eta_test = TestFunctions(self.function_space)
  
            f += inner(self.water_height_displacement(solution), eta_test)*self.dx
            
        return f


    def residual_uv_eta(self, label, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions): # Wei
        f = 0
        for term in self.select_terms(label):
            f += term.residual(uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions) # Wei
        return f


class ShallowWaterEquations(BaseShallowWaterEquation):
    """
    2D depth-averaged shallow water equations in non-conservative form.

    This defines the full 2D SWE equations :eq:`swe_freesurf` -
    :eq:`swe_momentum`.
    """
    def __init__(self, function_space,
                 bathymetry,
                 nonlin=True,
                 wd_alpha=None,
                 wd_mindep=None,
                 include_grad_div_viscosity_term=False,
                 include_grad_depth_viscosity_term=True):
        """
        :arg function_space: Mixed function space where the solution belongs
        :arg bathymetry: bathymetry of the domain
        :type bathymetry: :class:`Function` or :class:`Constant`
        :kwarg bool nonlin: If False defines the linear shallow water equations
        :kwarg wd_alpha: wetting-drying parameter
        :kwarg bool include_grad_div_viscosity_term: If True includes grad(nu div(u))
            viscosity term
        :kwarg bool include_grad_depth_viscosity_term: If True includes grad(H) term
            in viscosity operator
        """
        super(ShallowWaterEquations, self).__init__(function_space, bathymetry, nonlin, wd_alpha, wd_mindep)

        u_test, eta_test = TestFunctions(function_space)
        u_space, eta_space = function_space.split()

        self.add_momentum_terms(u_test, u_space, eta_space,
                                bathymetry, nonlin, wd_alpha, wd_mindep,
                                include_grad_div_viscosity_term,
                                include_grad_depth_viscosity_term)

        self.add_continuity_terms(eta_test, eta_space, u_space, bathymetry, nonlin, wd_alpha, wd_mindep)

    def mass_term(self, solution):
        """if modify eta, need handle here"""
        f = super(ShallowWaterEquations, self).mass_term(solution)
        f += self.ElevationDisplacementMassTerm(solution) # Wei
        return f

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        if isinstance(solution, list):
            uv, eta = solution
        else:
            uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        total_h = self.update_depth_wd(solution_old)
        #uv, eta = solution.split()
        return self.residual_uv_eta(label, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions) # Wei


class ModeSplit2DEquations(BaseShallowWaterEquation):
    r"""
    2D depth-averaged shallow water equations for mode splitting schemes.

    Defines the equations :eq:`swe_freesurf_modesplit` -
    :eq:`swe_momentum_modesplit`.
    """
    def __init__(self, function_space,
                 bathymetry,
                 nonlin=True,
                 include_grad_div_viscosity_term=False,
                 include_grad_depth_viscosity_term=True):
        """
        :arg function_space: Mixed function space where the solution belongs
        :arg bathymetry: bathymetry of the domain
        :type bathymetry: :class:`Function` or :class:`Constant`
        :kwarg bool nonlin: If False defines the linear shallow water equations
        :kwarg bool include_grad_div_viscosity_term: If True includes grad(nu div(u))
            viscosity term
        :kwarg bool include_grad_depth_viscosity_term: If True includes grad(H) term
            in viscosity operator
        """
        # TODO remove include_grad_* options as viscosity operator is omitted
        super(ModeSplit2DEquations, self).__init__(function_space, bathymetry, nonlin)

        u_test, eta_test = TestFunctions(function_space)
        u_space, eta_space = function_space.split()

        self.add_momentum_terms(u_test, u_space, eta_space,
                                bathymetry,
                                nonlin,
                                include_grad_div_viscosity_term,
                                include_grad_depth_viscosity_term)

        self.add_continuity_terms(eta_test, eta_space, u_space, bathymetry, nonlin)

    def add_momentum_terms(self, *args):
        self.add_term(ExternalPressureGradientTerm(*args), 'implicit')
        self.add_term(CoriolisTerm(*args), 'explicit')
        self.add_term(MomentumSourceTerm(*args), 'source')

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        if isinstance(solution, list):
            uv, eta = solution
        else:
            uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        return self.residual_uv_eta(label, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions)


class FreeSurfaceEquation(BaseShallowWaterEquation):
    """
    2D free surface equation :eq:`swe_freesurf` in non-conservative form.
    """
    def __init__(self, eta_test, eta_space, u_space,
                 bathymetry,
                 nonlin=True,
                 wd_alpha=None):
        """
        :arg eta_test: test function of the elevation function space
        :arg eta_space: elevation function space
        :arg u_space: velocity function space
        :arg function_space: Mixed function space where the solution belongs
        :arg bathymetry: bathymetry of the domain
        :type bathymetry: :class:`Function` or :class:`Constant`
        :kwarg bool nonlin: If False defines the linear shallow water equations
        :kwarg wd_alpha: wetting-drying parameter
        """
        super(FreeSurfaceEquation, self).__init__(eta_space, bathymetry, nonlin, wd_alpha)
        self.add_continuity_terms(eta_test, eta_space, u_space, bathymetry, nonlin, wd_alpha)
        self.bathymetry_displacement_mass_term = BathymetryDisplacementMassTerm(eta_test, eta_space, u_space, bathymetry, nonlin, wd_alpha)

    def mass_term(self, solution):
        f = super(ShallowWaterEquations, self).mass_term(solution)
        f += self.bathymetry_displacement_mass_term.residual(solution)
        return f

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        uv = fields['uv']
        uv_old = fields_old['uv']
        eta = solution
        eta_old = solution_old
        return self.residual_uv_eta(label, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions)


class ShallowWaterMomentumEquation(BaseShallowWaterEquation):
    """
    2D depth averaged momentum equation :eq:`swe_momentum` in non-conservative
    form.
    """
    def __init__(self, eta_test, eta_space, u_space,
                 bathymetry,
                 nonlin=True,
                 wd_alpha=None,
                 include_grad_div_viscosity_term=False,
                 include_grad_depth_viscosity_term=True):
        """
        :arg eta_test: test function of the elevation function space
        :arg eta_space: elevation function space
        :arg u_space: velocity function space
        :arg function_space: Mixed function space where the solution belongs
        :arg bathymetry: bathymetry of the domain
        :type bathymetry: :class:`Function` or :class:`Constant`
        :kwarg bool nonlin: If False defines the linear shallow water equations
        :kwarg wd_alpha: wetting-drying parameter
        :kwarg bool include_grad_div_viscosity_term: If True includes grad(nu div(u))
            viscosity term
        :kwarg bool include_grad_depth_viscosity_term: If True includes grad(H) term
            in viscosity operator
        """
        super(ShallowWaterMomentumEquation, self).__init__(u_space, bathymetry, nonlin, wd_alpha)
        self.add_momentum_terms(eta_test, u_space, eta_space,
                                bathymetry, nonlin, wd_alpha,
                                include_grad_div_viscosity_term,
                                include_grad_depth_viscosity_term)

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        uv = solution
        uv_old = solution_old
        eta = fields['eta']
        eta_old = fields_old['eta']
        return self.residual_uv_eta(label, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions)