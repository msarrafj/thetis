"""
TELEMAC-2D point discharge with diffusion test case
===================================================

Solves tracer advection equation in a rectangular domain with
uniform fluid velocity, constant diffusivity and a constant
tracer source term. Neumann conditions are imposed on the
channel walls and a Dirichlet condition is imposed on the
inflow boundary, with the outflow boundary remaining open.

Further details can be found in [1].

[1] A. Riadh, G. Cedric, M. Jean, "TELEMAC modeling system:
    2D hydrodynamics TELEMAC-2D software release 7.0 user
    manual." Paris:  R&D, Electricite de France, p. 134
    (2014).
"""
from thetis import *


class PassiveTracerParameters():
    def __init__(self):

        # Parametrisation of point source
        self.source_x, self.source_y, self.source_r = 2.0, 5.0, 0.08
        self.source_value = 100.0

        # Physical parameters
        self.diffusivity = Constant(0.1)
        self.uv = Constant(as_vector([1.0, 0.0]))
        self.elev = Constant(0.0)
        self.bathymetry = Constant(1.0)

        # Boundary conditions
        neumann = {'diff_flux': Constant(0.0)}
        dirichlet = {'value': Constant(0.0)}
        outflow = {'open': None}
        self.boundary_conditions = {1: dirichlet, 2: outflow, 3: neumann, 4: neumann}

    def ball(self, mesh, scaling=1.0, eps=1.0e-10):
        x, y = SpatialCoordinate(mesh)
        expr = lt((x-self.source_x)**2 + (y-self.source_y)**2, self.source_r**2 + eps)
        return conditional(expr, scaling, 0.0)

    def source(self, fs):
        nrm = assemble(self.ball(fs.mesh())*dx)
        scaling = 0.5*self.source_value*pi*self.source_r**2/nrm
        source = self.ball(fs.mesh(), scaling=scaling)
        return interpolate(source, fs)


def solve_tracer(n):
    mesh2d = RectangleMesh(100*2**n, 20*2**n, 50, 10)
    P1_2d = FunctionSpace(mesh2d, "CG", 1)

    params = PassiveTracerParameters()
    source = params.source(P1_2d)

    solver_obj = solver2d.FlowSolver2d(mesh2d, params.bathymetry)
    options = solver_obj.options
    options.timestepper_type = 'SteadyState'
    options.timestep = 20.0
    options.simulation_end_time = 18.0
    options.simulation_export_time = 18.0
    # options.timestepper_type = 'CrankNicolson'
    # options.timestep = 0.5
    # options.simulation_end_time = 100.0
    # options.simulation_export_time = 10.0
    options.fields_to_export = ['tracer_2d']
    options.solve_tracer = True
    options.use_lax_friedrichs_tracer = True
    options.tracer_only = True
    options.horizontal_diffusivity = params.diffusivity
    options.tracer_source_2d = source
    solver_obj.assign_initial_conditions(tracer=source, uv=params.uv)
    solver_obj.bnd_functions['tracer'] = params.boundary_conditions
    solver_obj.iterate()


if __name__ == "__main__":
    solve_tracer(2)
