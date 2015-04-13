# Wave equation in 3D
# ===================
#
# Rectangular channel geometry.
#
# Tuomas Karna 2015-03-11

from cofs import *

# set physical constants
physical_constants['z0_friction'].assign(0.0)

mesh2d = Mesh('channel_waveEq.msh')
layers = 6

depth = 50.0
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry2d = Function(P1_2d, name='Bathymetry')
bathymetry2d.assign(50.0)

x_func = Function(P1_2d).interpolate(Expression('x[0]'))
x_min = x_func.dat.data.min()
x_max = x_func.dat.data.max()
x_min = comm.allreduce(x_min, x_min, op=MPI.MIN)
x_max = comm.allreduce(x_max, x_max, op=MPI.MAX)
Lx = x_max - x_min

# set time step, and run duration
c_wave = float(np.sqrt(9.81*depth))
T_cycle = Lx/c_wave
n_steps = 20*2
dt = float(T_cycle/n_steps)
TExport = dt
T = 10*T_cycle + 1e-3
# explicit model
Umag = Constant(2.0)

# create solver
solverObj = solver.flowSolver(mesh2d, bathymetry2d, layers)
solverObj.nonlin = False
solverObj.solveSalt = True
solverObj.solveVertDiffusion = False
solverObj.useBottomFriction = False
solverObj.useALEMovingMesh = True
solverObj.baroclinic = False
solverObj.useSUPG = False
solverObj.useGJV = False
solverObj.dt = dt
solverObj.TExport = TExport
solverObj.T = T
solverObj.uAdvection = Umag
solverObj.checkVolConservation2d = True
solverObj.checkVolConservation3d = True
solverObj.checkSaltConservation = True
solverObj.fieldsToExport = ['uv2d', 'elev2d', 'uv3d',
                            'w3d', 'w3d_mesh', 'salt3d',
                            'uv2d_dav', 'barohead3d',
                            'barohead2d', 'gjvAlphaH3d', 'gjvAlphaV3d']
solverObj.timerLabels = []

solverObj.mightyCreator()

elev_init = Function(solverObj.H_2d)
elev_init.project(Expression('-eta_amp*cos(2*pi*x[0]/Lx)', eta_amp=1.0,
                             Lx=Lx))

salt_init3d = Function(solverObj.H, name='initial salinity')
salt_init3d.interpolate(Expression('4.5'))

solverObj.assingInitialConditions(elev=elev_init, salt=salt_init3d)
solverObj.iterate()

exit(0)

# ------------------ oldies VV

# Function spaces for 2d mode
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
U_2d = VectorFunctionSpace(mesh2d, 'DG', 1)
U_visu_2d = VectorFunctionSpace(mesh2d, 'CG', 1)
U_scalar_2d = FunctionSpace(mesh2d, 'DG', 1)
H_2d = FunctionSpace(mesh2d, 'CG', 2)
W_2d = MixedFunctionSpace([U_2d, H_2d])

solution2d = Function(W_2d, name='solution2d')
# Mean free surface height (bathymetry)
bathymetry2d = Function(P1_2d, name='Bathymetry')

uv_bottom2d = Function(U_2d, name='Bottom Velocity')
z_bottom2d = Function(P1_2d, name='Bot. Vel. z coord')
bottom_drag2d = Function(P1_2d, name='Bottom Drag')

use_wd = False
nonlin = False
swe2d = mode2d.freeSurfaceEquations(mesh2d, W_2d, solution2d, bathymetry2d,
                                    uv_bottom=None, bottom_drag=None,
                                    nonlin=nonlin, use_wd=use_wd)

x_func = Function(P1_2d).interpolate(Expression('x[0]'))
x_min = x_func.dat.data.min()
x_max = x_func.dat.data.max()
x_min = comm.allreduce(x_min, x_min, op=MPI.MIN)
x_max = comm.allreduce(x_max, x_max, op=MPI.MAX)
Lx = x_max - x_min

depth_oce = 50.0
depth_riv = 50.0
bath_x = np.array([0, Lx])
bath_v = np.array([depth_oce, depth_riv])
depth = 20.0


def bath(x, y, z):
    padval = 1e20
    x0 = np.hstack(([-padval], bath_x, [padval]))
    vals0 = np.hstack(([bath_v[0]], bath_v, [bath_v[-1]]))
    return interp1d(x0, vals0)(x)

#define a bath func depending on x,y,z
bathymetry2d.dat.data[:] = bath(x_func.dat.data, 0, 0)

outputDir = createDirectory('outputs')
bathfile = File(os.path.join(outputDir, 'bath.pvd'))
bathfile << bathymetry2d

elev_init = Function(H_2d)
elev_init.project(Expression('-eta_amp*cos(2*pi*x[0]/Lx)', eta_amp=1.0,
                             Lx=Lx))

# create 3d equations

# extrude mesh
n_layers = 6
mesh = extrudeMeshSigma(mesh2d, n_layers, bathymetry2d)

# function spaces
P1 = FunctionSpace(mesh, 'CG', 1, vfamily='CG', vdegree=1)
U = VectorFunctionSpace(mesh, 'DG', 1, vfamily='CG', vdegree=1)
U_visu = VectorFunctionSpace(mesh, 'CG', 1, vfamily='CG', vdegree=1)
U_scalar = FunctionSpace(mesh, 'DG', 1, vfamily='CG', vdegree=1)
H = FunctionSpace(mesh, 'CG', 2, vfamily='CG', vdegree=1)

eta3d = Function(H, name='Elevation')
eta3d_nplushalf = Function(H, name='Elevation')
bathymetry3d = Function(P1, name='Bathymetry')
copy2dFieldTo3d(swe2d.bathymetry, bathymetry3d)
uv3d = Function(U, name='Velocity')
uv_bottom3d = Function(U, name='Bottom Velocity')
z_bottom3d = Function(P1, name='Bot. Vel. z coord')
# z coordinate in the strecthed mesh
z_coord3d = Function(P1, name='Bot. Vel. z coord')
# z coordinate in the reference mesh (eta=0)
z_coord_ref3d = Function(P1, name='Bot. Vel. z coord')
bottom_drag3d = Function(P1, name='Bottom Drag')
uv3d_dav = Function(U, name='Depth Averaged Velocity')
uv2d_dav = Function(U_2d, name='Depth Averaged Velocity')
uv2d_dav_old = Function(U_2d, name='Depth Averaged Velocity')
w3d = Function(H, name='Vertical Velocity')
w_mesh3d = Function(H, name='Vertical Velocity')
dw_mesh_dz_3d = Function(H, name='Vertical Velocity')
w_mesh_surf3d = Function(H, name='Vertical Velocity')
salt3d = Function(H, name='Salinity')
viscosity_v3d = Function(P1, name='Vertical Velocity')

salt_init3d = Function(H, name='initial salinity')
salt_init3d.interpolate(Expression('4.5'))


def getZCoord(zcoord):
    fs = zcoord.function_space()
    tri = TrialFunction(fs)
    test = TestFunction(fs)
    a = tri*test*dx
    L = fs.mesh().coordinates[2]*test*dx
    solve(a == L, zcoord)
    return zcoord

getZCoord(z_coord3d)
z_coord_ref3d.assign(z_coord3d)

mom_eq3d = mode3d.momentumEquation(mesh, U, U_scalar, swe2d.boundary_markers,
                                   swe2d.boundary_len, uv3d, eta3d,
                                   bathymetry3d, w=w3d,
                                   w_mesh=w_mesh3d,
                                   dw_mesh_dz=dw_mesh_dz_3d,
                                   viscosity_v=None,
                                   nonlin=nonlin)
salt_eq3d = mode3d.tracerEquation(mesh, H, salt3d, eta3d, uv3d, w=w3d,
                                  w_mesh=w_mesh3d,
                                  dw_mesh_dz=dw_mesh_dz_3d,
                                  bnd_markers=swe2d.boundary_markers,
                                  bnd_len=swe2d.boundary_len)
vmom_eq3d = mode3d.verticalMomentumEquation(mesh, U, U_scalar, uv3d, w=None,
                                            viscosity_v=viscosity_v3d,
                                            uv_bottom=uv_bottom3d,
                                            bottom_drag=bottom_drag3d)

# set time step, and run duration
c_wave = float(np.sqrt(9.81*depth_oce))
T_cycle = Lx/c_wave
n_steps = 20*2
dt = float(T_cycle/n_steps)
TExport = dt
T = 10*T_cycle + 1e-3
# explicit model
Umag = Constant(2.0)
#mesh_dt = swe2d.getTimeStepAdvection(Umag=Umag)
#dt_3d = float(np.floor(mesh_dt.dat.data.min()/20.0))
##print dt_3d
#exit(0)
mesh2d_dt = swe2d.getTimeStep(Umag=Umag)
dt_2d = float(np.floor(mesh2d_dt.dat.data.min()/20.0))
dt_2d = round(comm.allreduce(dt_2d, dt_2d, op=MPI.MIN))
dt_2d = float(dt/np.ceil(dt/dt_2d))
M_modesplit = int(dt/dt_2d)
if commrank == 0:
    print 'dt =', dt
    print '2D dt =', dt_2d, M_modesplit
    sys.stdout.flush()

solver_parameters = {
    #'ksp_type': 'fgmres',
    #'ksp_monitor': True,
    'ksp_rtol': 1e-12,
    'ksp_atol': 1e-16,
    #'pc_type': 'fieldsplit',
    #'pc_fieldsplit_type': 'multiplicative',
}
subIterator = timeIntegration.SSPRK33(swe2d, dt_2d, solver_parameters)
timeStepper2d = timeIntegration.macroTimeStepIntegrator(subIterator,
                                               M_modesplit,
                                               restartFromAv=True)

timeStepper_mom3d = timeIntegration.SSPRK33(mom_eq3d, dt,
                                   funcs_nplushalf={'eta': eta3d_nplushalf})
timeStepper_salt3d = timeIntegration.SSPRK33(salt_eq3d, dt)
timeStepper_vmom3d = timeIntegration.CrankNicolson(vmom_eq3d, dt, gamma=0.6)

U_2d_file = exporter(U_visu_2d, 'Depth averaged velocity', outputDir, 'Velocity2d.pvd')
eta_2d_file = exporter(P1_2d, 'Elevation', outputDir, 'Elevation2d.pvd')
eta_3d_file = exporter(P1, 'Elevation', outputDir, 'Elevation3d.pvd')
uv_3d_file = exporter(U_visu, 'Velocity', outputDir, 'Velocity3d.pvd')
w_3d_file = exporter(P1, 'V.Velocity', outputDir, 'VertVelo3d.pvd')
w_mesh_3d_file = exporter(P1, 'Mesh Velocity', outputDir, 'MeshVelo3d.pvd')
salt_3d_file = exporter(P1, 'Salinity', outputDir, 'Salinity3d.pvd')
uv_dav_2d_file = exporter(U_visu_2d, 'Depth Averaged Velocity', outputDir, 'DAVelocity2d.pvd')
uv_bot_2d_file = exporter(U_visu_2d, 'Bottom Velocity', outputDir, 'BotVelocity2d.pvd')
visc_3d_file = exporter(P1, 'Vertical Viscosity', outputDir, 'Viscosity3d.pvd')

# assign initial conditions
uv2d, eta2d = solution2d.split()
eta2d.assign(elev_init)
copy2dFieldTo3d(elev_init, eta3d)
#getZCoord(z_coord3d)
updateCoordinates(mesh, eta3d, bathymetry3d, z_coord3d, z_coord_ref3d)
salt3d.assign(salt_init3d)
computeVertVelocity(w3d, uv3d, bathymetry3d)  # at t{n+1}
computeMeshVelocity(eta3d, uv3d, w3d, w_mesh3d, w_mesh_surf3d,
                    dw_mesh_dz_3d, bathymetry3d, z_coord_ref3d)

timeStepper2d.initialize(solution2d)
timeStepper_mom3d.initialize(uv3d)
timeStepper_salt3d.initialize(salt3d)
timeStepper_vmom3d.initialize(uv3d)

# Export initial conditions
U_2d_file.export(solution2d.split()[0])
eta_2d_file.export(solution2d.split()[1])
eta_3d_file.export(eta3d)
uv_3d_file.export(uv3d)
w_3d_file.export(w3d)
w_mesh_3d_file.export(w_mesh3d)
salt_3d_file.export(salt3d)
uv_dav_2d_file.export(uv2d_dav)
uv_bot_2d_file.export(uv_bottom2d)
visc_3d_file.export(viscosity_v3d)

# The time-stepping loop
T_epsilon = 1.0e-5
cputimestamp = timeMod.clock()
t = 0
i = 0
iExp = 1
next_export_t = t + TExport


updateForcings = None
updateForcings3d = None

Vol_0 = compVolume2d(eta2d, swe2d.dx)
Vol3d_0 = compVolume3d(mom_eq3d.dx)
Mass3d_0 = compTracerMass3d(salt3d, mom_eq3d.dx)
if commrank == 0:
  print 'Initial volume', Vol_0, Vol3d_0

from pyop2.profiling import timed_region, timed_function, timing

while t <= T + T_epsilon:

    # SSPRK33 time integration loop
    with timed_region('mode2d'):
        timeStepper2d.advance(t, dt_2d, solution2d, updateForcings)
    with timed_region('aux_functions'):
        eta_n = solution2d.split()[1]
        copy2dFieldTo3d(eta_n, eta3d)  # at t_{n+1}
        eta_nph = timeStepper2d.solution_nplushalf.split()[1]
        copy2dFieldTo3d(eta_nph, eta3d_nplushalf)  # at t_{n+1/2}
        updateCoordinates(mesh, eta3d, bathymetry3d, z_coord3d, z_coord_ref3d)
    with timed_region('momentumEq'):
        timeStepper_mom3d.advance(t, dt, uv3d, updateForcings3d)
    #with timed_region('aux_functions'):
        #computeParabolicViscosity(uv_bottom3d, bottom_drag3d, bathymetry3d,
                                  #viscosity_v3d)
    #with timed_region('vert_diffusion'):
        #timeStepper_vmom3d.advance(t, dt, uv3d, None)
    with timed_region('continuityEq'):
        computeVertVelocity(w3d, uv3d, bathymetry3d)  # at t{n+1}
        computeMeshVelocity(eta3d, uv3d, w3d, w_mesh3d, w_mesh_surf3d,
                            dw_mesh_dz_3d, bathymetry3d, z_coord_ref3d)
        #dw_mesh_dz_3d.assign(0.0)
        #w_mesh3d.assign(0.0)
        computeBottomFriction(uv3d, uv_bottom2d, uv_bottom3d, z_coord3d,
                              z_bottom2d, z_bottom3d, bathymetry2d,
                              bottom_drag2d, bottom_drag3d)

    with timed_region('saltEq'):
        timeStepper_salt3d.advance(t, dt, salt3d, updateForcings3d)
    with timed_region('aux_functions'):
        bndValue = Constant((0.0, 0.0, 0.0))
        computeVerticalIntegral(uv3d, uv3d_dav, U,
                                bottomToTop=True, bndValue=bndValue,
                                average=True, bathymetry=bathymetry3d)
        copy3dFieldTo2d(uv3d_dav, uv2d_dav, useBottomValue=False)
        # 2d-3d coupling: restart 2d mode from depth ave 3d velocity
        timeStepper2d.solution_start.split()[0].assign(uv2d_dav)
    #with timed_region('continuityEq'):
        #computeVertVelocity(w3d, uv3d, bathymetry3d)  # at t{n+1}
        #computeMeshVelocity(eta3d, uv3d, w3d, w_mesh3d, w_mesh_surf3d,
                            #dw_mesh_dz_3d, bathymetry3d, z_coord_ref3d)
    # Move to next time step
    t += dt
    i += 1

    # Write the solution to file
    if t >= next_export_t - T_epsilon:
        cputime = timeMod.clock() - cputimestamp
        cputimestamp = timeMod.clock()
        norm_h = norm(solution2d.split()[1])
        norm_u = norm(solution2d.split()[0])

        Vol = compVolume2d(solution2d.split()[1], swe2d.dx)
        Vol3d = compVolume3d(dx)
        Mass3d = compTracerMass3d(salt3d, dx)
        saltMin = salt3d.dat.data.min()
        saltMax = salt3d.dat.data.max()
        saltMin = op2.MPI.COMM.allreduce(saltMin, op=MPI.MIN)
        saltMax = op2.MPI.COMM.allreduce(saltMax, op=MPI.MAX)
        uvAbsMax = np.hypot(uv3d.dat.data[:, 0], uv3d.dat.data[:, 1]).max()
        uvAbsMax = op2.MPI.COMM.allreduce(uvAbsMax, op=MPI.MAX)
        if commrank == 0:
            line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                    'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
            print(bold(line.format(iexp=iExp, i=i, t=t, e=norm_h,
                              u=norm_u, cpu=cputime)))
            line = 'Rel. {0:s} error {1:11.4e}'
            print(line.format('vol  ', (Vol_0 - Vol)/Vol_0))
            print(line.format('vol3d', (Vol3d_0 - Vol3d)/Vol3d_0))
            print(line.format('mass ', (Mass3d_0 - Mass3d)/Mass3d_0))
            print('salt deviation {:g} {:g}'.format(saltMin-4.5, saltMax-4.5))
            sys.stdout.flush()
        U_2d_file.export(solution2d.split()[0])
        eta_2d_file.export(solution2d.split()[1])
        eta_3d_file.export(eta3d)
        uv_3d_file.export(uv3d)
        w_3d_file.export(w3d)
        w_mesh_3d_file.export(w_mesh3d)
        salt_3d_file.export(salt3d)
        uv_dav_2d_file.export(uv2d_dav)
        uv_bot_2d_file.export(uv_bottom2d)
        visc_3d_file.export(viscosity_v3d)

        next_export_t += TExport
        iExp += 1

        #if commrank == 0:
            #labels = ['mode2d', 'momentumEq', 'vert_diffusion',
                      #'continuityEq', 'saltEq', 'aux_functions']
            #cost = {}
            #relcost = {}
            #totcost = 0
            #for label in labels:
                #value = timing(label, reset=True)
                #cost[label] = value
                #totcost += value
            #for label in labels:
                #c = cost[label]
                #relcost = c/totcost
                #print '{0:25s} : {1:11.6f} {2:11.2f}'.format(label, c, relcost)
                #sys.stdout.flush()
