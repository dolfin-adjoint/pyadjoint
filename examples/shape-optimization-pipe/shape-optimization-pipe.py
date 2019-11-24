# This code is based on the code in "Ham et al, Automated shape differentation in the Unified Form Language"
from fenics import *
from fenics_adjoint import *
set_log_level(100)

mesh = Mesh("mesh/pipe.xml")
boundaries = MeshFunction("size_t", mesh, "mesh/pipe_facet_region.xml")


S = VectorFunctionSpace(mesh, "CG", 1)
displacement = Function(S)
ALE.move(mesh, displacement)

p2 = VectorElement("CG", mesh.ufl_cell(), 2)
p1 = FiniteElement("CG", mesh.ufl_cell(), 1)
element = MixedElement([p2, p1])
Z = FunctionSpace(mesh, element)

z = Function(Z)
u, p = split(z)
test = TestFunction(Z)
v, q = split(test)

nu = 1./400.
e = nu*inner(grad(u), grad(v))*dx - p*div(v)*dx \
    + inner(dot(grad(u), u), v)*dx + div(u)*q*dx

uin = Expression(["6 * (1-x[1])*x[1]", "0"], degree=2)

bcs = [DirichletBC(Z.sub(0), Constant((0, 0)), boundaries, 3),
       DirichletBC(Z.sub(0), Constant((0, 0)), boundaries, 4),
       DirichletBC(Z.sub(0), uin, boundaries, 1)]

volume = Constant(1.) * dx(domain=mesh)
with stop_annotating():
  target_volume = assemble(volume)

# Solve state equation
solve(e==0, z, bcs=bcs)

# Compute functional
c = 0.1
J = assemble(nu * inner(grad(u), grad(u)) * dx)
J += c*(assemble(volume)-target_volume)**2

Jhat = ReducedFunctional(J, Control(displacement))

u, p = z.split(deepcopy=True)
File("u.pvd") << u
File("p.pvd") << p

# Optimization loop
phi = TrialFunction(S)
psi = TestFunction(S)

a_riesz = inner(grad(phi), grad(psi)) * dx
riesz_bcs = [DirichletBC(S, Constant((0,0)), boundaries, i) for i in range(1,4)]

coordinates = mesh.coordinates()
for i in range(25):

    J = Jhat(displacement)
    dJ = Jhat.derivative()

    # Compute Riesz representation
    L_riesz = inner(dJ, psi)*dx
    with stop_annotating():
      solve(a_riesz==L_riesz, dJ, riesz_bcs)

    print("i = %3i; J = %.6f; ||dJ|| = %.6f"
          % (i, J, (assemble(inner(grad(dJ), grad(dJ))*dx)**0.5)))

    File("dJ_{}.pvd".format(i)) << dJ
    
    # Update mesh
    displacement.vector()[:] += -50*dJ.vector()
    
