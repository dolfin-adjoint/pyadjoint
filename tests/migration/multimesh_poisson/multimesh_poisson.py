from fenics import *
from fenics_adjoint import *
#import moola



class Noslip(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


def solve_poisson():
    # Creating two intersecting meshes
    #mesh_0 = RectangleMesh(Point(-1.5, -0.75), Point(1.5, 0.75), 40, 20)
    mesh_0 = RectangleMesh(Point(0,0), Point(1,1), 20, 20) 
    mesh_1 = RectangleMesh(Point(0.55,0.1),  Point(0.9,0.85), 25 ,40)
    # mesh_2 = RectangleMesh(Point(-1.5,-1.5),  Point(0,0),10,10)
    multimesh = MultiMesh()
    multimesh.add(mesh_0)
    multimesh.add(mesh_1)
    # multimesh.add(mesh_2)
    multimesh.build()

    # Create function space for the temperature
    V = MultiMeshFunctionSpace(multimesh, "Lagrange", 1)

    # Create a function space for control-function f
    #W = MultiMeshFunctionSpace(multimesh, 'DG', 0)

    # Define intial guess for controll-function f (This could be zero)
    # Issue 675: We can't interpolate f into a multimeshfunctionspace
    #f = interpolate(Expression('x[0]+x[1]', degree=1), W, name='Control')
    # f = Constant(1)


    # Define trial and test functions and right-hand side
    u = TrialFunction(V)
    v = TestFunction(V)


    # Define facet normal and mesh size
    n = FacetNormal(multimesh)
    h = 2.0*Circumradius(multimesh)
    h = (h('+') + h('-')) / 2

    # Set parameters
    alpha = 4.0
    beta = 4.0

    # Define bilinear form,
    a = dot(grad(u), grad(v))*dX \
        - dot(avg(grad(u)), jump(v, n))*dI \
        - dot(avg(grad(v)), jump(u, n))*dI \
        + alpha/h*jump(u)*jump(v)*dI \
        + beta*dot(jump(grad(u)), jump(grad(v)))*dO # \
        # - (f*v)*dX
    # Define linear form
    L= Constant(1)*v*dX
    # L = Constant(0)*v*dX
    # Assemble linear system
    A = assemble_multimesh(a)
    b = assemble_multimesh(L)

    noslip=Noslip()
    bc0 = MultiMeshDirichletBC(V, Constant(0), noslip)

    bc0.apply(A,b)
    u = MultiMeshFunction(V)
    solve(A, u.vector(), b)
    
    # adj_html("forward.html", "forward")
    # domains = CellFunction("size_t", multimesh)
    J = Functional(u*u*u*dX)
    c = Control(u)

    #assemble_multimesh(u*v*dX)
    q= compute_gradient(J, c)
    plot(q.part(0), title='der0')
    plot(q.part(1), title='der1')
    plot(u.part(0), title='u0')
    plot(u.part(1), title='u1')
    interactive()
    # plot(multimesh)
    # interactive()



    
if __name__ == '__main__':
    solve_poisson()

