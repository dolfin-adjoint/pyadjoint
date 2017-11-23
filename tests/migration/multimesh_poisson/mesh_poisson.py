from dolfin import *
from dolfin_adjoint import *

class Noslip(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

def solve_poisson():
    mesh = RectangleMesh(Point(0,0), Point(1,1), 40, 20)
    V = FunctionSpace(mesh, 'Lagrange', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = dot(grad(u), grad(v))*dx
    L = Constant(1)*v*dx

    A = assemble(a)
    b = assemble(L)

    noslip=Noslip()
    bc0 = DirichletBC(V, Constant(0), noslip)
    bc0.apply(A,b)

    u = Function(V)
    solve(A, u.vector(), b)

    adj_html("forward.html", "forward")
    J = Functional(u*u*u*dx)
    c = Control(u)
    q= compute_gradient(J, c)
    plot(u)
    plot(q)
    interactive()


if __name__=='__main__':
    solve_poisson()
