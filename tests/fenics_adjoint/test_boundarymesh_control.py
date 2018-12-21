from dolfin import *
from dolfin_adjoint import *
set_log_level(LogLevel.ERROR)

def test_h1_smoothing():
    # Creating mesh and boundary mesh
    mesh = UnitSquareMesh(10,10)
    b_mesh = BoundaryMesh(mesh, "exterior")

    # Creating the control vector space
    V_b = VectorFunctionSpace(b_mesh, "CG", 1)
    s = Function(V_b, name="Design")


    # Interpolate values from BoundaryMesh to the full function space
    V = VectorFunctionSpace(mesh, "CG", 1)
    s_full = transfer_from_boundary(s, mesh)
    s_full.rename("Volume extension", "")
    # Solve deformation equation with the boundary values as input
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u),grad(v))*dx + inner(u,v)*dx
    l = inner(s_full, v)*ds
    deform = Function(V, name="Volume Deformation")
    solve(a==l, deform)
    
    # Deform volume mesh
    ALE.move(mesh, deform)
    
    # Solve Poisson problem
    Vs = FunctionSpace(mesh,"CG", 1)
    us, vs = TrialFunction(Vs), TestFunction(Vs)
    a_s = inner(grad(us), grad(vs))*dx
    bc  = DirichletBC(Vs, Constant(1, name="One"), "on_boundary")
    u_out = Function(Vs, name="State")
    x = SpatialCoordinate(mesh)
    f = sin(x[0])*x[1]
    l_s = f*vs*dx
    solve(a_s==l_s, u_out, bcs=bc)
    
    # Define Functional
    J = assemble(inner(grad(u_out),grad(u_out))*dx(domain=mesh))
    Jhat = ReducedFunctional(J, Control(s))
    tape = get_working_tape()
    tape.visualise()
    
    # Define the perturbation direction
    s1 = interpolate(Expression(("A*(x[0]-0.5)", "A*(x[1]-0.5)"),
                                A=10,degree=3),V_b)

    # Compute the derivative and compare length of derivative with the full
    # mesh vectorfunctionspace
    dJdm = Jhat.derivative()
    print("Gradient vector length vs deform length")
    print(len(dJdm.vector().get_local()), len(deform.vector().get_local()))

    # Compute 0th, 1st and  2nd taylor residuals
    rates=taylor_to_dict(Jhat, s, s1)
    print("----H1-mesh smoothing----")
    print("FD residuals")
    print(rates["FD"]["Residual"])
    print("Derivative residuals")
    print(rates["dJdm"]["Residual"])
    print("Hessian rate")
    print(rates["Hm"]["Residual"])

    print("FD rates")
    print(rates["FD"]["Rate"])
    assert(min(rates["FD"]["Rate"])>0.95)
    print("Derivative rates")
    print(rates["dJdm"]["Rate"])
    assert(min(rates["dJdm"]["Rate"])>1.95)
    print("Hessian rate")
    print(rates["Hm"]["Rate"])
    assert(min(rates["Hm"]["Rate"])>2.95)


def test_strong_boundary_enforcement():
    # Creating mesh and boundary mesh
    tape = get_working_tape()
    tape.clear_tape()
    mesh = UnitSquareMesh(10,10)
    b_mesh = BoundaryMesh(mesh, "exterior")

    # Creating the control vector space
    V_b = VectorFunctionSpace(b_mesh, "CG", 1)
    s = Function(V_b, name="Design")

    # Creating the control vector space
    V_b = VectorFunctionSpace(b_mesh, "CG", 1)
    s = Function(V_b, name="Design")

    # Interpolate values from BoundaryMesh to the full function space
    V = VectorFunctionSpace(mesh, "CG", 1)
    s_full = transfer_from_boundary(s, mesh)
    s_full.rename("Volume Extension", "")
    
    # Solve deformation equation with the boundary values as input
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u),grad(v))*dx + inner(u,v)*dx
    bc_s = DirichletBC(V, s_full, "on_boundary")
    deform = Function(V, name="Volume deformation")
    solve(lhs(a)==rhs(a), deform, bcs=bc_s)

    # Deform volume mesh
    ALE.move(mesh, deform)
    
    # Solve Poisson problem
    Vs = FunctionSpace(mesh,"CG", 1)
    us, vs = TrialFunction(Vs), TestFunction(Vs)
    a_s = inner(grad(us), grad(vs))*dx
    bc  = DirichletBC(Vs, Constant(1, name="One"), "on_boundary")
    u_out = Function(Vs, name="State")
    x = SpatialCoordinate(mesh)
    f = sin(x[0])*x[1]
    l_s = f*vs*dx
    solve(a_s==l_s, u_out, bcs=bc)
    
    # Define Functional
    J = assemble(inner(grad(u_out),grad(u_out))*dx(domain=mesh))
    Jhat = ReducedFunctional(J, Control(s))
    
    # Visualize compuational tape
    #tape.visualise("tape_boundary.dot", dot=True)
    
    # Define the perturbation direction
    s1 = interpolate(Expression(("A*(x[0]-0.5)", "A*(x[1]-0.5)"),
                                A=10,degree=3),V_b)
    # Compute the derivative and compare length of derivative with the full
    # mesh vectorfunctionspace
    dJdm = Jhat.derivative()
    print("Gradient vector length vs deform length")
    print(len(dJdm.vector().get_local()), len(s_full.vector().get_local()))
    
    # Compute 0th, 1st and  2nd taylor residuals
    rates=taylor_to_dict(Jhat, s, s1)
    print("---- Dirichlet Boundary movement -----")
    print("FD residuals")
    print(rates["FD"]["Residual"])
    print("Derivative residuals")
    print(rates["dJdm"]["Residual"])
    print("Hessian rate")
    print(rates["Hm"]["Residual"])

    
    print("FD rates")
    print(rates["FD"]["Rate"])
    assert(min(rates["FD"]["Rate"])>0.95)
    print("Derivative rates")
    print(rates["dJdm"]["Rate"])
    assert(min(rates["dJdm"]["Rate"])>1.95)
    print("Hessian rate")
    print(rates["Hm"]["Rate"])
    assert(min(rates["Hm"]["Rate"])>2.95)

if __name__ == "__main__":
    test_h1_smoothing()
    test_strong_boundary_enforcement()
