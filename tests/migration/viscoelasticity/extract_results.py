from __future__ import print_function
from dolfin import *
from paper import Z
import glob

def look_at_adjoints(dirname, var_name):
    adjoints = glob.glob("%s/adjoint*.xml" % dirname)
    adjoints.sort()

    filenames = [f for f in adjoints if var_name in f]
    filenames.sort(reverse=True)
    z = Function(Z)
    equation_nos = [int(f.split("_")[-1][:-4]) for f in filenames]
    equation_nos.sort()
    norm0s = []
    norm1s = []
    for no in equation_nos:
        f = "%s/adjoint_%s_%d.xml" % (dirname, var_name, no)
        print("Treating %s" % f)
        file = File(f)
        file >> z
        (sigma0, sigma1, v, gamma) = z.split()
        norm0s += [norm(sigma0)]
        norm1s += [norm(sigma1)]

    #pylab.figure()
    #pylab.plot(equation_nos, norm0s, '.-')
    #pylab.plot(equation_nos, norm1s, '.-')
    #pylab.show()

    return (equation_nos, norm0s, norm1s)

def look_at_forwards(dirname):

    forwards = glob.glob("%s/forward*.xml" % dirname)

    iterations = [int(f.split("_")[-2]) for f in forwards]
    iterations.sort()
    times = [float(f.split("_")[-1][:-4]) for f in forwards]
    times.sort()
    print(iterations)
    print(times)

    z = Function(Z)
    w = Function(Z.sub(2).collapse())
    norms = []
    pvds = File("%s/forwards.pvd" % dirname)
    vCG1 = VectorFunctionSpace(Z.mesh(), "CG", 1)
    for k in iterations:
        t = times[k]

        file = File("%s/forward_%d_%g.xml" % (dirname, k, t))
        file >> z
        (sigma0, sigma1, v, gamma) = z.split()
        norms += [assemble(inner(sigma0[2,:], sigma0[2,:])*dx)]

        cg_s = project(sigma0[2,:], vCG1)
        print("Saving .vtu at t = %g" % t)
        pvds << cg_s

    return times, norms

import pylab

dirname = "fine-run-paper"
#dirname = "test-results"

# Write some norms to file
times, norms = look_at_forwards(dirname)
forwarddata = open("forwarddata.py", 'w')
forwarddata.write("# Forward goals, J(t) = sigma0[2](t)*sigma0[2](t)*dx \n")
forwarddata.write("times = %r\n" % times)
forwarddata.write("Js = %r\n" % norms)
forwarddata.close()
pylab.plot(times, norms)

equations, n0s, n1s = look_at_adjoints(dirname, "w_3")
adjointdata = open("adjointdata.py", 'w')
adjointdata.write("# Adjoint norms for w_3, ni(t) = norm(sigmai)\n")
adjointdata.write("equations = %r\n" % equations)
adjointdata.write("n0s = %r\n" % n0s)
adjointdata.write("n1s = %r\n" % n1s)
adjointdata.close()
