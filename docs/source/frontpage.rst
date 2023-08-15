:orphan:

.. title:: Automating the derivation of adjoint and tangent linear models

The dolfin-adjoint project automatically derives the discrete adjoint
and tangent linear models from a forward model written in the Python
interface to DOLFIN.

.. raw:: html

   <style type="text/css">
       .codeblock {
         display: inline-block;
           width: 500px;
       }
       .imageblock {
         display: inline-block;
           width: 260px;
           margin: 1em;
       }
   </style>

   <div id="slider" class="flexslider">
     <ul class="slides">

     <!-- -------------------- Klein bottle -------------------------------- -->
       <li>
         <h2>
         Sensitivity analysis on a Klein bottle.
         </h2>
         <div class="codeblock">
               <pre><code class="f1 python"># Define variational formulation
 F = inner(grad(u), grad(v))*dx - inner(m, v)*dx 

 # Solve the variational form
 solve(F == 0, u)

 # Compute the sensitivity 
 J = Functional(inner(u, u)*dx)
 m = Control(f)
 dJdm = compute_gradient(J, m)

 # Plot the results
 plot(dJdm)</code></pre>
           </div>
           <div class="imageblock">
               <img src="_static/slider/klein.png" style="width:256px;height=256px"/>
           </div>
       </li>

     <!-- -------------------- Poisson topology -------------------------------- -->
       <li>
         <h2>
         Growing the optimal heat-sink. 
         </h2>
         <div class="codeblock">
               <pre><code class="f1 python"># Define variational formulation
 F = inner(grad(v), k(a)*grad(T))*dx - f*v*dx

 # Specify control and compliance as objective
 J = Functional(f*T*dx)
 m = Control(a)
 Jhat = ReducedFunctional(J, m)

 # Run optimization
 constraint = VolumeConstraint(V=0.4)
 nlp = rfn.ipopt_problem(bounds=(lb, ub),
                  constraints=constraint)
 a_opt = nlp.solve(full=False)</code></pre>
           </div>
           <div class="imageblock">
               <img src="_static/slider/poisson-topology.png" style="width:256px;height=256px"/>
           </div>
       </li>

     </ul>
   </div>
