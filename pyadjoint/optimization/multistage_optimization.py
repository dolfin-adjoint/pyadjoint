from dolfin import info_green, refine
from .optimization import minimize, maximize


def minimize_multistage(rf, coarse_mesh, levels):
    ''' Implements the MG/Opt multistage approach; a multigrid algorithym with a V-cycle templage for traversing the grids
    '''
    # Create the meshes
    meshes = [coarse_mesh]
    for l in range(levels - 1):
        meshes.append(refine(meshes[-1]))

    # Create multiple approximations of the reduced functional
    rfs = [rf]
    for l in range(levels - 1):
        rfs.append()

def mg_opt(rf, meshes, current_mesh_idx):
    if current_mesh_idx == len(meshes) - 1:
        info_green("Solve problem on coarsest grid")
        m_h_p1 = minimize(rf)
    else:
        m_h1 = minimize(rf, options = {"maxiter": 1})
