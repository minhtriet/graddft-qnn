from pyscf import gto, scf
from pyscf.tools import cubegen

mol = gto.M(
    atom="""O 0.00000000,  0.000000,  0.000000
            H 0.761561, 0.478993, 0.00000000
            H -0.761561, 0.478993, 0.00000000""",
    basis="6-31g*",
)
mf = scf.RHF(mol).run()
cubegen.density(mol, "h2o_den.cube", mf.make_rdm1())
cubegen.orbital(mol, "h2o_molecular_orbital.cube", mf.mo_coeff[:, 0])
