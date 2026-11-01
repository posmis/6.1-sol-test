"""Cell-centred Poisson solve with Neumann inlet and pressure-open boundaries.

Rectangles use separable FFT transforms, O(N log N) storage O(N). Large masked
domains use CG with the rectangle inverse as SPD preconditioner, without LU fill-in.
"""
import numpy as np
from scipy.fft import dct, idct, dst, idst
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized, cg, LinearOperator


class PressureSolver:
    DIRECT_LIMIT = 20000

    def __init__(self, fluid, dx, dy, bottom, top):
        self.fluid, self.dx, self.dy = fluid, dx, dy
        self.cancel = None
        self.iterations = 0
        ny, nx = fluid.shape
        b, t = bottom == "open", top == "open"
        if b and t:
            self.y_forward, self.y_inverse, self.y_type, offset = dst, idst, 2, 1.
        elif b:
            self.y_forward, self.y_inverse, self.y_type, offset = dst, idst, 4, .5
        elif t:
            self.y_forward, self.y_inverse, self.y_type, offset = dct, idct, 4, .5
        else:
            self.y_forward, self.y_inverse, self.y_type, offset = dct, idct, 2, 0.
        lx = 4*np.sin(np.pi*(np.arange(nx)+.5)/(2*nx))**2/dx**2
        ly = 4*np.sin(np.pi*(np.arange(ny)+offset)/(2*ny))**2/dy**2
        self.eigenvalues = ly[:, None]+lx[None, :]
        if fluid.all():
            self.method = "spectral"
            return
        ids = np.full(fluid.shape, -1, int)
        ids[fluid] = np.arange(fluid.sum())
        rows, cols, values = [], [], []
        diag = np.zeros(fluid.sum())
        for axis, h in ((0, dy), (1, dx)):
            a, z = [slice(None)]*2, [slice(None)]*2
            a[axis], z[axis] = slice(None, -1), slice(1, None)
            a, z = tuple(a), tuple(z)
            ok = fluid[a] & fluid[z]
            i, j = ids[a][ok], ids[z][ok]
            for row, col in ((i, j), (j, i)):
                rows.append(row); cols.append(col); values.append(np.full(len(row), -1/h**2))
                np.add.at(diag, row, 1/h**2)
        diag[ids[:, -1][fluid[:, -1]]] += 2/dx**2
        if b:
            diag[ids[0][fluid[0]]] += 2/dy**2
        if t:
            diag[ids[-1][fluid[-1]]] += 2/dy**2
        rows.append(np.arange(len(diag))); cols.append(np.arange(len(diag))); values.append(diag)
        self.matrix = coo_matrix((np.concatenate(values),
            (np.concatenate(rows), np.concatenate(cols))), shape=(len(diag),)*2).tocsr()
        if len(diag) <= self.DIRECT_LIMIT:
            self.method = "sparse-lu"
            self.direct = factorized(self.matrix.tocsc())
        else:
            self.method = "cg-spectral"
            self.preconditioner = LinearOperator(self.matrix.shape, matvec=self._precondition,
                                                  dtype=np.float64)
            self.previous = np.zeros(len(diag))

    def rectangle(self, rhs):
        q = dct(rhs, type=4, axis=1, norm="ortho")
        q = self.y_forward(q, type=self.y_type, axis=0, norm="ortho")
        q /= self.eigenvalues
        q = self.y_inverse(q, type=self.y_type, axis=0, norm="ortho")
        return idct(q, type=4, axis=1, norm="ortho")

    def _precondition(self, rhs):
        full = np.zeros(self.fluid.shape)
        full[self.fluid] = rhs
        return self.rectangle(full)[self.fluid]

    def __call__(self, rhs):
        if self.method == "spectral":
            return self.rectangle(rhs.reshape(self.fluid.shape)).ravel()
        if self.method == "sparse-lu":
            return self.direct(rhs)
        self.iterations = 0
        def callback(_):
            self.iterations += 1
            if self.cancel is not None and self.cancel.is_set():
                raise InterruptedError("Calculation cancelled")
        solution, status = cg(self.matrix, rhs, x0=self.previous, M=self.preconditioner,
                              rtol=1e-10, atol=1e-12, maxiter=1000, callback=callback)
        if status != 0:
            raise RuntimeError(f"Pressure CG did not converge (status={status}); try a coarser grid")
        self.previous = solution
        return solution
