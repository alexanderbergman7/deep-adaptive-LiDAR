from scipy.sparse import diags
from scipy.sparse.linalg import cg
MAX_VAL = 255.0
from scipy.sparse import csr_matrix
import numpy as np
import torch

# Fast Bilateral Solver code adapted from The Fast Bilateral Solver, (Jonathan T. Barron, Ben Poole)
class BilateralSolver():
    def __init__(self, sigma_luma=8, sigma_spatial=8, lam=0.0001, device="cpu"):
        self.grid_params = {
            'sigma_luma': sigma_luma,  # Brightness bandwidth
            'sigma_chroma': 1,  # Color bandwidth
            'sigma_spatial': sigma_spatial  # Spatial bandwidth
        }

        self.bs_params = {
            'lam': lam,  # The strength of the smoothness parameter
            'A_diag_min': 1e-5,  # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
            'cg_tol': 1e-5,  # The tolerance on the convergence in PCG
            'cg_maxiter': 25  # The number of PCG iterations
        }
        self.device = device

    def solve(self, points, depth_gt, monocular_depth):
        points = points.cpu().numpy()
        depth_gt = depth_gt.cpu().numpy()
        monocular_depth = monocular_depth.cpu().numpy()

        # handle case where points is either a sparse mask, or a list of points
        if points.shape != depth_gt.shape:
            confidence, target = self.generate_tc(points, depth_gt)
        else:
            confidence, target = points, points * depth_gt
        inpainted_depth = self.inpaint_depth(confidence, target, monocular_depth)

        return torch.from_numpy(inpainted_depth).to(self.device), torch.from_numpy(confidence).to(self.device)


    def generate_tc(self, points, depth_gt):
        confidence = np.zeros_like(depth_gt)
        for b in range(points.shape[0]):
            for pt in range(points.shape[1]):
                r = int(points[b, pt, 0] * depth_gt.shape[2])
                c = int(points[b, pt, 1] * depth_gt.shape[3])

                confidence[b, 0, r, c] = 1.0

        target = depth_gt * confidence
        return confidence, target

    def inpaint_depth(self, confidence, target, monocular_depth):
        inpainted_depth = np.zeros_like(target)

        for b in range(monocular_depth.shape[0]):
            reference = monocular_depth[b,0,:,:]
            reference = np.stack([reference, np.zeros_like(reference), np.zeros_like(reference)], axis=2)
            t = target[b,0,:,:].reshape(-1, 1).astype(np.double)
            c = confidence[b,0,:,:].reshape(-1, 1).astype(np.double)
            grid = self.BilateralGrid(reference, **self.grid_params)
            inpainted_depth[b,0,:,:] = self.BilateralSolver(grid, self.bs_params).solve(t, c).reshape((monocular_depth.shape[2], monocular_depth.shape[3]))

        return inpainted_depth

    class BilateralGrid(object):
        def __init__(self, im, sigma_spatial=32, sigma_luma=8, sigma_chroma=8):
            im_yuv = im
            # Compute 5-dimensional XYLUV bilateral-space coordinates
            Iy, Ix = np.mgrid[:im.shape[0], :im.shape[1]]
            x_coords = (Ix / sigma_spatial).astype(int)
            y_coords = (Iy / sigma_spatial).astype(int)
            luma_coords = (im_yuv[..., 0] / sigma_luma).astype(int)
            chroma_coords = (im_yuv[..., 1:] / sigma_chroma).astype(int)
            coords = np.dstack((x_coords, y_coords, luma_coords, chroma_coords))
            coords_flat = coords.reshape(-1, coords.shape[-1])
            self.npixels, self.dim = coords_flat.shape
            # Hacky "hash vector" for coordinates,
            # Requires all scaled coordinates be < MAX_VAL
            self.hash_vec = (MAX_VAL ** np.arange(self.dim))
            # Construct S and B matrix
            self._compute_factorization(coords_flat)

        def _compute_factorization(self, coords_flat):
            # Hash each coordinate in grid to a unique value
            hashed_coords = self._hash_coords(coords_flat)
            unique_hashes, unique_idx, idx = \
                np.unique(hashed_coords, return_index=True, return_inverse=True)
            # Identify unique set of vertices
            unique_coords = coords_flat[unique_idx]
            self.nvertices = len(unique_coords)
            # Construct sparse splat matrix that maps from pixels to vertices
            self.S = csr_matrix((np.ones(self.npixels), (idx, np.arange(self.npixels))))
            # Construct sparse blur matrices.
            # Note that these represent [1 0 1] blurs, excluding the central element
            self.blurs = []
            for d in range(self.dim):
                blur = 0.0
                for offset in (-1, 1):
                    offset_vec = np.zeros((1, self.dim))
                    offset_vec[:, d] = offset
                    neighbor_hash = self._hash_coords(unique_coords + offset_vec)
                    valid_coord, idx = self.get_valid_idx(unique_hashes, neighbor_hash)
                    blur = blur + csr_matrix((np.ones((len(valid_coord),)),
                                              (valid_coord, idx)),
                                             shape=(self.nvertices, self.nvertices))
                self.blurs.append(blur)

        def _hash_coords(self, coord):
            """Hacky function to turn a coordinate into a unique value"""
            return np.dot(coord.reshape(-1, self.dim), self.hash_vec)

        def splat(self, x):
            return self.S.dot(x)

        def slice(self, y):
            return self.S.T.dot(y)

        def blur(self, x):
            """Blur a bilateral-space vector with a 1 2 1 kernel in each dimension"""
            assert x.shape[0] == self.nvertices
            out = 2 * self.dim * x
            for blur in self.blurs:
                out = out + blur.dot(x)
            return out

        def filter(self, x):
            """Apply bilateral filter to an input x"""
            return self.slice(self.blur(self.splat(x))) / \
                   self.slice(self.blur(self.splat(np.ones_like(x))))

        def get_valid_idx(self, valid, candidates):
            """Find which values are present in a list and where they are located"""
            locs = np.searchsorted(valid, candidates)
            # Handle edge case where the candidate is larger than all valid values
            locs = np.clip(locs, 0, len(valid) - 1)
            # Identify which values are actually present
            valid_idx = np.flatnonzero(valid[locs] == candidates)
            locs = locs[valid_idx]
            return valid_idx, locs

    class BilateralSolver(object):
        def __init__(self, grid, params):
            self.grid = grid
            self.params = params
            self.Dn, self.Dm = self.bistochastize(grid)

        def solve(self, x, w):
            # Check that w is a vector or a nx1 matrix
            if w.ndim == 2:
                assert (w.shape[1] == 1)
            elif w.dim == 1:
                w = w.reshape(w.shape[0], 1)
            A_smooth = (self.Dm - self.Dn.dot(self.grid.blur(self.Dn)))
            w_splat = self.grid.splat(w)
            A_data = diags(w_splat[:, 0], 0)
            A = self.params["lam"] * A_smooth + A_data
            xw = x * w
            b = self.grid.splat(xw)
            # Use simple Jacobi preconditioner
            A_diag = np.maximum(A.diagonal(), self.params["A_diag_min"])
            M = diags(1 / A_diag, 0)
            # Flat initialization
            w_splat[w_splat == 0] = 1e-10
            y0 = self.grid.splat(xw) / w_splat
            yhat = np.empty_like(y0)
            for d in range(x.shape[-1]):
                yhat[..., d], info = cg(A, b[..., d], x0=y0[..., d], M=M, maxiter=self.params["cg_maxiter"],
                                        tol=self.params["cg_tol"])
            xhat = self.grid.slice(yhat)
            return xhat

        def bistochastize(self, grid, maxiter=10):
            """Compute diagonal matrices to bistochastize a bilateral grid"""
            m = grid.splat(np.ones(grid.npixels))
            n = np.ones(grid.nvertices)
            for i in range(maxiter):
                n = np.sqrt(n * m / grid.blur(n))
            # Correct m to satisfy the assumption of bistochastization regardless
            # of how many iterations have been run.
            m = n * grid.blur(n)
            Dm = diags(m, 0)
            Dn = diags(n, 0)
            return Dn, Dm