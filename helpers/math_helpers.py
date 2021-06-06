import torch
import torch as ch
import numpy as np

def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def is_PD(x):
    return np.all(np.linalg.eigvals(x) > 0)

def get_nearest_PD(A, nit=10):
    n = A.shape[0]
    W = np.identity(n) 
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk

def nearestPdCholesky(A, svd_dtype=torch.double, **kwargs):
    A1 = A
    if svd_dtype is not None:
        A1 = A1.to(svd_dtype)
    B = (A1 + A1.t()) / 2
    _, s, V = torch.svd(B)
    H = torch.mm(V, s[:, None] * V.t())
    A2 = ((B + H) / 2)
    A3 = ((A2 + A2.t()) / 2).to(A.dtype)
    try:
        return A3 
    except:
        pass
    finfo = torch.finfo(A.dtype)
    spacing = torch.norm(A).clamp(finfo.tiny) * finfo.eps
    I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    k = 1
    while True:
        mineig = torch.min(torch.real(torch.eig(A3)[0]))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
        try:
            return A3 
        except:
            pass
        
def zca_from_cov(cov):
    evals, evecs = torch.symeig(cov.double(), eigenvectors=True)
    zca = torch.mm(torch.mm(evecs, torch.diag
                            (evals.sqrt().clamp(1e-20).reciprocal())),
                   evecs.t()).to(cov.dtype)
    return zca

def zca_whitened_query_key(matrix, k):
    if len(k.shape) == 1:
        return torch.mm(matrix, k[:, None])[:, 0]
    return torch.mm(matrix, k.permute(1, 0)).permute(1, 0)