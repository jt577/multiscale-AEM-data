import numpy as np
import torch

def make_grid(L, N):
    """
    Build real‐space grid r with shape (N, N, N, 3) over [0,L)^3.
    """
    coords = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    return np.stack((X, Y, Z), axis=-1)  # (N,N,N,3)

def lamella(r, period, phase=0.0):
    """Single‐mode lamella along x with spacing `period`."""
    x = r[...,0]
    return np.cos(2*np.pi*x/period + phase)

def cylinders(r, period, phase=0.0):
    """Hexagonal cylinders in the xy-plane with nearest‐neighbor spacing `period`."""
    x, y = r[...,0], r[...,1]
    G = 2*np.pi/period
    modes = [(1,0), (1/2,np.sqrt(3)/2), (-1/2,np.sqrt(3)/2),
                (-1,0), (-1/2,-np.sqrt(3)/2), (1/2,-np.sqrt(3)/2)]
    field = np.zeros(r.shape[:-1])
    for (ux,uy) in modes:
        field += np.cos(G*(ux*x + uy*y) + phase)
    return field

def spheres(r, period, phase=0.0):
    """Body‐centered cubic spheres (8 modes) with spacing `period`."""
    G = 2*np.pi/period
    dirs = [( 1, 1, 1), ( 1, 1,-1), ( 1,-1, 1), ( 1,-1,-1),
            (-1, 1, 1), (-1, 1,-1), (-1,-1, 1), (-1,-1,-1)]
    field = np.zeros(r.shape[:-1])
    for (ux,uy,uz) in dirs:
        field += np.cos(G*(ux*r[...,0] + uy*r[...,1] + uz*r[...,2]) + phase)
    return field

def gyroid(r, period):
    """
    Double‐gyroid analytic form (Ia-3d).
    f = sin(2πx/a)cos(2πy/a) + sin(2πy/a)cos(2πz/a) + sin(2πz/a)cos(2πx/a)
    """
    a = period
    x, y, z = r[...,0], r[...,1], r[...,2]
    return (
        np.sin(2*np.pi*x/a)*np.cos(2*np.pi*y/a)
        + np.sin(2*np.pi*y/a)*np.cos(2*np.pi*z/a)
        + np.sin(2*np.pi*z/a)*np.cos(2*np.pi*x/a)
    )

def make_seed(morphology, num_phases, L, N, amplitude=1e-2, period=None, noise_amp=1e-3):
    """
    Build flattened torch.Tensor initial guess for an ABC triblock SCFT.
    
    morphology: 'lamella', 'cylinders', 'spheres', or 'gyroid'
    L         : physical box length
    N         : number of grid points per dim
    amplitude : seed amplitude
    period    : periodicity of structure (defaults to L for lamella/cyl, L/2 for gyroid)
    noise_amp : amplitude of small random noise
    """
    if period is None:
        period = L if morphology!='gyroid' else L/2
    
    # build grid
    r = make_grid(L, N)
    
    # choose field generator
    if   morphology == 'lamella':
        f = lamella(r, period)
    elif morphology == 'cylinders':
        f = cylinders(r, period)
    elif morphology == 'spheres':
        f = spheres(r, period)
    elif morphology == 'gyroid':
        f = gyroid(r, period)
    else:
        raise ValueError(f"Unknown morphology '{morphology}'")
    
    x0 = amplitude * f[None, ...] + noise_amp * np.random.uniform(-1, 1, size=(num_phases, N, N, N))

    x0 = x0.reshape(-1)
    return torch.tensor(x0, dtype=torch.float64)