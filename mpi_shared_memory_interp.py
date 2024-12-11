#!/usr/bin/env python3

### Goals for this project
### 0) interpolate a 3D mesh onto another 3D mesh in an efficient way
### 1) shared memory
###    - We want to refer to mesh objects on some controller
###      process as if they're local
### 2) streamlined MPI comms
###    - We'll likely require communication on 3 different
###      process subsets - intra-node, all controllers and
###      all
### 3) Topology awareness-ish
###    - Start by sharing memory on a single node
###      expand to other hardware units in time
from mpi4py import MPI
import sys
from socket import gethostname
from typing import Callable, List, Union, Optional
from numpy.typing import NDArray
import pyvista as pv
import numpy as np
import math
from multiprocessing import shared_memory
from scipy.spatial import cKDTree
from pathlib import Path
import vtk
import argparse

if sys.version_info.minor < 13:
    from multiprocessing import resource_tracker


# Do some post processing with pyvista
def post_processor(ds):
    surface = pv.Sphere(radius=1.23)
    surface = vtk.vtkSphere()
    surface.SetRadius(1.23)
    clipped = ds.slice_implicit(surface, generate_triangles=True)
    surf = ds.contour([-0.075, 0.075])
    smoothed = surf.smooth_taubin(n_iter=100, pass_band=0.25)
    return pv.merge([clipped, smoothed])


class MPI_setup:
    comm_world = MPI.COMM_WORLD
    world_rank = comm_world.Get_rank()
    world_size = comm_world.Get_size()
    host = gethostname()
    ops: dict[str, Callable] = {}
    node_list: List[str] = []
    node_comm = MPI.COMM_NULL
    node_comm_rank = -1
    node_comm_size = -1
    leader_comm = MPI.COMM_NULL
    leader_comm_rank = -1
    leader_comm_size = -1

    def __init__(self):
        self.setup_mpi()

    def setup_mpi(self) -> None:
        ### How many physical nodes are we on?
        proc_hosts = self.comm_world.allgather(self.host)
        self.node_list = list(dict.fromkeys(proc_hosts))

        node_comm_colour = self.node_list.index(self.host)
        node_comm_rank = self.world_rank - proc_hosts.index(self.host)

        self.node_comm = self.comm_world.Split(node_comm_colour, node_comm_rank)
        self.node_comm_rank = self.node_comm.Get_rank()
        self.node_comm_size = self.node_comm.Get_size()

        ### Now setup a leader comm
        leader_group = MPI.Group(self.comm_world.Get_group())
        leaders = []
        for node in self.node_list:
            leaders.append(proc_hosts.index(node))
        leader_group = leader_group.Incl(leaders)
        self.leader_comm = self.comm_world.Create(leader_group)
        if self.leader_comm != MPI.COMM_NULL:
            self.leader_comm_rank = self.leader_comm.Get_rank()
            self.leader_comm_size = self.leader_comm.Get_size()


class SharedMemoryHandler:
    shm = None

    def __init__(self):
        pass

    @classmethod
    def shared_memory_create(cls, size: int):
        out = cls()
        out.shm = shared_memory.SharedMemory(create=True, size=size)
        return out

    @classmethod
    def shared_memory_attach(cls, fn: str):
        out = cls()
        if sys.version_info.minor < 13:
            out.shm = shared_memory.SharedMemory(name=fn)
        else:
            out.shm = shared_memory.SharedMemory(name=fn, track=False)
        return out

    def __getattr__(self, attr):
        return getattr(self.shm, attr)

    def __del__(self):
        if sys.version_info.minor < 13:
            self.shm.close()
            try:
                self.shm.unlink()
            except FileNotFoundError:
                resource_tracker.unregister(self.shm._name, "shared_memory")
        else:
            self.shm.close()
            if self.shm._track:
                self.shm.unlink()


def known_function(coords: NDArray[np.float64]) -> np.float64:
    r = np.linalg.norm(coords)
    theta = np.atan2(coords[1], coords[0])
    phi = np.atan2(np.linalg.norm(coords[:2]), coords[2])
    return (
        np.sin(np.float64(6.0) * math.pi * r)
        + np.float64(0.2) * np.sin(np.float64(20.0) * theta)
        + 0.1 * np.sin(phi + math.pi / np.float64(2.0))
    )


def get_output_grid(fn: str) -> NDArray[np.float64]:
    ### In this case, we're just going to read a new mesh
    return pv.read(fn)


class InputDataDistributor:
    y: Optional[NDArray[np.float64]] = None
    f: Optional[NDArray[np.float64]] = None
    y_shm: Optional[SharedMemoryHandler] = None
    f_shm: Optional[SharedMemoryHandler] = None

    def __init__(self, fn: Union[str, Path], mpi):
        self.mpi = mpi
        if mpi.world_rank == 0:
            self.model = pv.read(fn)
            ### We need 2 shared memory buffers, one for the grid points and the other for the data
            y_global = np.array(self.model.points)
            print("Source data read")
        else:
            y_global = np.empty(1)

        y_size, y_shape, y_dtype = mpi.comm_world.bcast((y_global.nbytes, y_global.shape, y_global.dtype), root=0)

        if mpi.leader_comm != MPI.COMM_NULL:
            self.y_shm = SharedMemoryHandler.shared_memory_create(y_size)
            self.y = np.ndarray(y_shape, dtype=y_dtype, buffer=self.y_shm.buf)
            print("Shared memory regions created")

            if mpi.leader_comm_rank == 0:
                self.y[:] = y_global[:]  # Copy data
                
            mpi.leader_comm.Bcast([self.y, MPI.DOUBLE], root=0)
            
            print("Source data distributed to remote nodes")

        ### Let everyone on our node find our shared memory location
        
            y_shm_name = self.y_shm.name
        else:
            y_shm_name = None
            f_shm_name = None
        y_shm_name = mpi.node_comm.bcast(y_shm_name, root=0)
        f_shm_name = mpi.node_comm.bcast(f_shm_name, root=0)

        if mpi.node_comm_rank != 0:
            self.y_shm = SharedMemoryHandler.shared_memory_attach(y_shm_name)
            self.y = np.ndarray(y_shape, dtype=y_dtype, buffer=self.y_shm.buf)
            
            print("Shared memory regions attached")

    def distribute_field(self, field: str):
        if mpi.world_rank == 0:
            f_global = np.array(self.model.point_data[field])
        else:
            f_global = np.empty(1)
        f_size, f_shape, f_dtype = self.mpi.comm_world.bcast((f_global.nbytes, f_global.shape, f_global.dtype), root=0)
        if mpi.leader_comm != MPI.COMM_NULL:
            if self.f_shm is None:
                self.f_shm = SharedMemoryHandler.shared_memory_create(f_size)
                self.f = np.ndarray(f_shape, dtype=f_dtype, buffer=self.f_shm.buf)
        
            if mpi.leader_comm_rank == 0:
                self.f[:] = f_global[:]  # Copy data
            mpi.leader_comm.Bcast([self.f, MPI.DOUBLE], root=0)
            f_shm_name = self.f_shm.name
        f_shm_name = mpi.node_comm.bcast(f_shm_name, root=0)
        
        if mpi.node_comm_rank != 0:
            if self.f_shm is None:
                self.f_shm = SharedMemoryHandler.shared_memory_attach(f_shm_name)
                self.f = np.ndarray(f_shape, dtype=f_dtype, buffer=self.f_shm.buf)

                

class SiaInterpolator:
    epsilon_distance = 1e-8

    def __init__(
        self,
        source_grid: NDArray[np.float64],
        source_data: NDArray[np.float64],
        leafsize: int = 16,
        nneighbours: int = 16,
    ):
        self.tree = cKDTree(data=source_grid, leafsize=leafsize)
        self.data = source_data
        self.nneighbours = nneighbours
        self.source_data = source_data

    def __call__(self, target_coords: NDArray[np.float64]):
        dists, idx = self.tree.query(x=target_coords, k=self.nneighbours)

        # Use np.where to avoid division by very small values
        # Replace tiny distances with self.epsilon_distance to avoid division
        # by tiny values while keeping original distances intact for later use
        safe_dists = np.where(dists < self.epsilon_distance, self.epsilon_distance, dists)

        # Then, calculate the weighted average using safe_dists
        weights = self.kernel_weights(safe_dists)
        interped = np.sum(weights * self.source_data[idx], axis=1)

        close_points_mask = dists[:, 0] < self.epsilon_distance
        # Now handle the case where points are too close to each other:
        interped[close_points_mask] = self.source_data[idx[close_points_mask, 0]]
        return interped

    def kernel_weights(self, dist: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.normalise(np.exp(-(dist**2)))
        # return self.normalise(1 / dist)

    def normalise(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.einsum("i, ij->ij", 1 / np.sum(weights, axis=1), weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="mpi_shared_memory_interp.py", description="Interpolates a large pvtu file onto a smaller grid"
    )
    parser.add_argument("-i", "--infile", required=True, help="The input file to interpolate")
    parser.add_argument(
        "-g", "--input_grid", required=True, help="The input file containing the grid to interpolate to"
    )
    parser.add_argument("-o", "--outfile", required=True, help="Path to the final output")
    parser.add_argument("-f", "--field",required=True, help="Name(s) of field to be interpolated, comma separated")

    ns = parser.parse_args(sys.argv[1:])

    fields=ns.field.split(',')

    mpi = MPI_setup()
    input_data = InputDataDistributor(ns.infile, mpi, ns.field)
    ### interp = RBFInterpolator(input_data.y, input_data.f, neighbors=n, smoothing=smoothing, kernel=k, epsilon=eps)
    interp = SiaInterpolator(input_data.y, input_data.f, nneighbours=32)
    print("Interpolant constructed")

    if mpi.world_rank == 0:
        out_model = get_output_grid(ns.input_grid)
        out_grid = np.array(out_model.points)
        downscaled = pv.UnstructuredGrid()
        downscaled.copy_structure(out_model)
        # Distribute outgrid points
        reqs = []
        subgrid_size = len(out_grid) // mpi.world_size
        remainder = len(out_grid) % mpi.world_size
        my_subgrid_size = subgrid_size + (1 if remainder > 0 else 0)
        subgrid = out_grid[:my_subgrid_size]
        end = my_subgrid_size
        for i in range(1, mpi.world_size):
            start = end
            end = start + subgrid_size + (1 if i < remainder else 0)
            reqs.append(mpi.comm_world.isend(out_grid[start:end], dest=i, tag=99))
        _ = MPI.Request.waitall(reqs)
    else:
        subgrid = mpi.comm_world.recv(source=0, tag=99)
    print("Target grid distributed")

    for field in fields:

        input_data.distribute_field(field)
        out_f_part = interp(subgrid)
        out_f_parts = mpi.comm_world.gather(out_f_part, root=0)

        ### Rank 0 from here on out
        if mpi.world_rank == 0:
            out_f = np.empty(len(out_grid), dtype=np.float64)

            end = 0
            for i, part in enumerate(out_f_parts):
                start = end
                end = start + subgrid_size + (1 if i < remainder else 0)
                out_f[start:end] = part

            downscaled[ns.field] = out_f

        # post_processor(downscaled).save("/scratch/xd2/dr4292/postproc_downscaled_l5_surfonly.vtp")
        downscaled.save(ns.outfile)

        print("=================================================================")

    mpi.comm_world.Barrier()

    del input_data
