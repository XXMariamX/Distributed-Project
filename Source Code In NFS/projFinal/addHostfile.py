from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#comm.Set_errhandler(MPI.ERRORS_THROW_EXCEPTIONS)

info = MPI.Info.Create()
info.Set("add-host", "slave2")
childArgs = "addHostfile.py"
child_spawned = MPI.COMM_SELF.Spawn(sys.executable, args=childArgs,
#slots=1,
maxprocs=2, info=info)

print(info)
name= MPI.Get_processor_name()
print(name + str(rank))
#	if rank == 1:
#		child_spawned.Disconnect()
#	else: 	
#		child_spawned.Disconnect()
