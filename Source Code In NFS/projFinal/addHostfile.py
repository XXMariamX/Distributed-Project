from mpi4py import MPI
import sys
import subprocess


comm = MPI.COMM_WORLD
name = MPI.Get_processor_name()


#comm.Set_errhandler(MPI.ERRORS_THROW_EXCEPTIONS)
if (name=='Master'):
	rank = comm.Get_rank()
	size = comm.Get_size()
	answer = subprocess.check_output(['./testSlave3']).decode().strip()
	if(answer=='Go'):
	    print('entered')
	    info = MPI.Info.Create()
	    info.Set("add-host","Slave3")
	    childArgs =["addHostfile.py","1"]
	    icomm = MPI.COMM_SELF.Spawn(sys.executable, args=childArgs,maxprocs=1 ,info=info)
	    icomm.send("Test_icomm",dest=0)
	    msg = icomm.recv(source=MPI.ANY_SOURCE,tag=20)
	    print(str(msg)+ "From Slave")
	


if (name =='Slave3'):
	icomm = MPI.Comm.Get_parent()
#	irank = icomm.Get_rank()
#	isize = icomm.Get_size()
	msg = icomm.recv(source=MPI.ANY_SOURCE,tag=0)
	print(msg+"From Master")
	icomm.send(1,dest=0,tag=20)

	

#MPI.Finalize()
#	if rank == 1:
#		child_spawned.Disconnect()
#	else: 	
#		child_spawned.Disconnect()
