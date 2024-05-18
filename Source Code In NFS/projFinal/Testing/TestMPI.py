from mpi4py import MPI

# Simulate sending and receiving data between processes
# In a real test, this would simulate actual MPI send/recv
# Master will send hello to slaves and expect to recieve hello + slave name

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name= MPI.Get_processor_name()

if(name=='Master'):	
    for i in range(1, size):
       comm.send('Hello',dest=i)
    for source in range(1,size):
      reply = comm.recv(source=source)
      print(reply)
    if(source ==1 and reply!='Hello back from Slave1'):
          print('Error. Test failed :(')
    elif(source ==2 and reply!='Hello back from Slave2'):
        print('Error. Test failed :(')
    elif(source ==3 and reply!='Hello back from Slave3'):
        print('Error. Test failed :(')
    else:
        print('Test Passed :)');

else:
    msg = comm.recv(source=0)
    comm.send(msg+' back from '+name, dest=0)

