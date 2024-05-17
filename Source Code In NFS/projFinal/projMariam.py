import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from mpi4py import MPI
import os


IMAGE=np.zeros((1,1,1))
IMAGELIST=[]
all_labels = []


# Function to apply various image processing operations
def apply_image_processing(image, operation):
    if operation == 'blur':
        result = cv2.blur(image, (5, 5))  # Applying Gaussian blur
    elif operation == 'gaussian':
        result = cv2.GaussianBlur(image, (5, 5), 0)  # Applying Gaussian blur
    elif operation == 'median':
        result = cv2.medianBlur(image, 5)  # Applying median blur
    elif operation == 'bilateral':
        result = cv2.bilateralFilter(image, 9, 75, 75)  # Applying bilateral filter
    elif operation == 'canny':
        result = cv2.Canny(image, 100, 200)  # Applying Canny edge detection
    elif operation == 'invert':
        result = cv2.bitwise_not(image)  # Invert colors
    elif operation == 'brightness_increase':
        result = cv2.convertScaleAbs(image, alpha=1.5, beta=0)  # Increase brightness
    elif operation == 'to_gray':
        result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    elif operation == 'to_bw':
        _, result = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)  # Convert image to black and white
    elif operation == 'contrast_stretching':
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        result = cdf[image]
    else:
        result = image  # No operation
    
    return result

# Function to open file dialog and load image
def open_files():
    files= filedialog.askopenfilenames()
    #print(root.tk.splitlist(files))
    global IMAGELIST
    IMAGELIST=[]
    for file in root.tk.splitlist(files):
    	img = cv2.imread(file)
    	IMAGELIST.append(img)
    	display_images_from_files(files)

    	    	
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
    	global IMAGE
    	IMAGE = cv2.imread(file_path)
    	display_image(IMAGE)

# Function to save processed image
def save_file():
    global IMAGE
    cv2.imwrite("result.png", IMAGE)
    print("Image saved successfully!")

def save_batch():
    global IMAGELIST
    j=0
    if not os.path.exists("batch"):
        os.makedirs("batch")
    for i in IMAGELIST:
         path="batch/result"+str(j)+".png"
         j+=1
         cv2.imwrite(path, i)
    print("Image Batch saved successfully!")
    
def display_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image_tk = ImageTk.PhotoImage(image)
    
    for label in all_labels:
        label.destroy()

    i=0
    label = tk.Label(image=image_tk)
    label.photo = image_tk   # assign to class variable to resolve problem with bug in `PhotoImage`
        
    label.grid(row=2, column=i)
    all_labels.append(label)

    
def display_images_from_files(files):
    for label in all_labels:
        label.destroy()
    i=0
    for file in files:
        image = ImageTk.PhotoImage(Image.open(file))
        label = tk.Label(image=image)
        label.photo = image   # assign to class variable to resolve problem with bug in `PhotoImage`
        
        label.grid(row=2, column=i)
        i+=1      
        all_labels.append(label)


 
def display_images():#display IMGLIST
    global IMAGELIST
     
    for label in all_labels:
        label.destroy()
    i=0
    for image in IMAGELIST:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)
        label = tk.Label(image=image_tk)
        label.photo = image_tk  
        
        label.grid(row=2, column=i)
        i+=1      
        all_labels.append(label)


    
# Function to display image in GUI
def prcs_image():
    print(' -------- Distributing Processing of One Image -------- ')
    global IMAGE
    height, width, channels = IMAGE.shape
    #print(height,width,channels)
    workloads = [ (height // (size-1)) for i in range(size-1) ]
    #print(workloads)
    for i in range( height % size ):
      	workloads[i] += 1

    #print(workloads)
    start = 0
    operation = selected_operation.get()

    for i in range(1, size):
        if i!=1:
            start +=workloads[i-1]
        #print(start+" "+rank)
        end = start + workloads[i-1]
        print("Master Assigned Rank/VM: ",i, "Rows Starting From: ",start, " To: ",end)
        comm.send((IMAGE[start:end][:][:],start,end,operation),dest=i)

    	

    for source in range(1,size):
      	    result,start,end = comm.recv(source=source)
      	    print("Master with rank ",rank,"Recieved From",source,"Rows Starting From: ",start, " To: ",end)
      	    IMAGE[start:end][:][:]=result[:][:][:]
    print("Master Done Merging Results") 	
    '''cv2.imshow("prcsd.png",IMAGE)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    display_image(IMAGE)
    

def prcs_batch():
    print(' -------- Distributing Batches -------- ')
    global IMAGELIST
    workloads = [ (len(IMAGELIST)) // (size-1) for i in range(size-1) ]
    for i in range( len(IMAGELIST) % (size-1) ):
        workloads[i] += 1
    start = 0
    end = 0
    operation = selected_operation.get()

    for i in range(1, size):
        if i!=1:
            start =end
        #print(start+" "+rank)
        end = start + workloads[i-1]
        print("Master Assigned Rank/VM: ",i, "Batch Starting From: ",start, " To: ",end)
        comm.send((IMAGELIST[start:end][:][:],start,end,operation),dest=i)
    	
        
    for source in range(1,size):
      	result,start,end = comm.recv(source=source)
      	print("Master with rank ",rank,"Recieved From",source,"Batch Starting From: ",start, " To: ",end)
      	IMAGELIST[start:end]=result
    print("Master Done Merging Batches") 	
    display_images()
    '''cv2.imshow("prcsd.png",IMAGE)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    display_image(IMAGE)'''
  
    
# MPI CODE
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name= MPI.Get_processor_name()

if name == 'Master':   # Master node
# Create main GUI window
    root = tk.Tk()
    root.title("Image Processing")

# Create frame for buttons
    button_frame = tk.Frame(root)
    button_frame.grid(row=0,column=0)

# Create button to open file dialog
    open_button = tk.Button(button_frame, text="Open Image", command=open_file)
    open_button.grid(row=0, column=1)
    
# Create button to download processed image
    upload_Imgs_button = tk.Button(button_frame, text="Uplaod Batch of Images", command=open_files)
    upload_Imgs_button.grid(row=1, column=1)
    

    prcs_button = tk.Button(button_frame, text="Process Image", command=prcs_image)
    prcs_button.grid(row=0, column=2)
    
    prcs_batch_button = tk.Button(button_frame, text="Process Batch of Images",command=prcs_batch)
    prcs_batch_button.grid(row=1, column=2)
    
    
# Create button to download processed image
    download_button = tk.Button(button_frame, text="Download Image", command=save_file)
    download_button.grid(row=0, column=3)
    
    download_button = tk.Button(button_frame, text="Download Batch", command=save_batch)#!!!!!add cmd
    download_button.grid(row=1, column=3)
    


# Create dropdown menu for selecting operations
    operations = ['blur', 'gaussian', 'median', 'bilateral', 'canny', 'invert', 'brightness_increase', 'to_gray', 'to_bw', 'contrast_stretching']
    selected_operation = tk.StringVar(root)
    selected_operation.set(operations[0])
    operation_menu = tk.OptionMenu(button_frame, selected_operation, *operations)
    operation_menu.grid(row=0, column=4)

	# Create canvas to display images
    canvas = tk.Canvas(root, width=500, height=500)
    canvas.grid(row=1, column=0)

    canvas_image = canvas.create_image(0, 0, anchor=tk.NW, image=None)  # Initialize canvas image

    root.mainloop()
    
else: # Worker nodes
 
    #img = img[height//][:][:]	
    while True:
        img,start,end,operation = comm.recv(source=0)
        
        if (type(img)==type([])): # Batch Images
            print("Rank/VM: ",rank, "Batch Rows Starting From: ",start, " To: ",end)
            result=[]
            j=0
            for i in img:
                print("Rank/VM: ",rank, "Finished: ",j, " /",end-start)
                j+=1
                prcsd= apply_image_processing(i,operation)
                if operation in ['to_bw','to_gray','canny']:
                    prcsd = cv2.cvtColor(prcsd, cv2.COLOR_GRAY2BGR)
                result.append(prcsd)

                
                #cv2.imshow(str(rank)+"-"+str(i)+".png",i)
                #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            print("Rank/VM: ",name, "Finished: ",j, " /",end-start)
            comm.send((result,start,end), dest=0)
            print("Rank/VM: ",name, "Finished and Send Batch Starting From: ",start, " To: ",end)
                
        else: #1 Imge
            print("Rank/VM: ",name, "Received Rows Starting From: ",start, " To: ",end)
            result= apply_image_processing(img,operation)
            if operation in ['to_bw','to_gray','canny']:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
           
            comm.send((result,start,end), dest=0)
            print("Rank/VM: ",name, "Finished and Send Rows Starting From: ",start, " To: ",end)
            '''cv2.imshow("prcsd.png",result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''
