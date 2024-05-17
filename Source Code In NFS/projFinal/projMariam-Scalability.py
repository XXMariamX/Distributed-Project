import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from mpi4py import MPI
import os
import math
import zlib
import io
import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Global variables
IMAGE = np.zeros((1, 1, 1))
IMAGELIST = []
all_labels = []

# Function to apply image processing operations
def apply_image_processing(image, operation):
    # Same implementation as before
    pass

# Function to open file dialog and load image
def open_files():
    files = filedialog.askopenfilenames()
    global IMAGELIST
    IMAGELIST = []
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
    j = 0
    if not os.path.exists("batch"):
        os.makedirs("batch")
    for i in IMAGELIST:
        path = "batch/result" + str(j) + ".png"
        j += 1
        cv2.imwrite(path, i)
    print("Image Batch saved successfully!")

def display_image(image):
    # Same implementation as before
    pass

def display_images_from_files(files):
    # Same implementation as before
    pass

def display_images():
    # Same implementation as before
    pass

# Function to distribute workload for image processing
def distribute_workload_image(operation):
    global IMAGE
    height, width, channels = IMAGE.shape

    # Calculate workload per node
    workload_per_node = math.ceil(height / (size - 1))

    # Compress image
    _, img_buffer = cv2.imencode('.png', IMAGE)
    compressed_image = zlib.compress(img_buffer)

    # Send workload to worker nodes
    start = 0
    for i in range(1, size):
        end = min(start + workload_per_node, height)
        try:
            comm.send((compressed_image, start, end, operation), dest=i)
        except MPI.Exception as e:
            print(f"Error while sending data to node {i}: {e}")
            redistribute_workload_image(operation, start, end)  # Resend the task to another node
        start = end

    # Receive processed image parts from worker nodes and merge them
    for _ in range(1, size):
        try:
            result, start, end = comm.recv()
            IMAGE[start:end] = result
        except MPI.Exception as e:
            print(f"Error while receiving data from a worker node: {e}")
            # Handle error and attempt to recover or redistribute workload
            redistribute_workload_image(operation, start, end)  # Resend the task to another node

# Function to redistribute workload for image processing
def redistribute_workload_image(operation, start, end):
    global IMAGE
    try:
        # Find another available worker node
        for i in range(1, size):
            if i != rank:
                comm.send((True,), dest=i)  # Notify the node to prepare for receiving the task
                _, img_buffer = cv2.imencode('.png', IMAGE[start:end])
                compressed_image = zlib.compress(img_buffer)
                comm.send((compressed_image, start, end, operation), dest=i)  # Send the task
                result, _, _ = comm.recv(source=i)  # Receive the processed image
                IMAGE[start:end] = result
                return
    except MPI.Exception as e:
        print(f"Error while redistributing workload: {e}")
        # Handle error and attempt to recover or exit gracefully

# Function to distribute workload for batch image processing
def distribute_workload_batch(operation):
    global IMAGELIST

    # Compress images
    compressed_images = []
    for img in IMAGELIST:
        _, img_buffer = cv2.imencode('.png', img)
        compressed_images.append(zlib.compress(img_buffer))

    # Calculate workload per node
    workload_per_node = math.ceil(len(IMAGELIST) / (size - 1))

    # Send workload to worker nodes
    start = 0
    for i in range(1, size):
        end = min(start + workload_per_node, len(IMAGELIST))
        try:
            comm.send((compressed_images[start:end], start, end, operation), dest=i)
        except MPI.Exception as e:
            print(f"Error while sending data to node {i}: {e}")
            redistribute_workload_batch(operation, start, end)  # Resend the task to another node
        start = end

    # Receive processed batch parts from worker nodes and merge them
    for _ in range(1, size):
        try:
            result, start, end = comm.recv()
            IMAGELIST[start:end] = result
        except MPI.Exception as e:
            print(f"Error while receiving data from a worker node: {e}")
            # Handle error and attempt to recover or redistribute workload
            redistribute_workload_batch(operation, start, end)  # Resend the task to another node

# Function to redistribute workload for batch image processing
def redistribute_workload_batch(operation, start, end):
    global IMAGELIST
    try:
        # Find another available worker node
        for i in range(1, size):
            if i != rank:
                comm.send((True,), dest=i)  # Notify the node to prepare for receiving the task
                compressed_images = []
                for img in IMAGELIST[start:end]:
                    _, img_buffer = cv2.imencode('.png', img)
                    compressed_images.append(zlib.compress(img_buffer))
                comm.send((compressed_images, start, end, operation), dest=i)  # Send the task
                result, _, _ = comm.recv(source=i)  # Receive the processed images
                IMAGELIST[start:end] = result
                return
    except MPI.Exception as e:
        print(f"Error while redistributing workload: {e}")
        # Handle error and attempt to recover or exit gracefully

# Main image processing function
def process_image():
    operation = selected_operation.get()
    if rank == 0:
        print(' -------- Distributing Processing of One Image -------- ')
        distribute_workload_image(operation)
        print("Master Done Merging Results")
        display_image(IMAGE)
    else:
        while True:
            try:
                compressed_img, start, end, operation = comm.recv(source=0)
                img_buffer = zlib.decompress(compressed_img)
                img = cv2.imdecode(np.frombuffer(img_buffer, np.uint8), cv2.IMREAD_COLOR)
                result = apply_image_processing(img, operation)
                comm.send((result, start, end), dest=0)
            except MPI.Exception as e:
                print(f"Error in worker node {rank}: {e}")
                redistribute_workload_image(operation, start, end)  # Resend the task to another node

# Main batch image processing function
def process_batch():
    operation = selected_operation.get()
    if rank == 0:
        print(' -------- Distributing Batches -------- ')
        distribute_workload_batch(operation)
        print("Master Done Merging Batches")
        display_images()
    else:
        while True:
            try:
                compressed_images, start, end, operation = comm.recv(source=0)
                images = []
                for compressed_img in compressed_images:
                    img_buffer = zlib.decompress(compressed_img)
                    img = cv2.imdecode(np.frombuffer(img_buffer, np.uint8), cv2.IMREAD_COLOR)
                    images.append(img)
                result = [apply_image_processing(img, operation) for img in images]
                comm.send((result, start, end), dest=0)
            except MPI.Exception as e:
                print(f"Error in worker node {rank}: {e}")
                redistribute_workload_batch(operation, start, end)  # Resend the task to another node

if _name_ == "_main_":
    if rank == 0:
        # Create main GUI window
        root = tk.Tk()
        root.title("Image Processing")

        # Create frame for buttons
        button_frame = tk.Frame(root)
        button_frame.grid(row=0, column=0)

        # Create button to open file dialog
        open_button = tk.Button(button_frame, text="Open Image", command=open_file)
        open_button.grid(row=0, column=1)

        # Create button to download processed image
        upload_Imgs_button = tk.Button(button_frame, text="Upload Batch of Images", command=open_files)
        upload_Imgs_button.grid(row=1, column=1)

        prcs_button = tk.Button(button_frame, text="Process Image", command=process_image)
        prcs_button.grid(row=0, column=2)

        prcs_batch_button = tk.Button(button_frame, text="Process Batch of Images", command=process_batch)
        prcs_batch_button.grid(row=1, column=2)

        # Create button to download processed image
        download_button = tk.Button(button_frame, text="Download Image", command=save_file)
        download_button.grid(row=0, column=3)

        download_button = tk.Button(button_frame, text="Download Batch", command=save_batch)
        download_button.grid(row=1, column=3)

        # Create dropdown menu for selecting operations
        operations = ['blur', 'gaussian', 'median', 'bilateral', 'canny', 'invert', 'brightness_increase', 'to_gray',
                      'to_bw', 'contrast_stretching']
        selected_operation = tk.StringVar(root)
        selected_operation.set(operations[0])
        operation_menu = tk.OptionMenu(button_frame, selected_operation, *operations)
        operation_menu.grid(row=0, column=4)

        # Create canvas to display images
        canvas = tk.Canvas(root, width=500, height=500)
        canvas.grid(row=1, column=0)

        canvas_image = canvas.create_image(0, 0, anchor=tk.NW, image=None)  # Initialize canvas image

        root.mainloop()
    else:
        if rank == 1:
            process_image()
        else:
            process_batch()
