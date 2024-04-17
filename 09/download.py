import os
import requests
import threading
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool

def download_image(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)

def download_images_no_parallel(urls, save_folder):
    os.makedirs(save_folder, exist_ok=True)

    for i, url in enumerate(urls):
        save_path = os.path.join(save_folder, f"{i}.jpg")
        download_image(url, save_path)
        
def download_images_threading(urls, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    threads = []

    for i, url in enumerate(urls):
        save_path = os.path.join(save_folder, f"{i}.jpg")
        thread = threading.Thread(target=download_image, args=(url, save_path))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

def download_wrapper(args):
    download_image(*args)

def download_images_multiprocessing(urls, save_folder, num_processes):
    os.makedirs(save_folder, exist_ok=True)
    pool = Pool(processes=num_processes)

    args_list = [(url, os.path.join(save_folder, f"{i}.jpg")) for i, url in enumerate(urls)]
    pool.map(download_wrapper, args_list)

if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) != 4:
        print("Usage: python download.py /path/to/images.txt /path/to/saved/files/ 8")
        sys.exit(1)

    file_path, save_folder, num_processes = sys.argv[1], sys.argv[2], int(sys.argv[3])

    with open(file_path, 'r') as file:
        urls = [line.strip() for line in file]

    print(f"Downloading {len(urls)} images using {num_processes} processes.")

    start_time = time.time()
    download_images_no_parallel(urls, save_folder)
    end_time = time.time()
    execution_time_no_parallel = end_time - start_time
    print(f"Time taken without parallelism: {execution_time_no_parallel} seconds")

    start_time = time.time()
    download_images_threading(urls, save_folder)
    end_time = time.time()
    execution_time_threading = end_time - start_time
    print(f"Time taken with threading: {execution_time_threading} seconds")

    start_time = time.time()
    download_images_multiprocessing(urls, save_folder, num_processes)
    end_time = time.time()
    execution_time_multiprocessing = end_time - start_time
    print(f"Time taken with multiprocessing: {execution_time_multiprocessing} seconds")
    
    implementation_times = {"Without Parallelism": execution_time_no_parallel, "Threading": execution_time_threading, "Multiprocessing": execution_time_multiprocessing }

    fastest_implementation = min(implementation_times, key=implementation_times.get)
    slowest_implementation = max(implementation_times, key=implementation_times.get)

    print(f"Fastest implementation: {fastest_implementation} ({implementation_times[fastest_implementation]} seconds)")
    print(f"Slowest implementation: {slowest_implementation} ({implementation_times[slowest_implementation]} seconds)")