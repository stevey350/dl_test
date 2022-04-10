import ctypes
from configparser import ConfigParser
from ctypes import *
from multiprocessing import Process, Queue
import threading
import time
import numpy as np
import logging

# load C++ dynamic library
lib_ad = cdll.LoadLibrary('./libtest.so')
# lib_ad = ctypes.cdll.LoadLibrary('/workspace/backbone_check/dfr_od/anomaly_detect/build/libanomaly_detect.so')
# lib_ad = ctypes.cdll.LoadLibrary('/workspace/backbone_check/dfr_od/anomaly_detect.1/build/libanomaly_detect.so')
lib_obj = []
ini_file = './ad.ini'                       # configuration file

MAX_GPU_NUM_PER_PROC = 4                    # Maximum number of GPUs for a process
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

class GPUInfo(Structure):
    _fields_ = [("gpu_id", c_uint), ("batch_size", c_uint), ("engine", c_char * 256)]


class GPUArr(Structure):
    _fields_ = [("gpu_num", c_uint), ("gpu_info", GPUInfo * MAX_GPU_NUM_PER_PROC)]


class AnomalyDetection(object):
    def __init__(self, anomalyDetectList):
        logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
        self.process = []
        self.queue = []

        # parameters parse
        self.process_num = len(anomalyDetectList)
        self.rawImage = []
        self.anomalyImage = []
        self.gpuInfos = []
        # self.lib_obj = []
        for ad_task in anomalyDetectList:
            self.rawImage.append(ad_task['rawImage'])
            self.anomalyImage.append(ad_task['anomalyImage'])
            self.gpuInfos.append(ad_task['gpuInfos'])
            lib_obj.append(lib_ad.anomalydetect_new())

        self.cfg_init()
        self.status = 'IDLE'        # process status: IDLE, RUNNING, ERROR, ABORTED, FINISH

    def start(self):
        if self.status == 'RUNNING':
            logging.warning('AnomalyDetection module is running now')
            return False
        else:
            self.status = 'RUNNING'
            self.create_process()
            logging.info('AnomalyDetection module create process OK')
            return True

    def close(self):
        for idx in range(self.process_num):
            lib_ad.anomaly_detect_stop(lib_obj[idx])
            # self.process[idx].kill()

        self.status = "IDLE"

    def get_status(self):
        # 1. Make sure it is in the RUNNING status
        if self.status != 'RUNNING':
            return 0, self.status

        # 2. Getting process status and progress
        progress_list = []
        for idx in range(self.process_num):
            if not self.queue[idx].empty():
                if self.queue[idx].get(block=False) == 'ERROR':
                    self.status = 'ERROR'
                    break
            progress_list.append(lib_ad.anomaly_detect_state(lib_obj[idx]))        # int

        # 3. Handling ERROR case
        if self.status == 'ERROR':
            for idx in range(self.process_num):
                lib_ad.anomaly_detect_stop(lib_obj[idx])
            self.process = []
            self.queue = []
        else:
            progress = int(np.mean(progress_list))
            if progress == 100:
                self.status = 'FINISH'

        return {"progress": progress, "state": self.status}

    def cfg_init(self):
        section_name = 'ad'

        cfg = ConfigParser()
        cfg.read(ini_file)
        self.dataload_num = int(cfg.get(section_name, 'dataload_num'))
        self.postprocess_num = int(cfg.get(section_name, 'postprocess_num'))
        self.img_queue_capacity = int(cfg.get(section_name, 'img_queue_capacity'))
        self.score_queue_capacity = int(cfg.get(section_name, 'score_queue_capacity'))

        # print('dataload_num, postprocess_num, img_queue_capacity, score_queue_capacity: ', \
        #       self.dataload_num, self.postprocess_num, self.img_queue_capacity, self.score_queue_capacity)

    def create_process(self):
        # 1. create process for each task
        for p_idx in range(self.process_num):
            # 1.1 further parse the gpu sub-info
            gpu_arr = GPUArr()
            gpu_arr.gpu_num = len(self.gpuInfos[p_idx])
            assert gpu_arr.gpu_num <= MAX_GPU_NUM_PER_PROC, "create_process: Exceeding the Maximum number of GPUs for a process"
            for g_idx, gpuInfo in enumerate(self.gpuInfos[p_idx]):
                gpu_arr.gpu_info[g_idx].gpu_id = gpuInfo['id']
                gpu_arr.gpu_info[g_idx].batch_size = gpuInfo['batch_size']
                gpu_arr.gpu_info[g_idx].engine = bytes(gpuInfo['engine'], 'utf-8')

            # 1.2 create queue and process
            q = Queue()
            self.queue.append(q)
            # self.process.append(Process(target=worker, args=(p_idx, q,
            #                                                  bytes(self.rawImage[p_idx], 'utf-8'),
            #                                                  bytes(self.anomalyImage[p_idx], 'utf-8'),
            #                                                  gpu_arr,
            #                                                  self.dataload_num, self.postprocess_num,
            #                                                  self.img_queue_capacity, self.score_queue_capacity)
            #                             ))
            self.process.append(threading.Thread(target=worker, args=(p_idx, q,
                                                                      bytes(self.rawImage[p_idx], 'utf-8'),
                                                                      bytes(self.anomalyImage[p_idx], 'utf-8'),
                                                                      gpu_arr,
                                                                      self.dataload_num, self.postprocess_num,
                                                                      self.img_queue_capacity, self.score_queue_capacity)
                                                 ))

        # 2. start all the created processes
        for p_idx in range(self.process_num):
            self.process[p_idx].start()


def worker(p_idx, q, rawImage, anomalyImage, gpu_arr, data_load_num, postprocess_num, img_queue_capacity, score_queue_capacity):
    global lib_ad, lib_obj
    # lib_test.display.argtypes = (c_char_p, c_char_p, POINTER(GPUArr), c_int, c_int)
    # lib_test.display(rawImage, anomalyImage, byref(gpu_arr), data_load_num, postprocess_num)
    logging.debug('anomaly_detect_start')
    ret = lib_ad.anomaly_detect_start(lib_obj[p_idx], rawImage, anomalyImage, byref(gpu_arr), data_load_num, postprocess_num, img_queue_capacity, score_queue_capacity)
    logging.info('worker:{} exit'.format(p_idx))
    if ret == 0:
        q.put('SUCCESS')
    elif ret == -1:
        q.put('ERROR')


if __name__ == '__main__':
    anomalyDetectList = [
        {
            'rawImage': "D:/data/YS56070-8_L_2/images/raw/C0",
            'anomalyImage': "D:/data/YS56070-8_L_2/images/anomaly/C0",
            'gpuInfos': [{"id": 0, "engine": "engine1", "batch_size": 4},
                         {"id": 1, "engine": "engine1", "batch_size": 4}]
        },
        {
            'rawImage': "D:/data/YS56070-8_L_2/images/raw/C1",
            'anomalyImage': "D:/data/YS56070-8_L_2/images/anomaly/C1",
            'gpuInfos': [{"id": 2, "engine": "engine2", "batch_size": 4},
                         {"id": 3, "engine": "engine2", "batch_size": 4}]
        }
    ]

    ad = AnomalyDetection(anomalyDetectList)
    ad.start()
    # time.sleep(1)

