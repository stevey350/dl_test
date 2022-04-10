import ctypes
from configparser import ConfigParser
from ctypes import *
from multiprocessing import Process, Queue, Manager
from multiprocessing.managers import BaseManager
import time
import numpy as np
import logging
import os


#lib_path = 'anomaly_detect\\anomaly_detect.dll'
#ini_path = 'anomaly_detect\\ad.ini'           # configuration file
lib_path = '/workspace/ssj/MVP_module/libanomaly_detect.so'
ini_path = 'ad.ini'           # configuration file

MAX_GPU_NUM_PER_PROC = 4        # Maximum number of GPUs for a process
LOG_FORMAT = "%(filename)s:[%(funcName)s] - %(levelname)s - %(message)s"
STATUS_PRIO = ["IDLE", "FINISH", "RUNNING", "ABORTED", "ERROR"]

class GPUInfo(Structure):
    _fields_ = [("gpu_id", c_uint), ("batch_size", c_uint), ("engine", c_char * 256)]


class GPUArr(Structure):
    _fields_ = [("gpu_num", c_uint), ("gpu_info", GPUInfo * MAX_GPU_NUM_PER_PROC)]


class Config(Structure):
    _fields_ = [
                ("raw_img_path", c_char*256), ("anomaly_img_path", c_char*256), ("enviroment", c_char*32), ("inferred_dir", c_char*256), 
                ("gt_dir", c_char*256),       ("width", c_uint),                ("height", c_uint),        ("dataload_num", c_uint),
                ("postprocess_num", c_uint),  ("img_queue_capacity", c_uint),   ("score_queue_capacity", c_uint), ("gpu_arr", GPUArr)
               ]

class ResultState(Structure):
    _fields_ = [("progress", c_int), ("state", c_char * 20)]


class AnomalyDetection(object):
    def __init__(self, anomalyDetectList, managerId=None, width=640, height=640, env=None):
        logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
        self.process = []
        self.managerId = managerId
        self.width = width          # new
        self.height = height        # new
        self.env = env              # new

        # parameters parse
        self.process_num = len(anomalyDetectList)
        self.rawImage = []
        self.anomalyImage = []
        self.gpuInfos = []
        self.inferredDir = []   # new
        self.gtDir = []         # new
        manager = Manager()
        self.task_status = manager.list()
        for ad_task in anomalyDetectList:
            self.rawImage.append(ad_task['rawImage'])
            self.anomalyImage.append(ad_task['anomalyImage'])
            self.gpuInfos.append(ad_task['gpuInfos'])
            self.inferredDir.append(ad_task['inferredDir'])
            self.gtDir.append(ad_task["gtDir"])
            self.task_status.append({"is_stop": 0, "status": "IDLE", "progress": 0})

        self.cfg_init()
        self.status = "IDLE"

    def start(self):
        if self.status == 'RUNNING':
            logging.warning('AnomalyDetection module is running now')
            return False
        else:
            try:
                self.create_process()
            except Exception as e:
                logging.error('AnomalyDetection start failed, e={}'.format(e))
                return False

            logging.info('AnomalyDetection module create process OK')
            return True

    def close(self):
        for idx in range(self.process_num):
            cur_dict = self.task_status[idx]
            cur_dict["is_stop"] = 1
            self.task_status[idx] = cur_dict

        # wait the process to finish
        while True:
            cnt = 0
            for idx in range(self.process_num):
                if self.task_status[idx]["status"] != "RUNNING":
                    cnt += 1
            if cnt == self.process_num:
                break
            time.sleep(0.2)
            # logging.warn('waiting the process to finish')

    def get_status(self):
        if self.process_num == 0:
            logging.error("Length of anomalyDetectList is 0")
            return {"managerId": self.managerId, "progress": 0, "state": "ERROR"}
            
        # 1. Getting process status and progress
        sp_status = []
        sp_progress = []
        for idx in range(self.process_num):
            # sp_status.append(to_str(self.task_status[idx]["status"]))
            sp_status.append(self.task_status[idx]["status"])
            sp_progress.append(self.task_status[idx]["progress"])
        logging.debug("sp_status: {}, sp_progress: {}".format(sp_status, sp_progress))

        # 2. Handling ERROR case in one of processes
        if ("ERROR" in sp_status) and (str(sp_status).count("ERROR") < self.process_num):
            logging.warn("An ERROR occurred in one of processes")
            for idx in range(self.process_num):
                if(sp_status[idx] != "ERROR"):
                    cur_dict = self.task_status[idx]
                    cur_dict["is_stop"] = 1
                    self.task_status[idx] = cur_dict
        
        # 3. Fusing into one progess and status
        status_index = 0     # i.e., IDLE
        for idx in range(self.process_num):
            tmp_index = STATUS_PRIO.index(sp_status[idx])
            if tmp_index > status_index:
                status_index = tmp_index

        self.status = STATUS_PRIO[status_index]
        progress = np.mean(sp_progress)
        logging.debug("progress: {}, state: {}".format(progress, self.status))

        return {"managerId": self.managerId, "progress": progress, "state": self.status}

    def cfg_init(self):
        section_name = 'ad'

        cfg = ConfigParser()
        cfg.read(ini_path)
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

            # 1.2 create process
            self.process.append(Process(target=worker, args=(p_idx, self.task_status,
                                                             bytes(self.rawImage[p_idx], 'utf-8'),
                                                             bytes(self.anomalyImage[p_idx], 'utf-8'),
                                                             bytes(self.env, 'utf-8'),
                                                             bytes(self.inferredDir[p_idx], 'utf-8'),
                                                             bytes(self.gtDir[p_idx], 'utf-8'),
                                                             gpu_arr,
                                                             self.width, self.height,
                                                             self.dataload_num, self.postprocess_num,
                                                             self.img_queue_capacity, self.score_queue_capacity)
                                        ))

        # 2. start all the created processes
        for p_idx in range(self.process_num):
            self.process[p_idx].start()
            cur_dict = self.task_status[p_idx]
            cur_dict["status"] = "RUNNING"
            self.task_status[p_idx] = cur_dict


def to_str(bytes_or_str):
    if isinstance(bytes_or_str, bytes):
        return bytes_or_str.decode('utf-8')
    return bytes_or_str


def worker(p_idx, task_status, rawImage, anomalyImage, env, inferredDir, gtDir, gpu_arr, 
            width, height, data_load_num, postprocess_num, img_queue_capacity, score_queue_capacity):
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
    
    # reconstruct config struct
    config_para = Config()
    config_para.raw_img_path = rawImage
    config_para.anomaly_img_path = anomalyImage
    config_para.enviroment = env
    config_para.inferred_dir = inferredDir
    config_para.gt_dir = gtDir
    config_para.width = width
    config_para.height = height
    config_para.dataload_num = data_load_num
    config_para.postprocess_num = postprocess_num
    config_para.img_queue_capacity = img_queue_capacity
    config_para.score_queue_capacity = score_queue_capacity
    config_para.gpu_arr = gpu_arr

    # load C++ dynamic library
    lib_ad = cdll.LoadLibrary(lib_path)
    lib_obj = lib_ad.anomalydetect_new()
    lib_ad.anomaly_detect_start.argtypes = [c_void_p, POINTER(Config)]
    lib_ad.anomaly_detect_state.restype = POINTER(ResultState)

    logging.debug('worker:{} start'.format(p_idx))
    ret = lib_ad.anomaly_detect_start(lib_obj, byref(config_para))

    while True:
        time.sleep(2)
        # update status
        cur_status = lib_ad.anomaly_detect_state(lib_obj)
        # logging.debug("progress: {}, status: {}".format(cur_status.contents.progress, cur_status.contents.state))
        cur_dict = task_status[p_idx]
        cur_dict["status"] = to_str(cur_status.contents.state)      # bytes to unicode
        cur_dict["progress"] = cur_status.contents.progress
        task_status[p_idx] = cur_dict

        # stop action listen
        if task_status[p_idx]["is_stop"] == 1:
            lib_ad.anomaly_detect_stop(lib_obj)

        # quit
        if task_status[p_idx]["status"] != "RUNNING":
            break

    logging.debug('worker:{} exit'.format(p_idx))


if __name__ == '__main__':
    import keyboard
   
    anomalyDetectList = [
        {
            'rawImage': "D:\\wxp\\MVP_module\\src_img2",
            'anomalyImage': "D:\\wxp\\MVP_module\\anomaly_dir",
            'gpuInfos': [{"id": 0, "engine": "D:\\guls\\mvp\\code\\trt_convert\\AutoEncoder_encrypt_b8_0.engine", "batch_size": 1}],
                         #{"id": 7, "engine": "./AutoEncoder_fp16_b16_20210609.engine", "batch_size": 8}]
            'inferredDir': "D:/data/YS56070-8_L_2/images/anomaly/",
            'gtDir': "D:/data/YS56070-8_L_2/images/gt/"
        },
        #{
        #    'rawImage': "D:\\wxp\\MVP_module\\src_img2",
        #    'anomalyImage': "D:\\wxp\\MVP_module\\anomaly_dir2",
        #    'gpuInfos': [{"id": 1, "engine": "D:\\guls\\mvp\\code\\trt_convert\\AutoEncoder_encrypt_b8_0.engine", "batch_size": 1}]
        #                 #{"id": 7, "engine": "./AutoEncoder_fp16_b16_20210609.engine", "batch_size": 8}]
        #}
    ]


    
    while True:
        managerId = "110"
        width = 640
        height = 640
        env = "huayan"
        ad = AnomalyDetection(anomalyDetectList, managerId, width, height, env)
        logging.debug("start")
        ad.start()
    
        for i in range(15):
            time.sleep(2)
            logging.debug('status 1:{}'.format(ad.get_status()))
            
        if keyboard.is_pressed('s'):
            logging.debug('key s is pressed')
            break
        
        logging.debug('close')
        ad.close()


    logging.debug('status 5: {}'.format(ad.get_status()))


