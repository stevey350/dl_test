from multiprocessing import Process, Queue
import time
from AnomalyDetection3 import AnomalyDetection
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

def to_str(bytes_or_str):
    if isinstance(bytes_or_str, bytes):
        return bytes_or_str.decode('utf-8')
    return bytes_or_str


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
    anomalyDetectList = [
        {
            'rawImage': "/workspace/huayan_nfs/localhost/YS50083-10_BD/ad_rawimages/part1_C0_test/",
            'anomalyImage': "/workspace/backbone_check/dfr_od/tmp/test/anomaly_py/",
            'gpuInfos': [{"id": 1, "engine": "/workspace/backbone_check/dfr_od/anomaly_detect/build/AutoEncoder_fp16_b16_20210609.engine", "batch_size": 8},
                         {"id": 2, "engine": "/workspace/backbone_check/dfr_od/anomaly_detect/build/AutoEncoder_fp16_b16_20210609.engine", "batch_size": 8}]
        },
        # {
        #     'rawImage': "D:/data/YS56070-8_L_2/images/raw/C1",
        #     'anomalyImage': "D:/data/YS56070-8_L_2/images/anomaly/C1",
        #     'gpuInfos': [{"id": 2, "engine": "engine2", "batch_size": 4},
        #                  {"id": 3, "engine": "engine2", "batch_size": 4}]
        # }
    ]

    a = "RUNNING"
    b = b"FINISH"
    print(a)
    print(b)
    print(type(a))
    print(type(b))
    print(type(to_str(a)))
    print(type(to_str(b)))

    ad = AnomalyDetection(anomalyDetectList)
    logging.debug("start")
    ad.start()
    time.sleep(10)
    logging.debug('status 1:{}'.format(ad.get_status()))

    time.sleep(2)
    logging.debug('status 2:{}'.format(ad.get_status()))

    time.sleep(2)
    logging.debug('status 3:{}'.format(ad.get_status()))

    time.sleep(2)
    logging.debug('status 4: {}'.format(ad.get_status()))

    time.sleep(2)
    logging.debug('close')
    ad.close()

