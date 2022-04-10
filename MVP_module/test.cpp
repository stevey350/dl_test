#include <iostream>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>


using namespace std;

// g++ -o libanomaly_detect.so -shared -fPIC test.cpp

typedef struct gpu_info_s {
    uint32_t gpu_id;
    uint32_t batch_size;
    char engine[256];
} gpu_info_t;

typedef struct gpu_arr_s {
    uint32_t gpu_num;
    gpu_info_t gpu_info[4];
} gpu_arr_t;

typedef struct config_s {
    char raw_img_path[256];
    char anomaly_img_path[256];
    char enviroment[32];
    char inferred_dir[256];
    char gt_dir[256];
    uint32_t width;
    uint32_t height;
    uint32_t dataload_num;
    uint32_t postprocess_num;
    uint32_t img_queue_capacity;
    uint32_t score_queue_capacity;
    gpu_arr_t gpu_arr;
} config_t;

typedef struct Result_s {
    int progress;
    char state[20];
} Result_t;

uint32_t cnt;
pthread_t g_pth;
Result_t result;
uint8_t is_stop;

class AnomalyDetect
{
public:
    AnomalyDetect() {
        cnt = 0;
        result.progress = 0;
//        strcpy(result.state, "IDLE");
//        obj = (void *)0;
    }
	// int anomaly_detect_start(const char *raw_img_path, const char *anomaly_img_path, gpu_arr_t *gpu_arr, float threshold, \
	//                          int dataload_num, int postprocess_num, int img_queue_capacity, int score_queue_capacity);
    int anomaly_detect_start(config_t *config);

    int anomaly_detect_stop(void);
    Result_t * anomaly_detect_state(void);

private:
    AnomalyDetect *obj;

};


void *selfthread(void *arg)
{
    while (!is_stop) {
        result.progress++;
        strcpy(result.state, "RUNNING");
        cout << "-->thread running" << endl;
        sleep(1);
    }
    strcpy(result.state, "FINISH");

    pthread_exit(NULL);
}


// int AnomalyDetect::anomaly_detect_start(const char *raw_img_path, const char *anomaly_img_path, gpu_arr_t *gpu_arr, float threshold, \
// 	                                     int dataload_num, int postprocess_num, int img_queue_capacity, int score_queue_capacity)
// {
//     cout << "-->raw_img_path: " << raw_img_path << endl;
//     cout << "-->anomaly_img_path: " << anomaly_img_path << endl;
//     cout << "-->dataload_num: " << dataload_num << endl;
//     cout << "-->postprocess_num: " << postprocess_num << endl;
//     cout << "-->img_queue_capacity: " << img_queue_capacity << endl;
//     cout << "-->socre_queue_capacity: " << score_queue_capacity << endl;
//     cout << "-->gpu_num: " << gpu_arr->gpu_num << endl;
//     cout << "threshold: " << threshold << endl;
//     for(int i=0; i<gpu_arr->gpu_num; i++) {
//         cout << "  -->id: " << gpu_arr->gpu_info[i].gpu_id << endl;
//         cout << "  -->batch_size: " << gpu_arr->gpu_info[i].batch_size << endl;
//         cout << "  -->engine: " << gpu_arr->gpu_info[i].engine << endl;
//     }

//     pthread_create(&g_pth, NULL, selfthread, NULL);
//     strcpy(result.state, "RUNNING");

//     return 0;
// }

int AnomalyDetect::anomaly_detect_start(config_t *config)
{
    cout << "-->raw_img_path: " << config->raw_img_path << endl;
    cout << "-->anomaly_img_path: " << config->anomaly_img_path << endl;
    cout << "-->enviroment: " << config->enviroment << endl;
    cout << "-->inferred_dir: " << config->inferred_dir << endl;
    cout << "-->gt_dir: " << config->gt_dir << endl;
    cout << "-->width: " << config->width << endl;
    cout << "-->height: " << config->height << endl;
    cout << "-->dataload_num: " << config->dataload_num << endl;
    cout << "-->postprocess_num: " << config->postprocess_num << endl;
    cout << "-->img_queue_capacity: " << config->img_queue_capacity << endl;
    cout << "-->socre_queue_capacity: " << config->score_queue_capacity << endl;
    cout << "-->gpu_num: " << config->gpu_arr.gpu_num << endl;
    for(int i=0; i<config->gpu_arr.gpu_num; i++) {
        cout << "  -->id: " << config->gpu_arr.gpu_info[i].gpu_id << endl;
        cout << "  -->batch_size: " << config->gpu_arr.gpu_info[i].batch_size << endl;
        cout << "  -->engine: " << config->gpu_arr.gpu_info[i].engine << endl;
    }

    pthread_create(&g_pth, NULL, selfthread, NULL);
    strcpy(result.state, "RUNNING");

    return 0;
}

int AnomalyDetect::anomaly_detect_stop(void) {
    cout << "-->anomaly_detect_stop" << endl;

    is_stop = 1;

    return 0;
}

Result_t *AnomalyDetect::anomaly_detect_state(void) {
    cout << "-->anomaly_detect_state: " << cnt << endl;
    return (Result_t *)&result;
}


extern "C" {

    AnomalyDetect* anomalydetect_new() {
        return new AnomalyDetect;
    }

    // int anomaly_detect_start(AnomalyDetect* cb, const char *raw_img_path, const char *anomaly_img_path, gpu_arr_t *gpu_arr, float threshold, \
    //                           int dataload_num, int postprocess_num, int img_queue_capacity, int score_queue_capacity) {
    //     return cb->anomaly_detect_start(raw_img_path, anomaly_img_path, gpu_arr, threshold, dataload_num, postprocess_num, img_queue_capacity, score_queue_capacity);
    // }

    int anomaly_detect_start(AnomalyDetect* cb, config_t *config) {
        return cb->anomaly_detect_start(config);
    }

    Result_t *anomaly_detect_state(AnomalyDetect* cb) {
        return cb->anomaly_detect_state();
    }
    int anomaly_detect_stop(AnomalyDetect* cb) {
        return cb->anomaly_detect_stop();
    }
}
