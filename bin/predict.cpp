/**
* @file main.cpp
* @author jiangzhengxiang
* @date 2016/08/02 16:56:47
* @brief 
*  
**/

#include "maxent.h"
#include "data_reader.h"

using namespace nlu::maxent;

int init_log(const char* proc_name) {
    glog_init("maxent", "log", "maxent");
    log_notice("Init Log Success");
    return 0;
}

/*!
 * @brief show usage
 */
void show_usage(const char* cmd) {
    printf("usage: %s model_file predict_file result_file\n", cmd);
    return;
}

int main(int argc, char* argv[]) {
    int  ret = 0;
    char c   = 0;

    if (argc != 4) {
        show_usage(argv[0]);
        return -1;
    }

    ret = init_log("maxent");
    if (ret != 0) {
        return ret;
    }

    LinearModel model;
    ret = model.load_model(argv[1]);
    if (ret != 0) {
        log_warn("load model from %s failed.", argv[1]);
        return -1;
    }

    DataReader data_reader;
    data_reader.set_feat_label_map(model.feat_map(), model.label_map());

    ret = data_reader.load_file(argv[2]);
    if (ret != 0) {
        log_warn("load data from %s failed.", argv[2]);
        return -1;
    }

    FILE* fp_result = fopen(argv[3], "wb");
    if (NULL == fp_result) {
        log_warn("open file %s for write failed.", argv[3]);
        return -1;
    }

    std::vector<Event>& events = data_reader.events();
    float* score = new float[data_reader.label_num()];
    int label       = 0;
    int correct_num = 0;

    fprintf(fp_result, "ref\tres\n");
    fprintf(fp_result, "==========\n");
    for (std::vector<Event>::iterator it = events.begin(); it != events.end(); it++) {
        model.predict(it->_contexts, score, label);

        if (label == it->_label) {
            correct_num ++;
        }
        fprintf(fp_result, "%d\t%d\n", it->_label, label);
    }

    fprintf(fp_result, "total number:%d\tcorrect number:%d\tcorrect rate:%f%%\n",
                events.size(), correct_num, (float)correct_num/events.size()*100);

    delete []score;
    fclose(fp_result);

    return 0;
}


/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

