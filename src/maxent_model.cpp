/**
* @file   maxent.cpp
* @author wanguanglu
* @date   2016/03/22 18:15:54
* @brief 
*  
**/

#include "maxent_model.h"

namespace nlu {
namespace maxent {


LinearModel::LinearModel() {
    _weight_mat = NULL;
    _feat_num  = 0;
    _label_num = 0;
}

LinearModel::~LinearModel() {
    if (_weight_mat) {
        delete []_weight_mat;
    }
}

int LinearModel::load_model(const char* model_file) {
    FILE* fp  = NULL;
    char  line[MAX_TEXT_LEN];
    char* pos1, *pos2;
    char  tmp[MAX_TEXT_LEN];
    int   len = 0;

    fp = fopen(model_file, "rb");
    if (NULL == fp) {
        log_warn("open model file %s failed.", model_file);
        return -1;
    }

    //read label map
    if(!fgets(line, MAX_TEXT_LEN, fp)) {
        log_warn("read label number failed.");
        return -1;
    }
    len = strlen(line);
    if (line[len - 1] == '\n') {
        line[len - 1] = '\0';
    }

    char* p = strtok(line, "\t ");
    if (p == NULL) {
        log_warn("wrong format %s", line);
        return -1;
    }
    p += strlen(p) + 1;
    _label_num = atoi(p);
    log_notice("label number:%d.", _label_num);

    if (_label_num <= 0) {
        log_warn("read label_num failed.");
        return -1;
    }

    for (int label_id=0; label_id<_label_num; label_id++) {
        fgets(line, MAX_TEXT_LEN, fp);

        int len = strlen(line);
        if (line[len - 1] == '\n') {
            line[len - 1] = '\0';
        }

        if (strlen(line) == 0) {
            continue;
        }

        _label_map.add_item(std::string(line));
    }

    //read feat map
    while(fgets(line, MAX_TEXT_LEN, fp)) {

        len = strlen(line);
        if (line[len - 1] == '\n') {
            line[len - 1] = '\0';
        }

        if (strlen(line) == 0) {
            continue;
        }
        break;
    }

    p = strtok(line, "\t ");
    if (p == NULL) {
        log_warn("wrong format %s", line);
        return -1;
    }
    p += strlen(p) + 1;
    _feat_num = atoi(p);
    log_notice("feature number:%d.", _feat_num);

    for (int feat_id=0; feat_id<_feat_num; feat_id++) {

        fgets(line, MAX_TEXT_LEN, fp);
        int len = strlen(line);
        if (line[len - 1] == '\n') {
            line[len - 1] = '\0';
        }
        _feat_map.add_item(std::string(line));
    }


    _weight_mat = new float[_feat_num*_label_num];
    if (NULL == _weight_mat) {
        log_warn("new for _weight_mat failed.");
        return -1;
    }

    int feat_idx = 0;
    while (fgets(line, MAX_TEXT_LEN, fp)) {
        if (line[strlen(line)-1] == '\n') {
            line[strlen(line)-1] = '\0';
        }

        if (strlen(line) == 0) {
            continue;
        }

        pos1 = strtok(line, "\t ");
        for (int i=0; i<_label_num; i++) {
            pos1 = strtok(NULL, "\t ");
            _weight_mat[feat_idx*_label_num+i ] = atof(pos1);
        }
        feat_idx ++;
    }

    fclose(fp);

    log_notice("Load linear model from %s success.", model_file);
    return 0;
}

int LinearModel::save_model(const char* model_file) {
    FILE* fp    = NULL;
    std::string item; 

    fp = fopen(model_file, "wb");
    if (fp == NULL) {
        log_warn("open file %s for write failed.", model_file);
        return -1;
    }

    //write label
    fprintf(fp, "#label\t%d\n", _label_num);
    for (int i=0; i<_label_num; i++) {
        _label_map.get_item_str(i, item);
        fprintf(fp, "%s\n", item.c_str());
    }
    fprintf(fp, "\n");

    //write feature
    fprintf(fp, "#feature\t%d\n", _feat_num);
    for (int i=0; i<_feat_num; i++) {
        _feat_map.get_item_str(i, item);
        fprintf(fp, "%s\n", item.c_str());
    }
    fprintf(fp, "\n");

    for (int feat_id=0; feat_id<_feat_num; feat_id++) {
        fprintf(fp, "%d:", feat_id);
        for (int label_id=0; label_id<_label_num; label_id++) {
            fprintf(fp, "\t%f", _weight_mat[feat_id*_label_num + label_id]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return 0;
}

float LinearModel::cal_score(const float* feat, int label_id) const {
    float score = 0.0;

    score += _weight_mat[label_id]; //bias;
    for (int feat_id=1; feat_id<_feat_num; feat_id++) {
        score += feat[feat_id-1] * _weight_mat[feat_id*_label_num+label_id];
    }
    return score;
}

float LinearModel::cal_score(const std::map<int, float>& feat, int label_id) const {
    float score = 0.0;
    std::map<int, float>::const_iterator iter;
    score += _weight_mat[label_id]; //bias;
    
    for (iter = feat.begin(); iter != feat.end(); iter++) {
        score += iter->second * _weight_mat[iter->first * _label_num + label_id];
    }
    return score;
}

float LinearModel::cal_score(const std::vector<FeatPair>& feat, int label_id) const {
    float score = 0.0;
    std::vector<FeatPair>::const_iterator iter;
    score += _weight_mat[label_id]; //bias;
    
    for (iter = feat.begin(); iter != feat.end(); iter++) {
        score += iter->_val * _weight_mat[iter->_feat_id * _label_num + label_id];
    }
    return score;
}

int LinearModel::normalize_score(float* score) const {
    float sum_score = 0.0;
    float max_score = MIN_FLOAT;

    for (int label_id=0; label_id<_label_num; label_id++) {
        if (score[label_id] > max_score) {
            max_score = score[label_id];
        }
    }

    for (int label_id=0; label_id<_label_num; label_id++) {
        score[label_id] = exp(score[label_id] - max_score);
        sum_score += score[label_id];
    }

    for (int label_id=0; label_id<_label_num; label_id++) {
        score[label_id] /= sum_score;
    }
    return 0;
}

int LinearModel::normalize_score(std::vector<float>& score) const {
    float sum_score = 0.0;
    for (int label_id = 0; label_id < _label_num; label_id++) {
        score[label_id] = exp(score[label_id]);
        sum_score += score[label_id];
    }

    for (int label_id=0; label_id < _label_num; label_id++) {
        score[label_id] /= sum_score;
    }
    return 0;
}

int LinearModel::predict(const std::map<int, float>& feat, std::vector<float>& score, int& label) const {

    label = -1;
    score.clear();

    for (int label_id = 0; label_id < _label_num; label_id++) {
        float scr = cal_score(feat, label_id);
        score.push_back(scr);

        if (label == -1 || scr > score[label]) {
            label = label_id;
        }
    }

    if (normalize_score(score) != 0) {
        log_warn("normalize score failed.");
    }

    return 0;
}

int LinearModel::predict(const std::vector<FeatPair>& feat, float* score, int& label) const {

    label = -1;

    for (int label_id = 0; label_id < _label_num; label_id++) {
        float scr = cal_score(feat, label_id);
        score[label_id] = scr;

        if (label == -1 || scr > score[label]) {
            label = label_id;
        }
    }

    if (normalize_score(score) != 0) {
        log_warn("normalize score failed.");
    }

    return 0;
}

int LinearModel::predict(const float* feat, int feat_len, float* score, int& label) const {

    if (feat_len != _feat_num-1) {
        log_warn("feature size not match. [model feat size: %d][input feat size:%d]", 
                _feat_num-1, feat_len);
    }

    float max_score = 0; 
    for (int label_id=0; label_id<_label_num; label_id++) {
        score[label_id] = cal_score(feat, label_id);
    }

    if (normalize_score(score) != 0) {
        log_warn("normalize score failed.");
    }

    for (int label_id=0; label_id<_label_num; label_id++) {
        if (score[label_id] > max_score) {
            max_score = score[label_id];
            label = label_id;
        }
    }
    return 0;

}


int LinearModel::set_feat_label_map(ItemMap& feat_map, ItemMap& label_map) {
    _feat_map  = feat_map;
    _label_map = label_map;

    _feat_num  = feat_map.item_num(); 
    _label_num = label_map.item_num();

    if (_weight_mat != NULL) {
        delete []_weight_mat;
    }

    _weight_mat = new float[_feat_num*_label_num];
    if (_weight_mat == NULL) {
        log_warn("new for weight matrix failed.");
        return -1;
    }

    bzero(_weight_mat, sizeof(float)*_feat_num*_label_num);

    return 0;
}


}
}
