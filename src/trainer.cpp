/**
* @file   trainer.cpp
* @author wanguanglu
* @date   2016/09/23 11:42:34
* @brief 
*  
**/


#include "trainer.h"


namespace nlu {
namespace maxent {


int Trainer::build_param(DataReader& data_reader, LinearModel& linear_model) {
    int ret = 0;

    ret = data_reader.merge_events();
    if (ret != 0) {
        log_warn("merge event failed.");
        return -1;
    }

    ret = linear_model.set_feat_label_map(data_reader._feat_map, data_reader._label_map);
    if (ret != 0) {
        log_warn("set feature & label map failed.");
        return -1;
    }

    return 0;
}

int Trainer::cmpt_tot_event_count(DataReader& data_reader) {
    _tot_event_count = 0.0;
    std::vector<Event>::iterator it;

    for (it=data_reader._events.begin(); it!=data_reader._events.end(); it++) {
        _tot_event_count += it->_count;
    }

    if (_tot_event_count <= 0.0) {
        log_warn("compute total event number failed.");
        return -1;
    }
    return 0;
}

int Trainer::cmpt_observed(DataReader& data_reader) {

    bzero(_observed, sizeof(float)*_feat_num*_label_num);
    float cur_obs = 0.0;

    std::vector<Event>::iterator it;
    for (it=data_reader._events.begin(); it!=data_reader._events.end(); it++) {
        Event& event    = *it;
        std::vector<FeatPair>::iterator feat_it;
        for (feat_it=event._contexts.begin(); feat_it != event._contexts.end(); feat_it++) {
            _observed [ feat_it->_feat_id*_label_num + event._label] += 
                            event._count * feat_it->_val;
        }
    }
    
    return 0;
}

int Trainer::cmpt_estimated(DataReader& data_reader, LinearModel& model, int& correct_num, float& log_likelihood) {

    int label = 0;
    float* score = NULL;

    bzero(_estimated, sizeof(float)*_feat_num*_label_num);
    correct_num    = 0;
    log_likelihood = 0.0;

    score = new float[_label_num];
    if (score == NULL) {
        log_warn("new for score failed.");
        return -1;
    }

    std::vector<Event>::iterator it;
    for (it=data_reader._events.begin(); it!=data_reader._events.end(); it++) {
        std::vector<FeatPair> contexts = it->_contexts;
        model.predict(contexts, score, label);

        if (label == it->_label) {
            correct_num += it->_count;
        }

        log_likelihood += log(score[it->_label]);

        std::vector<FeatPair>::iterator feat_it;
        for (feat_it=contexts.begin(); feat_it!=contexts.end(); feat_it++) {
            for (label=0; label<_label_num; label++) {
                _estimated [feat_it->_feat_id*_label_num + label] +=
                    it->_count * feat_it->_val * score[label];
            }
        }
    }

    delete []score;
    return 0;
}

int Trainer::init_train(DataReader& data_reader) {

    int ret = 0;

    _feat_num  = data_reader.feat_num();
    _label_num = data_reader.label_num();

    _observed = new float[_feat_num*_label_num];
    if (NULL == _observed) {
        log_warn("new for observed failed.");
        return -1;
    }

    _estimated = new float[_feat_num*_label_num];
    if (NULL == _estimated) {
        log_warn("new for estimated failed.");
        return -1;
    }

    ret = cmpt_observed(data_reader);
    if (ret != 0) {
        log_warn("compute observed failed.");
        return -1;
    }

    ret = cmpt_tot_event_count(data_reader);
    if (ret != 0) {
        log_warn("compute total event number failed.");
        return -1;
    }
    log_notice("compute total event number : %f.", _tot_event_count);

    return 0;
}



}
}
