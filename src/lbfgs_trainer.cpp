/**
* @file   lbfgs_trainer.cpp
* @author wanguanglu
* @date   2016/09/25 17:43:34
* @brief 
*  
**/

#include "lbfgs_trainer.h"

namespace nlu {
namespace maxent {

int LbfgsTrainer::init_train(DataReader& data_reader) {
    int ret = 0;

    ret = Trainer::init_train(data_reader);
    if (ret != 0) {
        log_warn("init train for lbfgs train failed.");
        return -1;
    }

    _weight_mat = lbfgs_malloc(_feat_num * _label_num);
    if (NULL == _weight_mat) {
        log_warn("malloc for weight matrix failed.");
        return -1;
    }
    bzero(_weight_mat, sizeof(lbfgsfloatval_t)*_feat_num*_label_num);
    
    lbfgs_parameter_init(&_lbfgs_param);

    return 0;
}

int LbfgsTrainer::progress(
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int nit,
    int ls) {   

        log_notice("%3d\t%f\t  %.3f%%", nit, -fx, (float)_correct_num/_tot_event_count*100);
        return 0;

}

lbfgsfloatval_t LbfgsTrainer::evaluate(
            const lbfgsfloatval_t *x,
            lbfgsfloatval_t *grad,
            const int n,
            const lbfgsfloatval_t step) {   

    int ret = 0;
    float log_likelihood = 0.0;
    int   index   = 0;
    float penalty = 0.0;
    update_weight();

    ret = cmpt_estimated(*_data_reader, *_model, _correct_num, log_likelihood);
    if (ret != 0) {
        log_warn("compute estimated value failed.");
        return 0.0;
    }

    log_likelihood *= -1;

    
    for (int feat_id=0; feat_id<_feat_num; feat_id++) {
        for (int label_id=0; label_id<_label_num; label_id++) {

            index = feat_id*_label_num + label_id;
            if (_sigma2 == 0.0 && _observed[index] == 0) {
                grad[index] = 0.0; //unseen case
            } else {
                grad[index] = _estimated[index] - _observed[index];
            }

        }
    }
    
    if (_sigma2) { 
        for(index = 0; index < _feat_num*_label_num; index++) {
            penalty = _weight_mat[index] / _sigma2;
            grad[index] += penalty;
            log_likelihood += (penalty * _weight_mat[index]) / 2;
        }
    }
    return (lbfgsfloatval_t)log_likelihood;
}



int LbfgsTrainer::update_weight() {
    for (int i=0; i<_feat_num*_label_num; i++) {
        _model->_weight_mat[i] = _weight_mat[i];
    }
    return 0;
}

int LbfgsTrainer::train (
        DataReader& data_reader, LinearModel& model,
        int iter, float tol, float sigma2) {

    int ret = 0;

    if (sigma2 < 0.0) {
        log_warn ("gauss prior should be greater than zero.");
        return -1;
    }

    ret = build_param(data_reader, model);
    if (ret != 0) {
        log_warn("build parameter failed.");
        return -1;
    }

    ret = init_train(data_reader);
    if (ret != 0) {
        log_warn("init train failed.");
        return -1;
    }

    _lbfgs_param.m = 5;
    _lbfgs_param.epsilon = tol;
    _lbfgs_param.max_iterations = iter;
    _model       = &model;
    _data_reader = &data_reader;
    _sigma2      = sigma2;
    _correct_num = 0;

    log_notice("start LBFGS iterations...");
    log_notice("number of feature:      %d.", _feat_num);
    log_notice("number of label:        %d.", _label_num);
    log_notice("Tolerance:              %E.", tol);
    log_notice("Gaussian Penalty:       %s", (sigma2?"on":"off"));
    log_notice("objective: min sum {-log p(y|x)} + 1/(2*sigma2)*||w||^2");

    log_notice("iters   loglikelihood    training accuracy");
    log_notice("==========================================");

    
    ret = lbfgs(_feat_num*_label_num, _weight_mat, NULL, _evaluate, _progress, this, &_lbfgs_param);
    if (ret != 0) {
        switch (ret){
            case -997:
                log_warn("LBFGS finished with max mIter reached!");
                break;
            default:
                log_warn("LBFGS exit with error code %d.", ret);
                break;
        }
    }

    update_weight();

    return 0;

}


}
}
