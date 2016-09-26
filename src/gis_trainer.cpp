/**
* @file   gis_trainer.cpp
* @author wanguanglu
* @date   2016/09/23 11:18:17
* @brief 
*  
**/

#include "gis_trainer.h"


namespace nlu {
namespace maxent {

GisTrainer::~GisTrainer() {
    if (_observed) {
        delete [] _observed;
        delete [] _estimated;
    }
}

int GisTrainer::train (
        DataReader& data_reader, LinearModel& model,
        int iter, float tol, float sigma2) {

    int ret = 0;
    int correct_num = 0;
    float log_likelihood = 0.0;
    char  file_name[MAX_PATH_LEN];

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

    log_notice("start GIS iterations...");
    log_notice("number of feature:      %d.", _feat_num);
    log_notice("number of label:        %d.", _label_num);
    log_notice("Tolerance:              %E.", tol);
    log_notice("Gaussian Penalty:       %s", (sigma2?"on":"off"));
    log_notice("objective: min sum {-log p(y|x)} + 1/(2*sigma2)*||w||^2");


    log_notice("iters   loglikelihood    training accuracy");
    log_notice("==========================================");

    for (int cur_iter=0; cur_iter<iter; cur_iter++) {

        ret = cmpt_estimated(data_reader, model, correct_num, log_likelihood);
        if (ret != 0) {
            log_warn("compute estimated failed.");
            return -1;
        }
        log_notice("%3d\t%f\t  %.3f%%", cur_iter, log_likelihood/_tot_event_count, 
                        (float)correct_num/_tot_event_count*100);
        

        //update parameter
        if (sigma2) {
            float* weight_mat = model._weight_mat;
            float delta = 0.0;

            for (int feat_id=0; feat_id<_feat_num; feat_id++) {
                for (int label_id=0; label_id<_label_num; label_id++) {
                    ret = newton(_estimated[feat_id*_label_num + label_id],
                                _observed[feat_id*_label_num + label_id],
                                weight_mat[feat_id*_label_num + label_id],
                                sigma2, delta);
                    if (ret != 0) {
                        log_warn("newton method failed.");
                        return -1;
                    }

                    weight_mat[feat_id*_label_num + label_id] += delta;
                }
            }
        } else {

            float* weight_mat = model._weight_mat;
            float  log_observed  = 0.0;
            float  log_estimated = 0.0;

            for (int feat_id=0; feat_id<_feat_num; feat_id++) {
                for (int label_id=0; label_id<_label_num; label_id++) {
                    //unseen feature
                    if (_observed[feat_id*_label_num + label_id] == 0.0) {
                        continue;
                    }

                    log_observed  = log(_observed[feat_id*_label_num + label_id]);

                    log_estimated = _estimated[feat_id*_label_num + label_id]==0 ? LOG_ZERO 
                                    : log(_estimated[feat_id*_label_num + label_id]);

                    weight_mat[feat_id*_label_num + label_id] += (log_observed-log_estimated)/_correct_constant;
                }
            }
        }

    }

    log_notice ("train by gis_trainer success.");
    return 0;
}


int GisTrainer::init_train(DataReader& data_reader) {

    Trainer::init_train(data_reader);

    int ret = 0;

    ret = cmpt_correct_constant(data_reader);
    if (ret != 0) {
        log_warn("compute correct constant failed.");
        return -1;
    }
    log_notice("compute correct constant : %f.", _correct_constant);

    return 0;
}

int GisTrainer::cmpt_correct_constant(DataReader& data_reader) {
    _correct_constant = -1.0;
    std::vector<Event>::iterator it;

    for (it=data_reader._events.begin(); it!=data_reader._events.end(); it++) {
        Event& event    = *it;
        float cur_count = 0.0;

        for (int i=0; i<event._contexts.size(); i++) {
            cur_count += event._contexts[i]._val;
        }

        if (cur_count > _correct_constant) {
            _correct_constant = cur_count;
        }
    }

    if (_correct_constant < 0.0) {
        log_warn("compute correct constant failed.");
        return -1;
    }
    
    return 0;    
}


// Calculate the ith GIS parameter updates with Gaussian prior
// using Newton-Raphson method
// the update rule is the solution of the following equation:
//                                       lambda_i + delta_i
// observe = estimate * exp(C*delta_i) + ------------------ * N
//                                             sigma_i^2
// note: observe and estimate were not divided by N

int GisTrainer::newton(float estimate, float observe, float weight, 
                            float sigma2, float& delta, float tol) {
    int max_iter = 50;
    float x0 = 0.0;
    delta    = 0.0;
    
    for (size_t iter = 1; iter <= max_iter; ++iter) {
        double t = estimate*exp(_correct_constant*x0);
        double fval = t + _tot_event_count*(weight+x0)/sigma2 - observe;
        double fpval = t*_correct_constant + _tot_event_count/sigma2;
        
        if (fpval == 0) {
            log_warn("zero-derivative encountered in newton() method.");
            return 0;
        }

        delta = x0 - fval/fpval;
        if (abs(delta-x0) < tol) {
            return 0;
        }

        x0 = delta;
    }
    log_warn("Failed to converge after 50 iterations in newton() method");
    return -1;
}


}
}
