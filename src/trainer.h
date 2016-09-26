/**
* @file   trainer.h
* @author wanguanglu
* @date   2016/09/23 10:28:37
* @brief 
*  
**/

#include "maxent_model.h"
#include "data_reader.h"


#ifndef NLU_MAXENT_TRAINER_H
#define NLU_MAXENT_TRAINER_H

namespace nlu {
namespace maxent {


class Trainer {

public:
    Trainer() : _feat_num(0), _label_num(0), 
                _observed(NULL), _estimated(NULL), _tot_event_count(0) {}
    virtual ~Trainer(){}

    virtual int train (
        DataReader& data_reader, LinearModel& model,
        int iter, float tol, float sigma2) = 0;

    int build_param(DataReader& data_reader, LinearModel& model);

protected:
    int cmpt_tot_event_count(DataReader& data_reader);
    int cmpt_observed(DataReader& data_reader);
    int cmpt_estimated(DataReader& data_reader, LinearModel& model, 
            int& correct_num, float& log_likelihood);

    virtual int init_train(DataReader& data_reader);

protected:
    int     _feat_num;
    int     _label_num;
    float*  _observed;
    float*  _estimated;
    double  _tot_event_count;
};


}
}



#endif
