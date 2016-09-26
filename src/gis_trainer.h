/**
* @file   gis_trainer.h
* @author wanguanglu
* @date   2016/09/23 11:12:07
* @brief 
*  
**/

#include "trainer.h"

#ifndef NLU_MAXENT_GIS_TRAINER_H
#define NLU_MAXENT_GIS_TRAINER_H

namespace nlu {
namespace maxent {


class GisTrainer : public Trainer {

public:
    GisTrainer() : Trainer(),
        _correct_constant(0) {}

    virtual ~GisTrainer();

    virtual int train (
        DataReader& data_reader, LinearModel& model,
        int iter, float tol, float sigma2);

private:
    double  _correct_constant;

private:
    virtual int init_train(DataReader& data_reader);
    int cmpt_correct_constant(DataReader& data_reader);
    int newton(float estimate, float observe, float weight, 
                    float sigma2, float& delta, float tol = 1.0E-6);
};


}
}


#endif
