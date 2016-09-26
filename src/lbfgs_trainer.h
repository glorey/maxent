/**
* @file   lbfgs_trainer.h
* @author wanguanglu
* @date   2016/09/25 14:29:16
* @brief 
*  
**/

#include "lbfgs.h"

#include "trainer.h"

namespace nlu {
namespace maxent {

class LbfgsTrainer : public Trainer {

public:
    LbfgsTrainer() : Trainer(),
        _weight_mat(NULL), _model(NULL) {}
    virtual ~LbfgsTrainer(){}
    virtual int train(
        DataReader& data_reader, LinearModel& model,
        int iter, float tol, float sigma2);

private:
    virtual int init_train(DataReader& data_reader);

    
    lbfgsfloatval_t evaluate(
            const lbfgsfloatval_t *x,
            lbfgsfloatval_t *g,
            const int n,
            const lbfgsfloatval_t step
    );
    
    
    int progress(
            const lbfgsfloatval_t *x,
            const lbfgsfloatval_t *g,
            const lbfgsfloatval_t fx,
            const lbfgsfloatval_t xnorm,
            const lbfgsfloatval_t gnorm,
            const lbfgsfloatval_t step,
            int n,
            int k,
            int ls
    );

    int update_weight();

    
    static lbfgsfloatval_t _evaluate(
                void *instance,
                const lbfgsfloatval_t *x,
                lbfgsfloatval_t *g,
                const int n,
                const lbfgsfloatval_t step) {
        return reinterpret_cast<LbfgsTrainer*>(instance)->evaluate(x, g, n, step);
    }
    
    static int _progress(
                void *instance,
                const lbfgsfloatval_t *x,
                const lbfgsfloatval_t *g,
                const lbfgsfloatval_t fx,
                const lbfgsfloatval_t xnorm,
                const lbfgsfloatval_t gnorm,
                const lbfgsfloatval_t step,
                int n,
                int k,
                int ls) {
        return reinterpret_cast<LbfgsTrainer*>(instance)->
                progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
    }

private:
    lbfgsfloatval_t*  _weight_mat;
    lbfgs_parameter_t _lbfgs_param;
    LinearModel*      _model;
    DataReader*       _data_reader;
    float             _sigma2;
    int               _correct_num;
};


}
}
