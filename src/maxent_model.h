/**
* @file   maxent.h
* @author wanguanglu
* @date   2016/03/22 18:00:31
* @brief 
*  
**/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <map>
#include <vector>
#include <string.h>
#include <sstream>

#include <logging.h>

#include "maxent_commons.h"
#include "item_map.h"
#include "event.h"


#ifndef NLU_MAXENT_MAXENT_MODEL_H
#define NLU_MAXENT_MAXENT_MODEL_H

namespace nlu {
namespace maxent { 

class LinearModel {
public:

    LinearModel();
    ~LinearModel();
    
    /**
     * @brief load model
     * @param model_file linear model file path
     * @return  errno
     *          0 success
     *         -1 failed
     */
    int load_model(const char* model_file);

    int save_model(const char* model_file);

    inline int feat_num()  { return _feat_num;  }
    inline int label_num() { return _label_num; }
    inline ItemMap& feat_map()  { return _feat_map;  };
    inline ItemMap& label_map() { return _label_map; };

    /**
     * @brief predict 
     * @param feat  feature 
     * @param score score vector
     * @param label label of max score
     * @return errno 
     *         0 success
     *        -1 failed
     */
    int predict(const float* feat, int feat_len, float* score, int& label) const;

    /**
     * @brief predict 
     * @param feat  feature 
     * @param score score vector
     * @param label label of max score
     * @return errno 
     *         0 success
     *        -1 failed
     */
    int predict(const std::map<int, float>& feat, std::vector<float>& score, int& label) const;

    int predict(const std::vector<FeatPair>& feat, float* score, int& label) const;

private:
    /**
     * @brief calculate score
     * @param feat feature
     * @param label_id label
     * @return score
     */
    float cal_score(const float* feat, int label_id) const;

    /**
     * @brief calculate score
     * @param feat feature
     * @param label_id label
     * @return score
     */
    float cal_score(const std::map<int, float>& feat, int label_id) const;

    float cal_score(const std::vector<FeatPair>& feat, int label_id) const;

    /**
     * @brief normlize score
     * @param score input score vector
     * @return errno
     *         0 success
     *        -1 failed
     */
    int normalize_score(float* score) const;

    /**
     * @brief normlize score
     * @param score input score vector
     * @return errno
     *         0 success
     *        -1 failed
     */
    int normalize_score(std::vector<float>& score) const;


    int set_feat_label_map(ItemMap& feat_map, ItemMap& label_map);

private:
    ItemMap _feat_map;
    ItemMap _label_map;

    /**
     * @NOTICE the 0 dimension is bias
     */
    float* _weight_mat;

    /**
     * @NOTICE bias is added
     */
    int    _feat_num;
    int    _label_num;

friend class Trainer;
friend class GisTrainer;
friend class LbfgsTrainer;
};
 
}
}

#endif
