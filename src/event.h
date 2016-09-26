/**
* @file   event.h
* @author wanguanglu
* @date   2016/09/22 10:26:30
* @brief 
*  
**/


#include <vector>

#ifndef NLU_MAXENT_EVENT_H
#define NLU_MAXENT_EVENT_H


namespace nlu {
namespace maxent {

class FeatPair {
public:
    int   _feat_id;
    float _val;

    FeatPair(int feat_id, float val) :
        _feat_id(feat_id), _val(val) {}

    bool operator< (const FeatPair& y) const;
    bool operator== (const FeatPair& y) const;

};


class Event {
public:
    std::vector<FeatPair> _contexts; 
    int _label;
    int _count;

    Event():_contexts(), _label(0), _count(0){}

public:
    inline void set_label(int label) { _label = label; }
    inline void set_count(int count) { _count = count; }
    int add_feat(int feat_id, float val);
    int clear();
    bool operator< (const Event& y) const;
    bool operator== (const Event& y) const;


    //for debug
    int print();
};


}
}


#endif
