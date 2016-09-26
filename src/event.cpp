/**
* @file   event.cpp
* @author wanguanglu
* @date   2016/09/22 10:37:59
* @brief 
*  
**/

#include <stdio.h>

#include "event.h"


namespace nlu {
namespace maxent {

bool FeatPair::operator< (const FeatPair& y) const {
    if (_feat_id < y._feat_id) {
        return true;
    } else if (_feat_id > y._feat_id) {
        return false;
    }

    if (_val < y._val) {
        return true;
    } else if (_val < y._val) {
        return false;
    }

    return false;
}

bool FeatPair::operator== (const FeatPair& y) const {
    if (_feat_id != y._feat_id) {
        return false;
    } 

    if (_val != y._val) {
        return false;
    } 

    return true;
}

int Event::add_feat(int feat_id, float val) {
    FeatPair feat_pair(feat_id, val);
    int pos = 0;
    
    for (pos=0; pos<_contexts.size(); pos++) {
        if (feat_pair < _contexts[pos]) {
            break;
        }
    }
    _contexts.insert(_contexts.begin()+pos, feat_pair);
    return 0;
}

int Event::clear() {
    _contexts.clear();
    _label = -1;
    _count = -1;
}

bool Event::operator< (const Event& y) const {
    if (_label < y._label) {
        return true;
    } else if (_label > y._label) {
        return false;
    }

    if (_contexts.size() < y._contexts.size()) {
        return true;
    } else {
        return false;
    }

    for (int i=0; i<_contexts.size(); i++) {
        if (_contexts[i] < y._contexts[i] ) {
            return true;
        } else if (y._contexts[i] < _contexts[i]) {
            return false;
        }
    }

    return false;
}

bool Event::operator == (const Event& y) const {
    if (_label != y._label) {
        return false;
    } 

    if (_contexts.size() != y._contexts.size()) {
        return false;
    } 

    for (int i=0; i<_contexts.size(); i++) {
        if ( !(_contexts[i] == y._contexts[i]) ) {
            return false;
        } 
    }

    return true;
}

int Event::print() {
    printf ("%d\t%d", _count, _label);
    for (int i=0; i<_contexts.size(); i++) {
        printf("\t%d:%f", _contexts[i]._feat_id, _contexts[i]._val);
    }
    printf ("\n");
    return 0;
}

}
}
