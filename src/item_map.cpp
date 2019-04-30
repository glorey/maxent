/**
* @file   item_map.cpp
* @author wanguanglu
* @date   2016/09/22 01:23:58
* @brief 
*  
**/

#include <glog.h>

#include "item_map.h"

namespace nlu {
namespace maxent {

int ItemMap::add_item(std::string item) {

    int id = 0;
    if ( (id=get_item_id(item)) >= 0) {
        return id;
    }

    _str_to_id[item] = _item_num;
    _id_to_str.push_back(item);
    _item_num ++;
    return _item_num-1;
}

int ItemMap::get_item_id(std::string item) {
    boost::unordered_map<std::string, int>::iterator it;

    it = _str_to_id.find(item);
    if (it == _str_to_id.end()) {
        return -1;
    }

    return it->second;
}

int ItemMap::get_item_str(int id, std::string& item) {
    if (id < 0 || id >= _item_num) {
        log_warn("index out of boundary.");
        return -1;
    }

    item = _id_to_str[id];

    return 0;
}

int ItemMap::item_num() {
    return _item_num;
}


}
}
