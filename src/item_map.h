/**
* @file   item_map.h
* @author wanguanglu
* @date   2016/09/22 01:16:48
* @brief 
*  
**/

#include <string>

#include <boost/unordered_map.hpp>

#ifndef NLU_MAXENT_ITEM_MAP_H
#define NLU_MAXENT_ITEM_MAP_H

namespace nlu {
namespace maxent {

class ItemMap {
public:
    ItemMap() : _item_num(0) {}
    ~ItemMap(){}

    int add_item(std::string item);
    int get_item_id(std::string item);
    int get_item_str(int id, std::string& item);
    int item_num();

private:
    boost::unordered_map<std::string, int> _str_to_id;
    std::vector<std::string>               _id_to_str;
    int                                    _item_num;
};

}
}

#endif
