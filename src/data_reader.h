/**
* @file   data_reader.h
* @author wanguanglu
* @date   2016/09/22 10:21:32
* @brief 
*  
**/

#include <vector>
#include <string>

#include "event.h"
#include "item_map.h"

#ifndef NLU_MAXENT_DATA_READER_H
#define NLU_MAXENT_DATA_READER_H

namespace nlu {
namespace maxent {

class DataReader {

private:
    std::vector<Event> _events;

    int _event_count;

    ItemMap _feat_map;
    ItemMap _label_map;

private:
    int parse_line(std::string line, Event& event, int is_binary);
    int check_binary_feature(const char* file_name);


public:
    int load_file(const char* file_name);
    int merge_events();
    int set_feat_label_map(ItemMap& feat_map, ItemMap& label_map);

    inline int feat_num()  { return _feat_map.item_num(); };
    inline int label_num() { return _label_map.item_num();};
    inline std::vector<Event>& events() { return _events; };

    //for debug
    int print_events();

friend class Trainer;
friend class GisTrainer;
};


}
}

#endif
