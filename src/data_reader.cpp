/**
* @file   data_reader.cpp
* @author wanguanglu
* @date   2016/09/22 14:12:06
* @brief 
*  
**/

#include <fstream>
#include <sstream>

#include <boost/lexical_cast.hpp>

#include <logging.h>

#include "data_reader.h"

namespace nlu {
namespace maxent {

int DataReader::check_binary_feature(const char* file_name) {
    std::ifstream reader(file_name);
    std::string line;
    int is_binary = 0;

    if (!reader.is_open()) {
        log_warn("open file %s for read failed.", file_name);
        return -1;
    }

    while (std::getline(reader, line)) {
        if (line.length() == 0) {
            continue;
        }
        break;
    }

    if (line.length() == 0) {
        log_warn("input file is empty.");
        return -1;
    }

    //get the first non-empty line
    std::istringstream line_reader(line);
    std::string label;
        
    //read label
    line_reader>>label;

    is_binary = 0;
    std::string feat;
    while(line_reader>>feat) {
        int pos = feat.find(':');

        if (pos == std::string::npos) {
            is_binary = 1;
            break;
        }

        try {
            boost::lexical_cast<float>(feat.substr(pos+1));
        } catch (boost::bad_lexical_cast&) {
            is_binary = 1;
            break;
        }
    }

    reader.close();

    if (is_binary) {
        log_notice("is binary feature.");
    } else {
        log_notice("is non-binary feature.");
    }

    return is_binary;
}

int DataReader::parse_line(std::string line, Event& event, int is_binary) {
    std::istringstream line_reader(line);
    std::string label;
    std::string feat;

    event.clear();
    
    line_reader>>label;

    int feat_id  = 0;
    int label_id = 0;
    float val    = 0.0;

    label_id = _label_map.add_item(label);
    event.set_label(label_id);
    event.set_count(1);

    while (line_reader>>feat) {
        if (is_binary) {
            feat_id = _feat_map.add_item(feat);
            event.add_feat(feat_id, (float)1.0);
        } else {
            size_t pos = feat.find(':');
            if (pos == std::string::npos) {
                log_warn("format error. line: %s", line.c_str());
                return -1;
            }

            feat_id = _feat_map.add_item(feat.substr(0, pos));
            val = atof(feat.substr(pos+1).c_str());
            
            if (val <= 0.0) {
                log_warn("feature value need to greater than 0.");
                log_warn("line:%s.", line.c_str());
                return -1;
            }

            event.add_feat(feat_id, val);
        }
    }

    return 0;
}

bool cutoff_event(Event& e) {
    return e._count == 0;
}

struct cutoff_event {
};

int DataReader::merge_events() {
    sort(_events.begin(), _events.end());


    std::vector<Event>::iterator start  = _events.begin();
    std::vector<Event>::iterator end    = _events.end();
    std::vector<Event>::iterator cur;

    while (start != end) {
        cur = start + 1;
        while (cur != end && *cur == *start) {
            start->_count += cur->_count;
            cur->_count = 0;
            cur ++;
        }
        start = cur;
    }

    _events.erase(remove_if(_events.begin(), _events.end(), cutoff_event), _events.end());

    return 0;
}

int DataReader::load_file(const char* file_name) {
    int is_binary = 0;
    int ret = 0;
    std::string line;

    is_binary = check_binary_feature(file_name);
    if (is_binary == -1) {
        log_warn("check binary feature failed.");
        return -1;
    }


    std::ifstream reader(file_name);

    if (!reader.is_open()) {
        log_warn("open file %s for read failed.", file_name);
        return -1;
    }

    while (std::getline(reader, line)) {
        if (line.length() == 0) {
            continue;
        }

        Event event;
        ret = parse_line(line, event, is_binary);
        if (ret  != 0) {
            log_warn("parse line %s failed.", line.c_str());
            return 0;
        }

        _events.push_back(event);
    }

    reader.close();

    ret = merge_events();
    if (ret != 0) {
        log_warn("merge event failed.");
        return -1;
    }

    log_notice("load train file success.");
    return 0;
}

int DataReader::set_feat_label_map(ItemMap& feat_map, ItemMap& label_map) {
    _feat_map  = feat_map;
    _label_map = label_map;
    return 0;
}

int DataReader::print_events() {
    for (int i=0; i<_events.size(); i++) {
        _events[i].print();
    }
    return 0;
}


}
}
