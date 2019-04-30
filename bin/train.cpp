/**
* @file   train.cpp
* @author wanguanglu
* @date   2016/09/22 15:07:12
* @brief 
*  
**/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>

#include <glog.h>

#include "data_reader.h"
#include "maxent.h"
#include "gis_trainer.h"
#include "lbfgs_trainer.h"


struct TrainConf {
    int   _iter;     //迭代次数
    float _gauss;    //高斯先验
    int   _binary;   //输出模型是否为二进制方式

    int   _train_method; //0 GIS 1 LBFGS

    char* _model_file;   //model file

    char** _inputs;
    int    _inputs_num;
};

char* get_opt_str(const char* str) {

    char* result = (char*)malloc(strlen(str)+1);
    if (result == NULL) {
        return NULL;
    }
    strcpy(result, str);

    return result;
}

int print_help() {
    printf("Purpos:\n");
    printf("  a command to train a maxent model\n");
    printf("  example: train  train_file --model model_file --iter 10 --gis --gauss 0.6\n\n");

    printf("Usage: train [OPTION]... [FILES]...\n");
    printf("    -h          --help      Print help and exit\n");
    printf("    -mSTRING    --model     Set model file name\n");
    printf("    -b          --binary    Save model in binary format(default=off)\n");
    printf("    -iINT       --iter      iterations for training algorithm(default=30)\n");
    printf("    -gFLOAT     --gauss     set Gaussian prior, disable if 0 (default='0.0')\n");

    printf("\n");
    printf("                --lbfgs     use L-BFGS parameter estimation (default)\n");
    printf("                --gis       use GIS parameter estimation\n");

    return 0;
}


int parse_cmdline(int argc, char* argv[], TrainConf& conf) {

    //default value
    conf._iter   = 30;
    conf._gauss  = 0.0;
    conf._binary = 0;
    conf._train_method = 1;
    conf._model_file   = NULL;
    conf._inputs_num   = 0;

    struct option options[] = {
        { "help",  0, NULL, 'h' },
        { "model", 1, NULL, 'm' },
        { "binary", 0, NULL, 'b' },
        { "iter",   1, NULL, 'i' },
        { "gauss",  1, NULL, 'g' },
        { "lbfgs",  0, NULL, 0},
        { "gis",    0, NULL, 0}
    };


    char* stop_char = NULL;
    int c = 0;
    int option_index = 0;

    while (true) {
        c = getopt_long(argc, argv, "h:m:b:i:g:c", options, &option_index);
        if (c == -1) {
            break;
        }
        
        switch (c) {
            case 'h':
                print_help();
                return -1;

            case 'm':
                if (conf._model_file != NULL) {
                    printf("--model (-m) option given more than once.\n");
                    return -1;
                }
                conf._model_file = get_opt_str(optarg);
                break;

            case 'b':
                conf._binary = 1;
                break;

            case 'i':
                conf._iter = strtol(optarg, &stop_char, 0);
                break;

            case 'g':
                try {
                    conf._gauss = (float)strtod(optarg, NULL);
                    break;
                } catch (...) {
                    log_warn("format incorrect.");
                    return -1;
                }

            case 0:
                if (strcmp(options[option_index].name, "lbfgs") == 0) {
                    conf._train_method = 1;
                    break;
                }

                if (strcmp(options[option_index].name, "gis") == 0) {
                    conf._train_method = 0;
                    break;
                }

            case '?':
                return -1;

            default:
                printf("option unknown: %c\n", c);
                return -1;
        }
    }

    if (optind < argc) {
        int i = 0;
        conf._inputs_num = argc - optind;
        conf._inputs     =
            (char **)(malloc ((conf._inputs_num)*sizeof(char*)));

        while (optind < argc) {
            conf._inputs[i++] = get_opt_str(argv[optind++]) ;
        }
    }


    return 0;
}

using namespace nlu::maxent;

int main(int argc, char* argv[]) {

    int ret = 0;
    TrainConf conf;

    ret = parse_cmdline(argc, argv, conf);
    if (ret != 0) {
        print_help();
        return -1;
    }

    if (conf._model_file == NULL) {
        printf("model file not set.\n\n");
        print_help();
        return -1;
    }

    if (conf._inputs_num != 1) {
        printf("input format error.");
        print_help();
        return -1;
    }

    glog_init("maxent", "log", "maxent");

    std::string train_file;
    std::string model_file;

    train_file = conf._inputs[0];
    model_file = conf._model_file;

    DataReader data_reader;
    ret = data_reader.load_file(train_file.c_str());
    if (ret != 0) {
        log_warn("load file failed.");
        return -1;
    }

    LinearModel model;
    Trainer* trainer;

    if (conf._train_method == 0) {
        trainer = new GisTrainer();
    } else if (conf._train_method == 1) {
        trainer = new LbfgsTrainer();
    } else {
        log_warn("unsupported train method.");
        return -1;
    }

    ret = trainer->train(data_reader, model, conf._iter, 1E-05, conf._gauss);
    if (ret != 0) {
        log_warn("train maxent model failed.");
        return -1;
    }

    model.save_model(conf._model_file);

    return 0;
}
