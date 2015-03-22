#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <boost/unordered_map.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include "movmf.h"
#include <boost/program_options.hpp>


using namespace std;
using namespace boost;
using namespace boost::program_options;



int main(int argc, char** argv) {

    string train, save, initialize = "";
    int nb_mixture, nb_iteration_em, nb_iteration_kmeans;

    options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("nb_mixture", value<int>(&nb_mixture)->default_value(32), "number of von mises-fisher distribution")
        ("nb_iteration_kmeans", value<int>(&nb_iteration_kmeans)->default_value(1), "number of iteration of EM algorithm")
        ("nb_iteration_em", value<int>(&nb_iteration_em)->default_value(3), "number of iteration of EM algorithm")
        ("initialize", value<string>(&initialize), "file containing initializing points")
        ("train", value<string>(&train), "training file")
        ("save", value<string>(&save), "save mixture of von mises-fisher")
        ;

    variables_map vm;

    try {
        store(command_line_parser(argc, argv).options(desc).run(), vm);
        notify(vm);    
    }
    catch(...) {
        cout << desc << endl;
        return 0;
    }

    if (vm.count("help")) {
        cout << desc << endl;
        return 0;
    }


    boost::unordered_map<string, Eigen::VectorXd> h_x = read_embedding_file(train);
    h_x = normalize(h_x);

    boost::unordered_map<string, Eigen::VectorXd> h_point;
    if( initialize != "") {
        h_point = read_embedding_file(initialize);
        h_point = normalize(h_point);
    }

    vector<vmf> mixture = movmf(h_x, h_point, nb_mixture, nb_iteration_kmeans, nb_iteration_em);
    save_mixture( mixture, save );

    return 0;
}

