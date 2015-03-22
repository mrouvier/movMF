#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <boost/unordered_map.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include "movmf.h"


using namespace std;
using namespace boost;


int main(int argc, char** argv) {

    string train = argv[1];
    string model = argv[2];


    boost::unordered_map<string, Eigen::VectorXd> h_x = read_embedding_file(train);
    h_x = normalize(h_x);

    vector<vmf> mixture = load_mixture( model );

    for(boost::unordered_map<string, Eigen::VectorXd>::iterator iter = h_x.begin(); iter != h_x.end(); iter++) {
        cout<<iter->first<<" ";
        loglikelihood(mixture, iter->second);
    }

    return 0;
}

