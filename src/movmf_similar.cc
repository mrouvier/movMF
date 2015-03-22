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

    for(int i = 0; i < mixture.size(); i++) {

        double best_score = -2000;
        string best_name = "";

        for(boost::unordered_map<string, Eigen::VectorXd>::iterator iter = h_x.begin(); iter != h_x.end(); iter++) {
            double temp = scoring_cosine(iter->second, mixture[i].mu);
            if(best_score < temp) {
                best_score = temp;
                best_name = iter->first;
            }
        }

        cout<<"mixture : "<<i<<endl;
        cout<<" -- name: "<<best_name<<" , "<<best_score<<endl;

    }

    return 0;
}

