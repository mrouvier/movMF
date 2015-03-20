#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/unordered_map.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>

using namespace boost;
using namespace std;


class vmf {
    public:
        double kappa;
        double alpha;
        Eigen::VectorXd mu;

        vmf() {
            kappa = 0;
            alpha = 0;
            mu =  Eigen::VectorXd::Zero(200);
        }

};

static inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}


static inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}


static inline std::string &trim(std::string &s) {
    return ltrim(rtrim(s));
}



boost::unordered_map<string, Eigen::VectorXd> read_embedding_file(const string& __file) {
    boost::unordered_map<string, Eigen::VectorXd> h;

    string line;
    fstream f(__file.c_str());
    while(getline(f, line)) {
        line = trim(line);
        vector<string> vec;
        split(vec, line, is_any_of(" "));

        Eigen::VectorXd v(vec.size()-1);
        for(int i = 1; i < vec.size(); i++) {
            v[i-1] = lexical_cast<double>( vec[i] );
        }
        h[ vec[0] ] = v;
    }
    f.close();


    return h;
}

double scoring_cosine(Eigen::VectorXd x, Eigen::VectorXd y) {
    double a = ( (x.transpose() * y) / ( sqrt( x.transpose()  * x) * sqrt( y.transpose()  * y) ) )[0];
    return a;
    return (1 - (1 - (acos(a-0.001) ) / 3.1416));
}



vector< Eigen::VectorXd > kmeans(boost::unordered_map<string, Eigen::VectorXd> h, int nb_cluster, int nb_iteration) {
    int size = h.begin()->second.size();
    vector< Eigen::VectorXd > points(nb_cluster);

    //Initialize centroids data
    boost::random::mt19937 rng;
    boost::random::uniform_int_distribution<> rnd(0,h.size()-1);

    for(int i = 0; i < nb_cluster; i++) {
        boost::unordered_map<string, Eigen::VectorXd>::iterator random_it = h.begin();
        std::advance(random_it, rnd(rng) );
        points[ i ] = random_it->second;
        //cout<< "DEBUG : " << points[ i ] <<endl;
    }

    for(int iteration = 0; iteration < nb_iteration; iteration++) {
        double obj_fct = 0;

        vector<int> new_values( nb_cluster );
        vector<Eigen::VectorXd> new_points( nb_cluster );
        for(int i = 0; i < nb_cluster; i++) {
            new_values[i] = 0;
            new_points[i] = Eigen::VectorXd::Zero(size);
        }

        for(boost::unordered_map<string, Eigen::VectorXd>::iterator iter = h.begin(); iter != h.end(); iter++) {

            double best_score = -20000000;
            int best_points = -1;

            for(int i = 0; i < nb_cluster; i++) {
                double temp = scoring_cosine(iter->second, points[i]);
                if(best_score < temp) {
                    best_score = temp;
                    best_points = i;
                }
            } 

            new_points[ best_points ] += iter->second;
            new_values[ best_points ] += 1;

            //cerr<< iter->first << " " << best_score << " " << size <<endl;
            obj_fct += best_score;
        }

        cerr<< "Objective function : "<< obj_fct/h.size() << endl;

        for(int i = 0; i < nb_cluster; i++) {
            cout<< "NEW VALUES : "<< new_values[i] <<endl;
            points[ i ] = new_points[i] / (double)new_values[i];
        }

    }

    return points;
}

double lengthNorm(Eigen::VectorXd l) {

    double c = 0;

    for(int i = 0; i < l.size(); i++) {
        c += l[i]*l[i];    
    }

    return sqrt(c);
}


boost::unordered_map<string, Eigen::VectorXd> normalize(boost::unordered_map<string, Eigen::VectorXd> h_x) {

    boost::unordered_map<string, Eigen::VectorXd> h;

    for(boost::unordered_map<string, Eigen::VectorXd>::iterator a = h_x.begin(); a != h_x.end(); a++) {
        h[ a->first ] = a->second / lengthNorm( a->second );
    }

    return h;
}

double logbesseli(double nu, double x) {
    double frac = x/nu;
    double square = 1 + frac*frac;
    double root = sqrt( square );
    double eta = root + log(frac) - log(1+root);
    double approx = -1 * log( sqrt( 2 * 3.1416* nu) ) + nu*eta - 0.25*log(square);
    return approx;
}



void movmf(boost::unordered_map<string, Eigen::VectorXd> h, int nb_cluster, int nb_iteration) {
    int size = h.begin()->second.size();
    int diff = 1;
    double epsilon = 0.0001;
    double value = 100;
    double kappaMax = 100.0;
    double kappaMin = 1.0;

    vector< Eigen::VectorXd > pt = kmeans(h, nb_cluster, 4);
    vector< vmf > mixture( nb_cluster );
    for(int i = 0; i < nb_cluster; i++) {
        mixture[i].kappa = 1.0;
        mixture[i].alpha = (double)1.0/(double)nb_cluster;
        mixture[i].mu = pt[i];

    }


    for(int iteration = 0; iteration < nb_iteration; iteration++) {
        cerr<< "Iteration : "<<iteration<<endl;

        vector< double > logNormalize(nb_cluster);
        for(int i = 0; i < nb_cluster; i++) {
            logNormalize[i] = log( mixture[i].alpha ) + (size/2-1)*log( mixture[i].kappa ) - (size/2) * log(2*3.1416) - logbesseli(size/2-1, mixture[i].kappa );
        }

        //Expectation
        double logLikeLiHood = 0;
        boost::unordered_map<string, double > logSum;
        boost::unordered_map<string, vector<double > > logProbMat;
        for(boost::unordered_map<string, Eigen::VectorXd>::iterator iter = h.begin(); iter != h.end(); iter++) {
            double somme = 0;
            for(int i = 0; i < nb_cluster; i++) {
                double temp =  ( ( iter->second.transpose() * mixture[i].mu * mixture[i].kappa )[0]  + logNormalize[i]  ) / 100;
                logProbMat[iter->first].push_back( temp );
                //cerr<< temp << " " << exp( temp ) <<endl;
                logSum[iter->first] += exp( temp );
            }
            logSum[iter->first] = log( logSum[iter->first] ); 
            for(int i = 0; i < nb_cluster; i++) {
                logProbMat[iter->first][i] -= logSum[ iter->first ];
            }
            logLikeLiHood += logSum[ iter->first ];
        }

        cerr<< "Obj fonction : "<< logLikeLiHood/h.size() <<endl;

        //Maximization
        for(int i = 0; i < nb_cluster; i++) {

            double kappa = 0;
            double alpha = 0;
            Eigen::VectorXd mu = Eigen::VectorXd::Zero(size);

            for(boost::unordered_map<string, Eigen::VectorXd>::iterator iter = h.begin(); iter != h.end(); iter++) {
                alpha += exp( logProbMat[ iter->first ][i] ) / h.size();
                mu += iter->second * exp( logProbMat[ iter->first ][i] ) / h.size();
            }
            double rbar = lengthNorm( mu ) / ( 1 * alpha) ;
            mu = mu / lengthNorm( mu );
            kappa = ( rbar * size - rbar*rbar*rbar ) / ( 1 - rbar*rbar );

            mixture[i].alpha = alpha;
            mixture[i].kappa = kappa;
            mixture[i].mu = mu;  
        }


    }


}


int main(int argc, char** argv) {

    string train = argv[1];
    boost::unordered_map<string, Eigen::VectorXd> h_x = read_embedding_file(train);
    h_x = normalize(h_x);

    movmf(h_x, 512, 20);

    return 0;
}

