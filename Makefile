BOOST_LIB=/storage/raid1/homedirs/mickael.rouvier/appli/local/lib/
BOOST_INCLUDE=/storage/raid1/homedirs/mickael.rouvier/appli/local/include/

#BOOST_LIB=/opt/local/lib
#BOOST_INCLUDE=/opt/local/include

all:movmf_train movmf_test movmf_similar

movmf_train:src/movmf_train.cc
	g++ -O3 -g -o bin/movmf_train src/movmf_train.cc -Ieigen3/ -Isrc/ -I${BOOST_INCLUDE}  -L${BOOST_LIB} -lboost_program_options

movmf_test:src/movmf_test.cc
	g++ -O3 -o bin/movmf_test src/movmf_test.cc -Ieigen3/ -Isrc/ -I${BOOST_INCLUDE}

movmf_similar:src/movmf_similar.cc
	g++ -O3 -o bin/movmf_similar src/movmf_similar.cc -Ieigen3/ -Isrc/ -I${BOOST_INCLUDE}

generate_data:
	python data/generate_data.py > data/vec
	cat data/vec | perl -MList::Util -e 'print List::Util::shuffle <>' | head -n 256 > data/initialize

train:
	./bin/movmf_train --nb_mixture 128 --nb_iteration_em 10  --train data/vec --save mixture.txt

train_initialize:
	./bin/movmf_train --initialize data/initialize  --nb_mixture 128 --nb_iteration_em 10  --train data/vec --save mixture.txt


test:
	./bin/movmf_test data/vec mixture.txt

similar:
	./bin/movmf_similar data/vec mixture.txt

clean:
	rm bin/movmf_train bin/movmf_test bin/movmf_similar
