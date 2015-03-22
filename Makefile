all:movmf_train movmf_test movmf_similar

movmf_train:src/movmf_train.cc
	g++ -O3 -o bin/movmf_train src/movmf_train.cc -Ieigen3/ -Isrc/ -I/opt/local/include/  -L/opt/local/lib/ -lboost_program_options-mt

movmf_test:src/movmf_test.cc
	g++ -O3 -o bin/movmf_test src/movmf_test.cc -Ieigen3/ -Isrc/ -I/opt/local/include/

movmf_similar:src/movmf_similar.cc
	g++ -O3 -o bin/movmf_similar src/movmf_similar.cc -Ieigen3/ -Isrc/ -I/opt/local/include/

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
