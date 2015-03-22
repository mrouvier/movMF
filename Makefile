movmf:src/movmf.cc
	g++ -O3 -o bin/movmf src/movmf.cc -Ieigen3/ -I/opt/local/include/

generate_date:
	python data/generate_data.py > data/vec

test:
	./bin/movmf data/vec
