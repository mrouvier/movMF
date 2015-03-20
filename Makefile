movmf:src/movmf.cc
    g++ -O3 -o bin/movmf src/movmf.cc -Ieigen3/ -I/opt/local/include/
test:
    ./bin/movmf data/frwiki.good.w2v
