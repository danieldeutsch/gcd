mkdir -p lib
wget https://nlp.cs.nyu.edu/evalb/EVALB.tgz -O lib/EVALB.tgz
tar xvzf lib/EVALB.tgz -C lib
cd lib/EVALB; make; cd ../..
