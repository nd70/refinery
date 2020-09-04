# Install nds2
cd $HOME/refinery
wget http://www.lsc-group.phys.uwm.edu/daswg/download/software/source/nds2-client-0.15.2.tar.gz
tar -xzvf nds2-client-0.15.2.tar.gz
mv nds2-client-0.15.2.tar.gz nds2-client-0.15.2/
cd $HOME/refinery/nds2-client-0.15.2/
mkdir obj
cd obj
cmake -DCMAKE_INSTALL_PREFIX=$HOME/refinery/ -DCMAKE_C_COMPILER=$(which cc) -DCMAKE_CXX_COMPILER=$(which c++) ..
cmake --build .
cmake --build . -- install
