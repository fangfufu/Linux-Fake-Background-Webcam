URL=https://blob.danieldk.eu/libtensorflow/libtensorflow-cpu-linux-x86_64-avx-fma-1.15.0.tar.gz
mkdir tmp
cd tmp
wget $URL -O libtensorflow.tar.gz
tar -xf libtensorflow.tar.gz
rm libtensorflow.tar.gz
rm ../node_modules/@tensorflow/tfjs-node/deps/* -rf
mv * ../node_modules/@tensorflow/tfjs-node/deps/
cd ..
rm -rf tmp
