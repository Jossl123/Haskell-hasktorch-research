######################################################
# Installation script for hasktorch
# Author : Josselin GAUTIER
# Date : 2021-04-06
######################################################

git clone https://github.com/hasktorch/hasktorch /tmp/hasktorch_install
#TODO : move these libraries to the user libs so it's not necessary to clone them and have them in every hasktorch project
echo "copying hasktorch"
cp -r /tmp/hasktorch_install/hasktorch .
echo "copying libtorch-ffi"
cp -r /tmp/hasktorch_install/libtorch-ffi .
echo "copying libtorch-ffi-helper"
cp -r /tmp/hasktorch_install/libtorch-ffi-helper .
echo "copying deps"
cp -r /tmp/hasktorch_install/deps .
rm -rf /tmp/hasktorch_install/hasktorch
pushd deps
./get-deps.sh
popd
echo "resolver: lts-20.2

packages:
- .
- libtorch-ffi
- libtorch-ffi-helper
- hasktorch

extra-deps:
- datasets-0.4.0@sha256:9bfd5b54c6c5e1e72384a890cf29bf85a02007e0a31c98753f7d225be3c7fa6a,4929
- require-0.4.10@sha256:41b096daaca0d6be49239add1149af9f34c84640d09a7ffa9b45c55f089b5dac,3759
- indexed-extras-0.2@sha256:e7e498023e33016fe45467dfee3c1379862e7e6654a806a965958fa1adc00304,1349
- normaldistribution-1.1.0.3@sha256:2615b784c4112cbf6ffa0e2b55b76790290a9b9dff18a05d8c89aa374b213477,2160
- term-rewriting-0.4.0.2@sha256:5412f6aa29c5756634ee30e8df923c83ab9f012a4b8797c460af3d7078466764,2740
- git: https://github.com/hasktorch/tintin
  commit: 0d5afba586da01e0a54e598745676c5c56189178
- git: https://github.com/hasktorch/tokenizers
  commit: 9d25f0ba303193e97f8b22b9c93cbc84725886c3
  subdirs:
  - bindings/haskell/tokenizers-haskell
- git: https://github.com/hasktorch/typelevel-rewrite-rules
  commit: 4176e10d4de2d1310506c0fcf6a74956d81d59b6
- git: https://github.com/hasktorch/type-errors-pretty
  commit: 32d7abec6a21c42a5f960d7f4133d604e8be79ec
- git: http://github.com/DaisukeBekki/nlp-tools.git
  commit: 0b6912850dbffd14ed16c3fa26a0773c877ba093
- union-find-array-0.1.0.3@sha256:242e066ec516d61f262947e5794edc7bbc11fd538a0415c03ac0c01b028cfa8a,1372
- clay-0.14.0@sha256:382eced24317f9ed0f7a0a4789cdfc6fc8dd32895cdb0c4ea50a1613bee08af3,2128
- inline-c-0.9.1.10
- inline-c-cpp-0.5.0.2

extra-include-dirs:
  - deps/libtorch/include/torch/csrc/api/include
  - deps/libtorch/include

extra-lib-dirs:
  - deps/libtorch/lib
  - deps/mklml/lib
  - deps/libtokenizers/lib" > stack.yaml

stack build

echo ""
echo "Testing sinus.hs"
stack runghc src/sinus.hs