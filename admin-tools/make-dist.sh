#!/bin/bash
PACKAGE=mathics3-graph

# FIXME put some of the below in a common routine
function finish {
  cd $mathics_graph_owd
}

cd $(dirname ${BASH_SOURCE[0]})
mathics_graph_owd=$(pwd)
trap finish EXIT

if ! source ./pyenv-versions ; then
    exit $?
fi


cd ..
source pymathics/graph/version.py
echo $__version__

if ! pyenv local $pyversion ; then
    exit $?
fi
pyenv local 3.13
pip wheel --wheel-dir=dist .
python -m build --sdist
finish
