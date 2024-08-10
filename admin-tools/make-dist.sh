#!/bin/bash
PACKAGE=pymathics-graph

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
python setup.py bdist_wheel --universal
mv -v dist/pymathics_graph-${__version__}-{py2.,}py3-none-any.whl
python ./setup.py sdist
finish
