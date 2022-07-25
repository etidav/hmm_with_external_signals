ARG TENSORFLOW_BASE=tensorflow/tensorflow:2.5.0-gpu
FROM ${TENSORFLOW_BASE} as base_img

ARG TENSORFLOW_BASE
RUN echo "" && echo "Friendly info: this docker uses the base image: ${TENSORFLOW_BASE}" && echo ""

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
RUN apt-get update

#COPY docker/requirements/fourcats-ubuntu-requirements.txt /ubuntu-requirements.txt
#COPY docker/requirements/E298A3A825C0D65DFD57CBB651716619E084DAB9.asc /
#RUN apt-key add /E298A3A825C0D65DFD57CBB651716619E084DAB9.asc

# install a few packages required to add custom repositories
#RUN DEBIAN_FRONTEND=noninteractive apt-get update --fix-missing && \
#    DEBIAN_FRONTEND=noninteractive apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common && \
#    apt-get clean

# setup of debian requirements (mostly R install which needs to be 3.5+)
#RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/' && \
#    add-apt-repository ppa:marutter/c2d4u3.5 -y && \
#    DEBIAN_FRONTEND=noninteractive apt-get update -y --fix-missing && \
#    DEBIAN_PACKAGES=$(egrep -v "^\s*(#|$)" /ubuntu-requirements.txt) && \
#    DEBIAN_FRONTEND=noninteractive apt-get install -y $DEBIAN_PACKAGES && \
#    apt-get clean && \
#    rm /ubuntu-requirements.txt

# fix for installing devtools
#RUN apt-get install -y libgit2-dev && \
#    apt-get install -y libcurl4-openssl-dev && \
#    apt-get install -y libgit2-dev && \
#    R -e "install.packages('devtools')"



# main R libraries install: latest source from github
#RUN Rscript -e "require('devtools'); install_version('forecast', version='8.15');" && \
#    Rscript -e "require('devtools'); devtools::install_github('ellisp/forecastHybrid/pkg', quick=T);" && \
#    Rscript -e "require('devtools'); devtools::install_github('robjhyndman/M4metalearning', quick=T);" && \
#    Rscript -e "require('devtools'); install.packages('tsfeatures');" && \
#    Rscript -e "require('devtools'); devtools::install_github('pmontman/customxgboost', quick=T);"

# alternative R libraries install: from a dump of github repos (last made on 2020-10-23)
#RUN mkdir /r_pkgs
#COPY docker/requirements/customxgboost.tar.gz \
#     docker/requirements/forecast.tar.gz \
#     docker/requirements/forecastHybrid.tar.gz \
#     docker/requirements/M4metalearning.tar.gz \
#     /r_pkgs/
#RUN Rscript -e "require('devtools'); untar('/r_pkgs/forecast.tar.gz', exdir='/r_pkgs/'); devtools::install('/r_pkgs/forecast', quick=T);" && \
#    Rscript -e "require('devtools'); untar('/r_pkgs/forecastHybrid.tar.gz', exdir='/r_pkgs/'); devtools::install('/r_pkgs/forecastHybrid', quick=T);" && \
#    Rscript -e "require('devtools'); untar('/r_pkgs/M4metalearning.tar.gz', exdir='/r_pkgs/'); devtools::install('/r_pkgs/M4metalearning', quick=T);" && \
#    Rscript -e "require('devtools'); install.packages('tsfeatures');" && \
#    Rscript -e "require('devtools'); untar('/r_pkgs/customxgboost.tar.gz', exdir='/r_pkgs/'); devtools::install('/r_pkgs/customxgboost', quick=T);"

# make sure we use python3.7 (prefect requirement)
#RUN ln -sf /usr/bin/python3.7 /usr/bin/python3
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 100
#RUN python3.7 -m pip install --upgrade pip

# Install python 3.7
 RUN apt install software-properties-common -y
 RUN add-apt-repository ppa:deadsnakes/ppa
 RUN apt install python3.7 -y

# Add 3.7 to the available alternatives
 RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
 RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 100
 RUN python3.7 -m pip install --upgrade pip

# python install from pip3 - only to have a tensorboard version that matches tensorflow
COPY docker/requirements/requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt && \
    rm /requirements.txt

# Make data and libs accressible
COPY data /hmm_with_external_signals/data
COPY run /hmm_with_external_signals/run
COPY result /hmm_with_external_signals/result
COPY model /hmm_with_external_signals/model
ENV PYTHONPATH=$PYTHONPATH:/hmm_with_external_signals

# Define workdir
WORKDIR /hmm_with_external_signals

# Define the correct entrypoint
ENTRYPOINT []


# installs a better version of rpy2
#COPY docker/requirements/rpy2.tar.gz /rpy2.tar.gz
#RUN tar xzf rpy2.tar.gz && \
#    cd rpy2-rpy2-* && \
#    python3 setup.py install

# Make libs accessible when mounting with the right /workspace/
#ENV PYTHONPATH=$PYTHONPATH:/workspace/fourcats/subprojects:/workspace/fourcats/subprojects/forecast/:/workspace/fourcats/subprojects/dataset/:/workspace/fourcats/subprojects/protobuf/

# Set Fourcats log env to 'debug' for logging package
#ENV LOG='debug'
