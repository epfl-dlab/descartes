FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

### Use bash as the default shelll
RUN chsh -s /bin/bash
SHELL ["bash", "-c"]

### Install basics
RUN apt-get update && \
    apt-get install -y openssh-server sudo nano screen wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
    apt-get clean

### Add user (get UID using: id -u username -> UID)
# TODO: UPDATE (start)
ARG UNAME=msakota
ARG UID=502
#ARG UPASSWORD=password
# TODO: UPDATE (end)

RUN groupadd -g 11106 DLAB-StaffU && groupadd -g 60220 dlab_AppGrpU && useradd -rm -d /home/$UNAME -s /bin/bash -g 11106 -G sudo,60220 -u $UID $UNAME
RUN echo "$UNAME ALL=(ALL:ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/$UNAME
WORKDIR /home/$UNAME
#COPY .ssh /home/$UNAME/.ssh
#RUN echo "$UNAME:$UPASSWORD" | chpasswd

### Login
USER $UNAME

### Install Anaconda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh && \
    sudo /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    sudo ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    sudo find /opt/conda/ -follow -type f -name '*.a' -delete && \
    sudo find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    sudo /opt/conda/bin/conda clean -afy

ENV PATH /opt/conda/bin:$PATH

### Install Environment
ENV envname transformers
RUN sudo env "PATH=$PATH" conda update conda && \
    sudo chown $UID -R /home/$UNAME && \
    conda create --name $envname python=3.8
    


### Install a version of pytorch that is compatible with the installed cudatoolkit
RUN conda install -n $envname pytorch=1.7.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch
#COPY requirements.yaml /tmp/requirements.yaml
#RUN conda env update --name $envname --file /tmp/requirements.yaml --prune
#COPY pip_requirements.txt /tmp/pip_requirements.txt
#RUN conda run -n $envname pip install -r /tmp/pip_requirements.txt
RUN conda install -n $envname ipykernel && \
    conda install -n $envname -c conda-forge sentencepiece && \
    conda install -n $envname -c conda-forge sacrebleu && \
    conda install -n $envname -c conda-forge gitpython && \
    conda run -n $envname pip install rouge-score && \
    conda run -n $envname pip install nltk && \
    conda run -n $envname pip install rouge-score && \
    conda run -n $envname pip install protobuf && \
    conda run -n $envname pip install py7zr && \
    conda run -n $envname pip install datasets

    
RUN echo "conda activate $envname" >> ~/.bashrc && \
    conda run -n $envname python -m ipykernel install --user --name=$envname


#EXPOSE 22
#EXPOSE 3534
EXPOSE 8888


ENTRYPOINT /bin/bash


# Mounting dlabdata1 to docker container
# export NFS_VOL_NAME=dlabdata1 NFS_LOCAL_MNT=/dlabdata1 NFS_SERVER=ic1files.epfl.ch NFS_SHARE=/ic_dlab_1_files_nfs/dlabdata1 NFS_OPTS=vers=3,soft
# docker run --mount "src=$NFS_VOL_NAME,dst=$NFS_LOCAL_MNT,volume-opt=device=:$NFS_SHARE,\"volume-opt=o=addr=$NFS_SERVER,$NFS_OPTS\",type=volume,volume-driver=local,volume-opt=type=nfs" -ti -P --rm --name t18dm t18dm bash
# conda env update --name env_name