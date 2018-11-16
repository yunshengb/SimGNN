FROM nvcr.io/nvidia/tensorflow:18.05-py3

RUN apt-get update && apt-get install --no-install-recommends -y \
        git \
        python3-tk \
  && rm -rf /var/lib/apt/lists/*

RUN pip install numpy==1.11.3 \
  scipy \
  networkx==1.10 \
  scikit-learn \
  beautifulsoup4 \
  matplotlib \
  lxml


# Installs Java.
ENV JAVA_VER 8
ENV JAVA_HOME /usr/lib/jvm/java-8-oracle
RUN echo 'deb http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main' >> /etc/apt/sources.list && \
    echo 'deb-src http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main' >> /etc/apt/sources.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys C2518248EEA14886 && \
    apt-get update && \
    echo oracle-java${JAVA_VER}-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections && \
    apt-get install -y --force-yes --no-install-recommends oracle-java${JAVA_VER}-installer oracle-java${JAVA_VER}-set-default && \
    apt-get clean && \
    rm -rf /var/cache/oracle-jdk${JAVA_VER}-installer
RUN update-java-alternatives -s java-8-oracle
RUN echo "export JAVA_HOME=/usr/lib/jvm/java-8-oracle" >> ~/.bashrc

# Installs Ant.
ENV ANT_VERSION=1.10.2
ENV ANT_HOME=/opt/ant
# change to tmp folder
WORKDIR /tmp
# Download and extract apache ant to opt folder
RUN wget --no-check-certificate --no-cookies http://archive.apache.org/dist/ant/binaries/apache-ant-${ANT_VERSION}-bin.tar.gz \
    && wget --no-check-certificate --no-cookies http://archive.apache.org/dist/ant/binaries/apache-ant-${ANT_VERSION}-bin.tar.gz.md5 \
    && echo "$(cat apache-ant-${ANT_VERSION}-bin.tar.gz.md5) apache-ant-${ANT_VERSION}-bin.tar.gz" | md5sum -c \
    && tar -zvxf apache-ant-${ANT_VERSION}-bin.tar.gz -C /opt/ \
    && ln -s /opt/apache-ant-${ANT_VERSION} /opt/ant \
    && rm -f apache-ant-${ANT_VERSION}-bin.tar.gz \
    && rm -f apache-ant-${ANT_VERSION}-bin.tar.gz.md5
# add executables to path
RUN update-alternatives --install "/usr/bin/ant" "ant" "/opt/ant/bin/ant" 1 && \
    update-alternatives --set "ant" "/opt/ant/bin/ant" 

WORKDIR /workspace


# rsync -avzh ~/Documents/GraphEmbedding yba@qilin.cs.ucla.edu:/home/yba/
# docker build .
# docker tag 89c8df0682d5 yba_graphembedding
# nvidia-docker run -v /home/yba/GraphEmbedding:/workspace --env CUDA_VISIBLE_DEVICES=_____ -it yba_graphembedding bash
# nvidia-docker run -v /home/yba/GraphEmbedding:/workspace -it yba_graphembedding bash
# cd model/Siamese && python main.py
# pip install seaborn










































