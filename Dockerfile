FROM trax-base:latest

ADD . /opt/source

# Install build & basic dependencies
RUN apt-get -y update \
    && apt-get install -y build-essential cmake \
    && apt-get install -y libopencv-dev \
    && cd /opt/source && mkdir build && cd build \
    && cmake .. && make  && cp mil /usr/bin/mil && cd .. && rm -rf build \
    && apt-get remove -y build-essential cmake libopencv-dev \
    && apt-get -y autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/

CMD ["/usr/bin/mil"]
