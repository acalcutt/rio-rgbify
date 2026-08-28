FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# ── Basics ───────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates lsb-release wget curl git \
    # Python
    zlib1g-dev libssl-dev libffi-dev python3 python3-pip python3-openssl \
    # GEOS build deps
    cmake \
    # proj build deps
    sqlite3 libsqlite3-dev libtiff-dev libcurl4-openssl-dev pkg-config \
    # GDAL extra format drivers
    libspatialite-dev libkml-dev libgeotiff-dev \
    && rm -rf /var/lib/apt/lists/*

# ── GEOS 3.10.2 ──────────────────────────────────────────────────────────────
RUN wget -q http://download.osgeo.org/geos/geos-3.10.2.tar.bz2 \
    && tar -xf geos-3.10.2.tar.bz2 \
    && cd geos-3.10.2 && ./configure && make -j$(nproc) && make install \
    && cd .. && rm -rf geos-3.10.2 geos-3.10.2.tar.bz2 \
    && ldconfig

# ── PROJ 7.2.1 ───────────────────────────────────────────────────────────────
RUN wget -q https://download.osgeo.org/proj/proj-7.2.1.tar.gz \
    && tar -xf proj-7.2.1.tar.gz \
    && cd proj-7.2.1 && ./configure && make -j$(nproc) && make install \
    && cd .. && rm -rf proj-7.2.1 proj-7.2.1.tar.gz \
    && ldconfig

# ── Apache Arrow ─────────────────────────────────────────────────────────────
RUN DISTRO=$(lsb_release --id --short | tr 'A-Z' 'a-z') \
    && CODENAME=$(lsb_release --codename --short) \
    && wget -q "https://apache.jfrog.io/artifactory/arrow/${DISTRO}/apache-arrow-apt-source-latest-${CODENAME}.deb" \
    && apt-get install -y -V "./apache-arrow-apt-source-latest-${CODENAME}.deb" \
    && apt-get update \
    && apt-get install -y -V \
        libarrow-dev \
        libarrow-glib-dev \
        libarrow-dataset-dev \
        libarrow-dataset-glib-dev \
        libarrow-acero-dev \
        libparquet-dev \
        libparquet-glib-dev \
    && rm -f "apache-arrow-apt-source-latest-${CODENAME}.deb" \
    && rm -rf /var/lib/apt/lists/*

# ── GDAL 3.8.4 ───────────────────────────────────────────────────────────────
RUN wget -q https://github.com/OSGeo/gdal/releases/download/v3.8.4/gdal-3.8.4.tar.gz \
    && tar -xf gdal-3.8.4.tar.gz \
    && cd gdal-3.8.4 && mkdir build && cd build \
    && cmake .. && cmake --build . -j$(nproc) && cmake --build . --target install \
    && cd ../.. && rm -rf gdal-3.8.4 gdal-3.8.4.tar.gz \
    && ldconfig

# ── rio-rgbify ───────────────────────────────────────────────────────────────
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --break-system-packages .

# /data — mount your input/output raster files here
RUN mkdir -p /data
VOLUME /data

WORKDIR /data

ENTRYPOINT ["rio"]
