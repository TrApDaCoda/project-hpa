# FROM gcc:latest
# COPY testcpp.cpp /app/
# WORKDIR /app
# RUN g++ -O3 -fopenmp -o testcpp testcpp.cpp
# ENTRYPOINT ["./testcpp"]

# FROM ubuntu:22.04

# # Install build dependencies
# RUN apt-get update \
#     && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
#        gcc libomp-dev build-essential \
#     && rm -rf /var/lib/apt/lists/*

# # Copy solver source and compile
# COPY powerplant.c /app/
# WORKDIR /app
# RUN gcc powerplant.c -O3 -march=native -mavx2 -fopenmp -o solver

# # Default entrypoint
# ENTRYPOINT ["/app/solver"]

# Dockerfile
FROM ubuntu:22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential libomp-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY testcpp.cpp /app/testcp.cpp

RUN g++ testcp.cpp -O3 -march=native -mavx2 -fopenmp -static -o testcp

ENTRYPOINT ["/app/testcp"]

