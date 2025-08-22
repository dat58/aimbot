FROM vtdat58/rust:1.86.0-cuda11.8.0-cudnn8-tensorrt8.6.1.6-onnxruntime1.21.1-opencv4.8.0-videoio-ndi-dylib AS chef
WORKDIR /app

FROM chef AS copier
RUN mkdir -p /release/lib
RUN --mount=src=/,dst=/artifacts,from=chef \
    cp -L /artifacts/usr/lib/x86_64-linux-gnu/libcap.so.2 /release/lib && \
    cp -L /artifacts/usr/lib/x86_64-linux-gnu/libsystemd.so.0 /release/lib && \
    cp -L /artifacts/usr/lib/x86_64-linux-gnu/libdbus-1.so.3 /release/lib && \
    cp -L /artifacts/usr/lib/x86_64-linux-gnu/libavahi-client.so.3 /release/lib && \
    cp -L /artifacts/usr/lib/x86_64-linux-gnu/libavahi-common.so.3 /release/lib && \
    cp -L /artifacts/usr/lib/libndi.so.6 /release/lib && \
    cp -L /artifacts/usr/lib/x86_64-linux-gnu/libndi.so.4 /release/lib

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --release

FROM gcr.io/distroless/base-debian12:latest
WORKDIR /app
ENV TZ=Asia/Ho_Chi_Minh \
    SSL_CERT_DIR=/etc/ssl/certs
COPY --from=copier /release/lib/* /usr/lib
COPY --from=chef /etc/ssl/certs /etc/ssl/certs
COPY --from=builder /app/target/release/aimbot .
ENTRYPOINT ["./aimbot"]