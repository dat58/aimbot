FROM vtdat58/rust:1.86.0-cuda11.8.0-cudnn8-tensorrt8.6.1.6-onnxruntime1.21.1-opencv4.8.0-videoio-ndi-dylib AS chef
WORKDIR /app

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
COPY --from=chef /release/lib/* /usr/lib
COPY --from=chef /etc/ssl/certs /etc/ssl/certs
COPY --from=builder /app/target/release/aimbot .
ENTRYPOINT ["./aimbot"]