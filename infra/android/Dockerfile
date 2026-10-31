FROM node:22-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    ANDROID_HOME=/opt/android-sdk \
    ANDROID_SDK_ROOT=/opt/android-sdk \
    EXPO_NO_TELEMETRY=1 \
    CI=1
ENV PATH="${JAVA_HOME}/bin:${ANDROID_HOME}/cmdline-tools/latest/bin:${ANDROID_HOME}/platform-tools:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
      openjdk-17-jdk-headless ca-certificates curl unzip git python3 make g++ \
    && rm -rf /var/lib/apt/lists/*
ARG ANDROID_COMMAND_TOOLS=13114758
RUN mkdir -p /opt/android-sdk/cmdline-tools \
    && curl --fail --location --retry 3 "https://dl.google.com/android/repository/commandlinetools-linux-${ANDROID_COMMAND_TOOLS}_latest.zip" -o /tmp/android-tools.zip \
    && unzip -q /tmp/android-tools.zip -d /opt/android-sdk/cmdline-tools \
    && mv /opt/android-sdk/cmdline-tools/cmdline-tools /opt/android-sdk/cmdline-tools/latest \
    && rm /tmp/android-tools.zip \
    && yes | sdkmanager --licenses >/dev/null
RUN sdkmanager 'platform-tools' 'platforms;android-36' 'build-tools;36.0.0' 'ndk;27.1.12297006' 'cmake;3.22.1'
COPY build.mjs /opt/asl/build.mjs
WORKDIR /workspace
ENTRYPOINT ["node", "/opt/asl/build.mjs"]
