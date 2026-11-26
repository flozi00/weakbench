FROM mcr.microsoft.com/playwright:v1.65.1-noble@sha256:1507cbfc550a63a6b7396791ca4f00df17beb56342d2447b3f106c306f525579
LABEL org.weakbench.arena-render="1"
WORKDIR /opt/arena
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY render.cjs ./
ENV HOME=/tmp TZ=UTC LANG=C.UTF-8
USER 10001:10001
ENTRYPOINT ["node","/opt/arena/render.cjs"]
