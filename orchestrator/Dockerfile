FROM node:20-bookworm-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY package.json package.json
RUN npm install --omit=dev
COPY src ./src

CMD ["npm", "run", "start"]
