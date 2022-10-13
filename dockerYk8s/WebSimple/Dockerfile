# Especificamos la imagen base
FROM node:14-alpine

WORKDIR /usr/app

# instalamos dependencias
COPY ./package.json ./
RUN npm install
COPY ./ ./

# Comandos por defecto
CMD ["npm", "start"]