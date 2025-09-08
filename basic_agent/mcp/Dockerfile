# Stage 1: Build
FROM node:22.12-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files and install dependencies first (leverages Docker cache)
COPY package*.json ./
RUN npm install

# Copy rest of the source code
COPY . .

# Build TypeScript
RUN npm run build

# Stage 2: Production image
FROM node:22-alpine

WORKDIR /app

# Copy built files + package.json & node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY package*.json ./

# Expose port (if your server listens on 3000, adjust if different)
EXPOSE 3000

# Start the app
CMD ["node", "dist/index.js"]
