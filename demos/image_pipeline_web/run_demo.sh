#!/bin/bash

# Build backend 
echo "Starting backend"
cd backend 
npm run runall &

# Run frontend
echo "Starting frontend"
python3 -m http.server 8085 --directory ../frontend



