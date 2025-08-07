#!/bin/bash

# Script to generate self-signed SSL certificates for UMRS HTTPS support
# This is for development/testing purposes only

SSL_DIR="$(dirname "$0")"
cd "$SSL_DIR"

echo "Generating SSL certificates for UMRS..."

# Generate private key
openssl genrsa -out server.key 2048

# Generate certificate signing request
openssl req -new -key server.key -out server.csr -subj "/C=FR/ST=Lorraine/L=Nancy/O=INRIA/OU=UMRS/CN=umrs-server"

# Generate self-signed certificate
openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt

# Create combined certificate file
cat server.crt server.key > server.pem

# Set appropriate permissions
chmod 600 server.key server.pem
chmod 644 server.crt

echo "SSL certificates generated successfully!"
echo "Files created:"
echo "  - server.key (private key)"
echo "  - server.crt (certificate)"
echo "  - server.pem (combined certificate and key)"
echo "  - server.csr (certificate signing request)"

# Clean up CSR file
rm server.csr

echo "Note: These are self-signed certificates for development only."
echo "For production, use certificates from a trusted Certificate Authority."
