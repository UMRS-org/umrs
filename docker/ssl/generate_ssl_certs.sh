#!/bin/bash

# Script to generate self-signed SSL certificates for UMRS HTTPS support
# This is for development/testing purposes only

SSL_DIR="$(dirname "$0")"
cd "$SSL_DIR"

echo "Generating SSL certificates for UMRS..."

# Generate private key for MinIO
openssl genrsa -out private.key 2048

# Generate config file for SAN
cat > san.cnf <<EOF
[req]
distinguished_name=req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C=FR
ST=Lorraine
L=Nancy
O=INRIA
OU=UMRS
CN=umrs-server

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = umrs-server
DNS.2 = minio
DNS.3 = localhost
EOF

# Generate CSR and self-signed certificate for UMRS/MinIO with SANs
openssl req -new -key private.key -out public.csr -config san.cnf
openssl x509 -req -days 365 -in public.csr -signkey private.key -out public.crt -extensions v3_req -extfile san.cnf

# Clean up
rm -f public.csr san.cnf

# Set appropriate permissions
chmod 600 private.key
chmod 644 public.crt

echo "SSL certificates generated successfully!"
echo "Files created:"
echo "  - private.key (private key)"
echo "  - public.crt (certificate)"

echo "Note: These are self-signed certificates for development only."
echo "For production, use certificates from a trusted Certificate Authority."

# --- MinIO SSL certificate setup ---
# MinIO expects: public.crt (certificate), private.key (private key)
# Optionally, a CAs/ directory for custom CA chains


echo "MinIO SSL files ready: public.crt, private.key."

# Optionally create CAs/ directory if custom CA chains are needed
if [ ! -d "CAs" ]; then
	mkdir CAs
	echo "Created empty CAs/ directory for custom CA chains (optional)."
fi

echo "MinIO SSL files ready: public.crt, private.key."
