echo "dropping database"
psql -d postgres  -c "DROP DATABASE text_and_images_ejeannot"

echo "Removing all buckets..."
echo "ej-bucket-1"
aws s3 rm s3://ej-bucket-1-c4e3b8ce14815fbc1b26b8677f0b2c14  --recursive 
aws s3 rb s3://ej-bucket-1-c4e3b8ce14815fbc1b26b8677f0b2c14 
echo "ej-bucket-2"
aws s3 rm s3://ej-bucket-2-c4e3b8ce14815fbc1b26b8677f0b2c14  --recursive 
aws s3 rb s3://ej-bucket-2-c4e3b8ce14815fbc1b26b8677f0b2c14 
