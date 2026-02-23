docker build --network host -t mcp-test .

docker run --rm -i --network host --cap-drop all --cap-add=sys_nice -v ./model-mcp-test.py:/app/test.py:ro mcp-test
