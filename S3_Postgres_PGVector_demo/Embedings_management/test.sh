
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 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" \
  -d '{
    "model": "gpt-4o-mini",
    "store": true,
    "messages": [
      {"role": "user", "content": "write a haiku about ai"}
    ]
  }'

