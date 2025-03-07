#!/bin/bash

# 사용법: ./gitpush.sh "커밋 메시지"
# 기본 메시지는 "Updated code with modifications" 입니다.

commit_message=${1:-"Updated code with modifications"}

echo "Adding changes..."
git add .

echo "Committing changes..."
git commit -m "$commit_message"

echo "Pushing to repository..."
git push origin main

echo "Done!"
