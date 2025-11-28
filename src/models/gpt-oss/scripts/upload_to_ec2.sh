#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./upload_to_ec2.sh <EC2_IP_ADDRESS>"
    exit 1
fi

EC2_IP=$1
KEY_PATH=~/.ssh/code-reasoning-key.pem

echo "Creating directory structure on EC2..."
ssh -i $KEY_PATH ubuntu@$EC2_IP << 'EOSSH'
mkdir -p ~/code-reasoning-aws/src/models/{bedrock_claude,gpt_oss}
mkdir -p ~/code-reasoning-aws/logs
mkdir -p ~/code-reasoning-aws/experiments/results
EOSSH

echo "Uploading files..."

# Upload root files
scp -i $KEY_PATH run_all.sh requirements.txt ubuntu@$EC2_IP:~/code-reasoning-aws/

# Upload bedrock_claude module
scp -i $KEY_PATH src/models/bedrock_claude/*.py ubuntu@$EC2_IP:~/code-reasoning-aws/src/models/bedrock_claude/

# Upload gpt_oss module
scp -i $KEY_PATH src/models/gpt_oss/*.py ubuntu@$EC2_IP:~/code-reasoning-aws/src/models/gpt_oss/

echo ""
echo "✓ Upload complete!"
echo ""
echo "Next steps:"
echo "1. SSH in: ssh -i $KEY_PATH ubuntu@$EC2_IP"
echo "2. cd code-reasoning-aws"
echo "3. ./setup_aws.sh"
echo "4. screen -S eval"
echo "5. ./run_all.sh"
echo "6. Ctrl+A, D to detach"