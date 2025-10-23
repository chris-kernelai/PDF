#!/bin/bash
################################################################################
# setup_iam_role.sh
#
# Creates IAM role for EC2 instances to access S3 and attaches to instances
#
# Usage:
#   ./setup_iam_role.sh create                    # Create role and instance profile
#   ./setup_iam_role.sh attach <instance-id>      # Attach role to instance
#   ./setup_iam_role.sh attach-all                # Attach to all PDF instances
#   ./setup_iam_role.sh check <instance-id>       # Check if instance has role
################################################################################

set -euo pipefail

export AWS_PROFILE="production"

ROLE_NAME="PDF-Pipeline-EC2-Role"
INSTANCE_PROFILE_NAME="PDF-Pipeline-EC2-Profile"
POLICY_NAME="PDF-Pipeline-S3-Access"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Instance IDs
INSTANCE_PDF="i-09f9f69a561efe64c"
INSTANCE_PDF2="i-04ee570e5bfab51d9"
INSTANCE_PDF3="i-0072a25219e846862"
INSTANCE_LONDON="i-0131ec0698e8c7bbf"

REGION_PDF="us-west-1"
REGION_PDF2="us-west-1"
REGION_PDF3="us-west-1"
REGION_LONDON="eu-west-2"

usage() {
    cat <<EOF
Usage: $0 <command> [args]

Commands:
  create                    Create IAM role and instance profile
  attach <instance-id>      Attach role to specific instance
  attach-all                Attach role to all PDF instances
  check <instance-id>       Check if instance has role attached
  
Examples:
  $0 create
  $0 attach i-04ee570e5bfab51d9
  $0 attach-all
  $0 check i-04ee570e5bfab51d9
EOF
}

create_role() {
    echo -e "${BLUE}Creating IAM role for EC2 instances...${NC}"
    
    # Trust policy for EC2
    TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}'
    
    # Check if role exists
    if aws iam get-role --role-name "$ROLE_NAME" &>/dev/null; then
        echo -e "${YELLOW}✓ Role $ROLE_NAME already exists${NC}"
    else
        echo "Creating role $ROLE_NAME..."
        aws iam create-role \
            --role-name "$ROLE_NAME" \
            --assume-role-policy-document "$TRUST_POLICY" \
            --description "Role for PDF pipeline EC2 instances to access S3"
        echo -e "${GREEN}✓ Role created${NC}"
    fi
    
    # S3 access policy
    S3_POLICY='{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::primer-production-librarian-documents",
        "arn:aws:s3:::primer-production-librarian-documents/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListAllMyBuckets"
      ],
      "Resource": "*"
    }
  ]
}'
    
    # Check if policy exists
    POLICY_ARN="arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/$POLICY_NAME"
    if aws iam get-policy --policy-arn "$POLICY_ARN" &>/dev/null; then
        echo -e "${YELLOW}✓ Policy $POLICY_NAME already exists${NC}"
    else
        echo "Creating policy $POLICY_NAME..."
        aws iam create-policy \
            --policy-name "$POLICY_NAME" \
            --policy-document "$S3_POLICY" \
            --description "S3 access for PDF pipeline"
        echo -e "${GREEN}✓ Policy created${NC}"
    fi
    
    # Attach policy to role
    echo "Attaching policy to role..."
    aws iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn "$POLICY_ARN" 2>/dev/null || echo "Policy already attached"
    echo -e "${GREEN}✓ Policy attached${NC}"
    
    # Create instance profile
    if aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" &>/dev/null; then
        echo -e "${YELLOW}✓ Instance profile $INSTANCE_PROFILE_NAME already exists${NC}"
    else
        echo "Creating instance profile $INSTANCE_PROFILE_NAME..."
        aws iam create-instance-profile \
            --instance-profile-name "$INSTANCE_PROFILE_NAME"
        echo -e "${GREEN}✓ Instance profile created${NC}"
        
        # Add role to instance profile
        echo "Adding role to instance profile..."
        sleep 2  # Wait for profile to be ready
        aws iam add-role-to-instance-profile \
            --instance-profile-name "$INSTANCE_PROFILE_NAME" \
            --role-name "$ROLE_NAME"
        echo -e "${GREEN}✓ Role added to instance profile${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}✅ IAM role setup complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Attach to instances: $0 attach-all"
    echo "  2. Verify: $0 check <instance-id>"
}

attach_role() {
    local INSTANCE_ID=$1
    local REGION=$2
    
    echo -e "${BLUE}Attaching IAM role to $INSTANCE_ID...${NC}"
    
    # Check if already has role
    HAS_ROLE=$(aws ec2 describe-instances \
        --region "$REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].IamInstanceProfile.Arn' \
        --output text 2>/dev/null || echo "None")
    
    if [ "$HAS_ROLE" != "None" ] && [ -n "$HAS_ROLE" ]; then
        echo -e "${YELLOW}✓ Instance already has IAM role: $HAS_ROLE${NC}"
        return 0
    fi
    
    # Attach the instance profile
    echo "Attaching instance profile..."
    if aws ec2 associate-iam-instance-profile \
        --region "$REGION" \
        --instance-id "$INSTANCE_ID" \
        --iam-instance-profile Name="$INSTANCE_PROFILE_NAME" 2>/dev/null; then
        echo -e "${GREEN}✓ IAM role attached to $INSTANCE_ID${NC}"
    else
        echo -e "${RED}✗ Failed to attach IAM role${NC}"
        return 1
    fi
}

check_role() {
    local INSTANCE_ID=$1
    local REGION=$2
    
    echo -e "${BLUE}Checking IAM role for $INSTANCE_ID...${NC}"
    
    HAS_ROLE=$(aws ec2 describe-instances \
        --region "$REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].IamInstanceProfile.Arn' \
        --output text 2>/dev/null || echo "None")
    
    if [ "$HAS_ROLE" != "None" ] && [ -n "$HAS_ROLE" ]; then
        echo -e "${GREEN}✓ Instance has IAM role: $HAS_ROLE${NC}"
        return 0
    else
        echo -e "${RED}✗ No IAM role attached${NC}"
        return 1
    fi
}

attach_all() {
    echo -e "${BLUE}Attaching IAM role to all PDF instances...${NC}"
    echo ""
    
    echo "PDF instance (us-west-1):"
    attach_role "$INSTANCE_PDF" "$REGION_PDF"
    echo ""
    
    echo "PDF-2 instance (us-west-1):"
    attach_role "$INSTANCE_PDF2" "$REGION_PDF2"
    echo ""
    
    echo "PDF-3 instance (us-west-1):"
    attach_role "$INSTANCE_PDF3" "$REGION_PDF3"
    echo ""
    
    echo "PDF-London instance (eu-west-2):"
    attach_role "$INSTANCE_LONDON" "$REGION_LONDON"
    echo ""
    
    echo -e "${GREEN}✅ Done attaching to all instances${NC}"
}

# Main command dispatcher
if [ $# -lt 1 ]; then
    usage
    exit 1
fi

COMMAND=$1
shift

case "$COMMAND" in
    create)
        create_role
        ;;
    attach)
        if [ $# -lt 1 ]; then
            echo "Error: attach requires instance-id"
            usage
            exit 1
        fi
        INSTANCE_ID=$1
        # Auto-detect region based on known instances
        case "$INSTANCE_ID" in
            "$INSTANCE_PDF")
                REGION="$REGION_PDF"
                ;;
            "$INSTANCE_PDF2")
                REGION="$REGION_PDF2"
                ;;
            "$INSTANCE_PDF3")
                REGION="$REGION_PDF3"
                ;;
            "$INSTANCE_LONDON")
                REGION="$REGION_LONDON"
                ;;
            *)
                echo "Unknown instance. Provide region:"
                read -p "Region (e.g., us-west-1): " REGION
                ;;
        esac
        attach_role "$INSTANCE_ID" "$REGION"
        ;;
    attach-all)
        attach_all
        ;;
    check)
        if [ $# -lt 1 ]; then
            echo "Error: check requires instance-id"
            usage
            exit 1
        fi
        INSTANCE_ID=$1
        # Auto-detect region
        case "$INSTANCE_ID" in
            "$INSTANCE_PDF")
                REGION="$REGION_PDF"
                ;;
            "$INSTANCE_PDF2")
                REGION="$REGION_PDF2"
                ;;
            "$INSTANCE_PDF3")
                REGION="$REGION_PDF3"
                ;;
            "$INSTANCE_LONDON")
                REGION="$REGION_LONDON"
                ;;
            *)
                read -p "Region (e.g., us-west-1): " REGION
                ;;
        esac
        check_role "$INSTANCE_ID" "$REGION"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac

